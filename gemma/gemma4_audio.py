# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PyTorch Gemma 4 E2B audio encoder."""

import dataclasses
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .gemma4_model import Gemma4RMSNorm
from .gemma4_vision import Gemma4ClippedLinear


@dataclasses.dataclass
class Gemma4AudioConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    conv_kernel_size: int = 5
    rms_norm_eps: float = 1e-6
    output_proj_dims: int = 1536
    residual_weight: float = 0.5
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1e9
    subsampling_conv_channels: tuple[int, int] = (128, 32)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))["audio_config"]
        return cls(
            hidden_size=data["hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            intermediate_size=data["hidden_size"] * 4,
            attention_chunk_size=data["attention_chunk_size"],
            attention_context_left=data["attention_context_left"],
            attention_context_right=data["attention_context_right"],
            conv_kernel_size=data["conv_kernel_size"],
            rms_norm_eps=data["rms_norm_eps"],
            output_proj_dims=data["output_proj_dims"],
            residual_weight=data["residual_weight"],
            attention_logit_cap=data["attention_logit_cap"],
            attention_invalid_logits_value=data["attention_invalid_logits_value"],
            subsampling_conv_channels=tuple(data["subsampling_conv_channels"]),
        )


class Gemma4AudioRelPositionalEncoding(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        context = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        count = config.hidden_size // 2
        increment = math.log(10000.0) / max(count - 1, 1)
        inv = torch.exp(torch.arange(count) * -increment)
        self.register_buffer("inv_timescales", inv.view(1, 1, -1), persistent=False)
        self.context_size = context

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.context_size // 2, -1, -1, device=hidden_states.device).view(-1, 1)
        scaled = positions * self.inv_timescales.to(hidden_states.device)
        return torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=-1).to(hidden_states.dtype)


class Gemma4AudioSubsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=eps, elementwise_affine=True, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1))
        return F.relu(hidden_states).permute(0, 3, 1, 2).contiguous()


class Gemma4AudioSubsample(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, feature_size: int = 128):
        super().__init__()
        c0, c1 = config.subsampling_conv_channels
        self.layer0 = Gemma4AudioSubsampleLayer(1, c0, config.rms_norm_eps)
        self.layer1 = Gemma4AudioSubsampleLayer(c0, c1, config.rms_norm_eps)
        self.input_proj_linear = nn.Linear((c0 // 4) * c1, config.hidden_size, bias=False)
        self.feature_size = feature_size

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        hidden_states = self.layer0(hidden_states)
        hidden_states = self.layer1(hidden_states)
        batch, _, length, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, length, -1)
        return self.input_proj_linear(hidden_states)


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.ffw_layer_1 = Gemma4ClippedLinear(config.hidden_size, config.hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippedLinear(config.hidden_size * 4, config.hidden_size)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = F.silu(self.ffw_layer_1(hidden_states))
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = self.post_layer_norm(hidden_states)
        return residual + self.configured_scale(hidden_states)

    def configured_scale(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 0.5


class Gemma4AudioLightConv(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        h = config.hidden_size
        self.linear_start = Gemma4ClippedLinear(h, h * 2)
        self.linear_end = Gemma4ClippedLinear(h, h)
        self.depthwise_conv1d = nn.Conv1d(h, h, config.conv_kernel_size,
                                          groups=h, bias=False)
        self.pre_layer_norm = Gemma4RMSNorm(h, config.rms_norm_eps)
        self.conv_norm = Gemma4RMSNorm(h, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.glu(self.linear_start(self.pre_layer_norm(hidden_states)), dim=-1)
        left_pad = self.depthwise_conv1d.kernel_size[0] - 1
        hidden_states = F.pad(hidden_states.transpose(1, 2), (left_pad, 0))
        hidden_states = self.depthwise_conv1d(hidden_states).transpose(1, 2)
        hidden_states = F.silu(self.conv_norm(hidden_states))
        return residual + self.linear_end(hidden_states)


class Gemma4AudioAttention(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        h = config.hidden_size
        self.config = config
        self.head_dim = h // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_proj = Gemma4ClippedLinear(h, h)
        self.k_proj = Gemma4ClippedLinear(h, h)
        self.v_proj = Gemma4ClippedLinear(h, h)
        self.post = Gemma4ClippedLinear(h, h)
        self.relative_k_proj = nn.Linear(h, h, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.softcap = config.attention_logit_cap
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

    def _blocks(self, states):
        blocks = (states.shape[1] + self.chunk_size - 1) // self.chunk_size
        padded = blocks * self.chunk_size
        states = F.pad(states, (0, 0, 0, 0, 0, padded - states.shape[1]))
        return states.reshape(states.shape[0], blocks, self.chunk_size, self.num_heads, self.head_dim)

    def _contexts(self, states):
        states = F.pad(states, (0, 0, 0, 0, self.max_past_horizon,
                                self.max_future_horizon + self.chunk_size - 1))
        return torch.movedim(states.unfold(1, self.context_size, self.chunk_size), -1, 2).contiguous()

    def _relative_shift(self, values):
        blocks, chunk, positions = values.shape[2:]
        values = F.pad(values, (0, self.context_size + 1 - positions))
        values = values.view(values.shape[0], values.shape[1], blocks, chunk * (self.context_size + 1))
        return values[..., :chunk * self.context_size].view(
            values.shape[0], values.shape[1], blocks, chunk, self.context_size
        )

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).float().view(shape)
        k = self.k_proj(hidden_states).float().view(shape)
        v = self.v_proj(hidden_states).float().view(shape)
        q = q * ((self.head_dim ** -0.5) / torch.log(torch.tensor(2., device=q.device)))
        q = q * F.softplus(self.per_dim_scale)
        k = k * (torch.log1p(torch.exp(torch.tensor(1., device=k.device))) / torch.log(torch.tensor(2., device=k.device)))
        qb = self._blocks(q); kc = self._contexts(k); vc = self._contexts(v)
        blocks = qb.shape[1]
        relative = self.relative_k_proj(position_embeddings).float().view(-1, self.num_heads, self.head_dim)
        queries = qb.permute(0, 3, 1, 2, 4)
        scores = queries @ kc.permute(0, 3, 1, 4, 2)
        rel = (queries.reshape(batch, self.num_heads, -1, self.head_dim)
               @ relative.permute(1, 2, 0))
        rel = rel.reshape(batch, self.num_heads, blocks, self.chunk_size, -1)
        scores = self._relative_shift(rel) + scores
        scores = torch.tanh(scores / self.softcap) * self.softcap
        qpos = torch.arange(blocks * self.chunk_size, device=hidden_states.device)
        offsets = torch.arange(self.context_size, device=hidden_states.device)
        kpos = qpos[:, None] - self.max_past_horizon + offsets[None, :]
        allowed = (kpos >= 0) & (kpos <= qpos[:, None] + self.max_future_horizon)
        allowed = allowed.reshape(blocks, self.chunk_size, self.context_size)
        scores = scores.masked_fill(~allowed[None, None], -1e9)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(vc.dtype)
        output = (weights @ vc.permute(0, 3, 1, 2, 4)).permute(0, 2, 3, 1, 4)
        output = output.reshape(batch, blocks * self.chunk_size, -1)[:, :length]
        return self.post(output.to(hidden_states.dtype))


class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config)
        self.lconv1d = Gemma4AudioLightConv(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        hidden_states = self.self_attn(self.norm_pre_attn(hidden_states), position_embeddings)
        hidden_states = residual + self.norm_post_attn(hidden_states)
        hidden_states = self.lconv1d(hidden_states)
        return self.norm_out(self.feed_forward2(hidden_states))


class Gemma4AudioModel(nn.Module):
    def __init__(self, config: Gemma4AudioConfig | None = None, feature_size: int = 128):
        super().__init__()
        self.config = config or Gemma4AudioConfig()
        self.subsample_conv_projection = Gemma4AudioSubsample(self.config, feature_size)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(self.config)
        self.layers = nn.ModuleList([
            Gemma4AudioLayer(self.config)
            for _ in range(self.config.num_hidden_layers)
        ])
        self.output_proj = nn.Linear(
            self.config.hidden_size, self.config.output_proj_dims
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.subsample_conv_projection(input_features)
        position_embeddings = self.rel_pos_enc(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return self.output_proj(hidden_states)
