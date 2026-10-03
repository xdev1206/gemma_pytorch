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
    gradient_clipping: float = 1e10
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
            gradient_clipping=data.get("gradient_clipping", 1e10),
            subsampling_conv_channels=tuple(data["subsampling_conv_channels"]),
        )


class Gemma4AudioRelPositionalEncoding(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        context = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        count = config.hidden_size // 2
        min_timescale = 1.0
        max_timescale = 10000.0
        increment = math.log(max_timescale / min_timescale) / max(count - 1, 1)
        inv = min_timescale * torch.exp(
            torch.arange(count, dtype=torch.float32) * -increment
        )
        inv = inv.to(dtype=torch.get_default_dtype())
        self.register_buffer("inv_timescales", inv.view(1, 1, -1), persistent=False)
        self.context_size = context

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            self.context_size // 2, -1, -1, device=hidden_states.device
        )
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


class Gemma4AudioSubsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=eps, elementwise_affine=True, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is not None:
            hidden_states = hidden_states * mask[:, None, :, None]
        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1))
        hidden_states = F.relu(hidden_states).permute(0, 3, 1, 2).contiguous()
        if mask is not None:
            mask = mask[:, ::2]
        return hidden_states, mask


class Gemma4AudioSubsample(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, feature_size: int = 128):
        super().__init__()
        c0, c1 = config.subsampling_conv_channels
        self.layer0 = Gemma4AudioSubsampleLayer(1, c0, config.rms_norm_eps)
        self.layer1 = Gemma4AudioSubsampleLayer(c0, c1, config.rms_norm_eps)
        self.input_proj_linear = nn.Linear((c0 // 4) * c1, config.hidden_size, bias=False)
        self.feature_size = feature_size

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = input_features.unsqueeze(1)
        hidden_states, mask = self.layer0(hidden_states, attention_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)
        batch, _, length, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, length, -1)
        hidden_states = self.input_proj_linear(hidden_states)
        if attention_mask is None:
            return hidden_states
        return hidden_states, mask


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
        self.ffw_layer_1 = Gemma4ClippedLinear(config.hidden_size, config.hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippedLinear(config.hidden_size * 4, config.hidden_size)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        limit = min(self.config.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        hidden_states = torch.clamp(hidden_states, -limit, limit)
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = F.silu(self.ffw_layer_1(hidden_states))
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -limit, limit)
        hidden_states = self.post_layer_norm(hidden_states)
        return residual + hidden_states * self.config.residual_weight

class Gemma4AudioLightConv(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
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
        limit = min(self.config.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        hidden_states = torch.clamp(hidden_states, -limit, limit)
        hidden_states = F.silu(self.conv_norm(hidden_states))
        return residual + self.linear_end(hidden_states)


class Gemma4AudioAttention(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        h = config.hidden_size
        self.config = config
        self.head_dim = h // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_scale = (self.head_dim ** -0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.q_proj = Gemma4ClippedLinear(h, h)
        self.k_proj = Gemma4ClippedLinear(h, h)
        self.v_proj = Gemma4ClippedLinear(h, h)
        self.post = Gemma4ClippedLinear(h, h)
        self.relative_k_proj = nn.Linear(h, h, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.register_buffer(
            "softcap",
            torch.tensor(config.attention_logit_cap),
            persistent=False,
        )
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).float().view(shape)
        k = self.k_proj(hidden_states).float().view(shape)
        v = self.v_proj(hidden_states).float().view(shape)
        q = q * self.q_scale
        q = q * F.softplus(self.per_dim_scale)
        k = k * self.k_scale
        qb = self._blocks(q); kc = self._contexts(k); vc = self._contexts(v)
        blocks = qb.shape[1]
        relative = self.relative_k_proj(position_embeddings).float().view(-1, self.num_heads, self.head_dim)
        queries = qb.permute(0, 3, 1, 2, 4)
        scores = queries @ kc.permute(0, 3, 1, 4, 2)
        rel = (queries.reshape(batch, self.num_heads, -1, self.head_dim)
               @ relative.permute(1, 2, 0))
        rel = rel.reshape(batch, self.num_heads, blocks, self.chunk_size, -1)
        scores = self._relative_shift(rel) + scores
        scores = scores / self.softcap
        scores = torch.tanh(scores)
        scores = scores * self.softcap
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=scores.device)
            scores = scores.masked_fill(
                attention_mask.logical_not(),
                self.config.attention_invalid_logits_value,
            )
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(vc.dtype)
        output = (weights @ vc.permute(0, 3, 1, 2, 4)).permute(0, 2, 3, 1, 4)
        output = output.reshape(batch, blocks * self.chunk_size, -1)
        output = output[:, :length].contiguous()
        return self.post(output.to(hidden_states.dtype))


class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config)
        self.lconv1d = Gemma4AudioLightConv(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        limit = min(self.config.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        hidden_states = torch.clamp(hidden_states, -limit, limit)
        hidden_states = self.self_attn(
            self.norm_pre_attn(hidden_states), position_embeddings, attention_mask
        )
        hidden_states = torch.clamp(hidden_states, -limit, limit)
        hidden_states = residual + self.norm_post_attn(hidden_states)
        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -limit, limit)
        return self.norm_out(hidden_states)


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
            self.config.hidden_size, self.config.output_proj_dims, bias=True
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        subsampled = self.subsample_conv_projection(input_features, attention_mask)
        if attention_mask is None:
            hidden_states = subsampled
            output_mask = torch.ones(
                hidden_states.shape[0], hidden_states.shape[1],
                dtype=torch.bool, device=hidden_states.device
            )
        else:
            hidden_states, output_mask = subsampled
        position_embeddings = self.rel_pos_enc(hidden_states)
        length = hidden_states.shape[1]
        blocks = (length + self.config.attention_chunk_size - 1) // self.config.attention_chunk_size
        padded = blocks * self.config.attention_chunk_size
        valid = output_mask.to(device=hidden_states.device, dtype=torch.bool)
        valid = valid[:, :length]
        context_size = (
            self.config.attention_chunk_size
            + self.config.attention_context_left - 1
            + self.config.attention_context_right
        )
        query = torch.arange(length, device=hidden_states.device)[:, None]
        key = torch.arange(length, device=hidden_states.device)[None, :]
        distance = query - key
        left_mask = (distance >= 0) & (
            distance < self.config.attention_context_left - 1
        )
        right_mask = (distance < 0) & (
            -distance < self.config.attention_context_right
        )
        standard = (left_mask | right_mask)[None, None]
        standard = standard & valid[:, None, None, :]
        standard = torch.where(
            standard,
            torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.finfo(hidden_states.dtype).min,
        )
        standard = F.pad(
            standard, (0, padded - length, 0, padded - length), value=False
        )
        standard = standard.reshape(
            standard.shape[0], 1, blocks, self.config.attention_chunk_size, padded
        )
        standard = F.pad(
            standard,
            (self.config.attention_context_left - 1, self.config.attention_context_right),
            value=False,
        )
        block_starts = torch.arange(blocks, device=hidden_states.device)
        block_starts = block_starts * self.config.attention_chunk_size
        offsets = torch.arange(context_size, device=hidden_states.device)
        key_indices = block_starts[:, None] + offsets[None, :]
        key_indices = key_indices[None, None, :, None, :].expand(
            standard.shape[0], 1, blocks, self.config.attention_chunk_size, context_size
        )
        attention_mask = torch.gather(standard, -1, key_indices)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        return self.output_proj(hidden_states)
