# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PyTorch Gemma 4 vision encoder for patch-grid inputs."""

import dataclasses
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .gemma4_model import Gemma4RMSNorm


@dataclasses.dataclass
class Gemma4VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    head_dim: int = 64
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))["vision_config"]
        return cls(
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            head_dim=data["head_dim"],
            patch_size=data["patch_size"],
            position_embedding_size=data["position_embedding_size"],
            pooling_kernel_size=data["pooling_kernel_size"],
            rms_norm_eps=data["rms_norm_eps"],
        )


class Gemma4ClippedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("input_min", torch.tensor(-float("inf")))
        self.register_buffer("input_max", torch.tensor(float("inf")))
        self.register_buffer("output_min", torch.tensor(-float("inf")))
        self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        return torch.clamp(hidden_states, self.output_min, self.output_max)


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(3 * config.patch_size ** 2, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size)
        )

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2 * (pixel_values - 0.5)
        pixel_values = pixel_values.to(self.input_proj.weight.dtype)
        hidden_states = self.input_proj(pixel_values)
        positions = pixel_position_ids.clamp(min=0)
        position_embeddings = (
            F.embedding(positions[..., 0], self.position_embedding_table[0])
            + F.embedding(positions[..., 1], self.position_embedding_table[1])
        )
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        return hidden_states + position_embeddings


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        size = config.hidden_size
        self.config = config
        self.q_proj = Gemma4ClippedLinear(size, size)
        self.k_proj = Gemma4ClippedLinear(size, size)
        self.v_proj = Gemma4ClippedLinear(size, size)
        self.o_proj = Gemma4ClippedLinear(size, size)
        self.q_norm = Gemma4RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(config.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(config.head_dim, config.rms_norm_eps, with_scale=False)

    def _rope(self, states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # Split the 64-dimensional head into independent x and y rotary halves.
        pieces = []
        axis_dim = states.shape[-1] // 2
        half = axis_dim // 2
        inv_freq = 1.0 / (100.0 ** (
            torch.arange(0, axis_dim, 2, device=states.device).float() / axis_dim
        ))
        for axis in range(2):
            part = states[..., axis * axis_dim:(axis + 1) * axis_dim]
            positions = position_ids[..., axis].float()
            freqs = positions[..., None] * inv_freq
            cos = freqs.cos().unsqueeze(2)
            sin = freqs.sin().unsqueeze(2)
            first, second = part[..., :half], part[..., half:]
            pieces.append(torch.cat((first * cos - second * sin,
                                     first * sin + second * cos), dim=-1))
        return torch.cat(pieces, dim=-1).to(dtype=states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        shape = (batch, length, self.config.num_attention_heads,
                 self.config.head_dim)
        q = self._rope(self.q_norm(self.q_proj(hidden_states).view(shape)), position_ids)
        k = self._rope(self.k_norm(self.k_proj(hidden_states).view(shape)), position_ids)
        v = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        # Gemma 4 uses RMS-normalized q/k and an attention scaling of 1.0.
        scores = torch.matmul(q, k.transpose(-1, -2))
        allowed = valid_mask[:, None, None, :] & valid_mask[:, None, :, None]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = weights @ v
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.gate_proj = Gemma4ClippedLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = Gemma4ClippedLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = Gemma4ClippedLinear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(hidden_states), approximate="tanh")
            * self.up_proj(hidden_states)
        )


class Gemma4VisionLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Gemma4VisionAttention(config)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Gemma4VisionMLP(config)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_ids, valid_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.post_attention_layernorm(
            self.self_attn(hidden_states, position_ids, valid_mask)
        )
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig | None = None):
        super().__init__()
        self.config = config or Gemma4VisionConfig()
        c = self.config
        self.patch_embedder = Gemma4VisionPatchEmbedder(c)
        self.layers = nn.ModuleList([Gemma4VisionLayer(c) for _ in range(c.num_hidden_layers)])

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int | None = None,
    ) -> torch.Tensor:
        """Encode flattened patches.

        Args:
            pixel_values: ``[batch, patches, 3 * patch_size**2]``.
            pixel_position_ids: ``[batch, patches, 2]``; ``(-1, -1)`` pads.
            output_length: number of pooled soft tokens. Defaults to one 3x3 pool.
        """
        output_dtype = pixel_values.dtype
        padding = (pixel_position_ids == -1).all(dim=-1)
        positions = pixel_position_ids.clamp(min=0)
        # Padding is only a batching convenience. Trim it before attention so
        # padded patches do not create invalid all-masked softmax rows.
        if padding.any():
            valid_length = int((~padding[0]).sum())
            pixel_values = pixel_values[:, :valid_length]
            pixel_position_ids = pixel_position_ids[:, :valid_length]
            padding = padding[:, :valid_length]
            positions = pixel_position_ids.clamp(min=0)
        x = self.patch_embedder(pixel_values, pixel_position_ids, padding)
        for layer in self.layers:
            x = layer(x, pixel_position_ids, ~padding)
        x = x.masked_fill(padding[..., None], 0.0)

        if output_length is None:
            output_length = max(1, x.shape[1] // (self.config.pooling_kernel_size ** 2))
        if x.shape[1] != output_length:
            kernel = self.config.pooling_kernel_size
            if x.shape[1] != output_length * kernel * kernel:
                raise ValueError("patch count must be output_length times pool area")
            # Pool spatially by (x, y) coordinates. Flattened patches are
            # row-major, so a plain reshape would mix neighboring rows.
            clamped = positions
            max_x = clamped[..., 0].max(dim=-1, keepdim=True).values + 1
            kernel_index = clamped // kernel
            kernel_index = kernel_index[..., 0] + (max_x // kernel) * kernel_index[..., 1]
            weights = F.one_hot(kernel_index.long(), output_length).float() / (kernel * kernel)
            x = weights.transpose(1, 2) @ x.float()
        return (x * self.config.hidden_size ** 0.5).to(output_dtype)
