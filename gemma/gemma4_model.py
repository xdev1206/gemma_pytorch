# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PyTorch Gemma 4 E2B text backbone.

This module implements the dense language path first. Vision and audio towers
are intentionally separate because their inputs and checkpoint contracts are
different from text generation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .gemma4_config import Gemma4TextConfig


class Gemma4RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        if self.with_scale:
            normalized = normalized * self.weight.float()
        return normalized.type_as(hidden_states)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.layer_type = config.layer_types[layer_index]
        self.head_dim = config.layer_head_dim(layer_index)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared = layer_index >= first_shared >= 0
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        if not self.is_kv_shared:
            self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        if not self.is_kv_shared:
            self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)

        self.rotary_dim = self.head_dim if self.layer_type == "sliding_attention" else self.head_dim // 4

    def _apply_rope(self, states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        theta = 10_000.0 if self.layer_type == "sliding_attention" else 1_000_000.0
        half_dim = self.head_dim // 2
        rotated_pairs = self.rotary_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rotated_pairs, 2,
                                                   device=states.device).float() / self.head_dim))
        if rotated_pairs < half_dim:
            inv_freq = torch.cat((inv_freq, torch.zeros(
                half_dim - rotated_pairs, device=states.device)))
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        rotate_half = torch.cat((-states[..., self.head_dim // 2:],
                                 states[..., :self.head_dim // 2]), dim=-1)
        return (states * cos + rotate_half * sin).to(dtype=states.dtype)

    def forward(self, hidden_states: torch.Tensor, shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
        positions = torch.arange(length, device=hidden_states.device).float()
        q = self._apply_rope(self.q_norm(q), positions)
        if self.is_kv_shared:
            k, v = shared_kv[self.layer_type]
        else:
            k = self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
            k = self._apply_rope(self.k_norm(k), positions)
            v = self.v_norm(v)
            # The last non-shared layer of each attention type supplies later layers.
            first_shared = self.config.num_hidden_layers - self.config.num_kv_shared_layers
            future_types = self.config.layer_types[first_shared:]
            if first_shared > 0 and self.layer_index == max(
                i for i in range(first_shared) if self.config.layer_types[i] == self.layer_type
            ):
                shared_kv[self.layer_type] = (k.transpose(1, 2), v.transpose(1, 2))
        if not self.is_kv_shared:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        q = q.transpose(1, 2)
        # Gemma 4 uses RMS-normalized q/k and an attention scaling of 1.0.
        scores = torch.matmul(q, k.transpose(-1, -2))
        allowed = torch.ones((length, length), dtype=torch.bool, device=hidden_states.device).tril()
        if self.layer_type == "sliding_attention":
            positions = torch.arange(length, device=hidden_states.device)
            allowed &= positions[None, :] >= positions[:, None] - self.config.sliding_window + 1
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        output = torch.softmax(scores.float(), dim=-1) @ v.float()
        output = output.to(dtype=q.dtype)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_index: int):
        super().__init__()
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Gemma4Attention(config, layer_index)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Gemma4MLP(config, layer_index)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.per_layer_input_gate = nn.Linear(config.hidden_size, config.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(config.hidden_size_per_layer_input, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor,
                shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.post_attention_layernorm(self.self_attn(hidden_states, shared_kv))
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = F.gelu(
            self.per_layer_input_gate(hidden_states), approximate="tanh"
        )
        hidden_states = self.per_layer_projection(hidden_states * per_layer_input)
        return (residual + self.post_per_layer_input_norm(hidden_states)) * self.layer_scalar


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_index: int):
        super().__init__()
        first_shared_layer = config.num_hidden_layers - config.num_kv_shared_layers
        is_double_wide = (
            getattr(config, "use_double_wide_mlp", False)
            and layer_index >= first_shared_layer > 0
        )
        intermediate_size = config.intermediate_size * (2 if is_double_wide else 1)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(hidden_states), approximate="tanh")
            * self.up_proj(hidden_states)
        )


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = Gemma4RMSNorm(
            config.hidden_size_per_layer_input, config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids) * self.config.hidden_size ** 0.5
        hidden_states = inputs_embeds
        context_ple = self.per_layer_model_projection(hidden_states) * self.config.hidden_size ** -0.5
        context_ple = context_ple.view(
            *hidden_states.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        context_ple = self.per_layer_projection_norm(context_ple)
        if per_layer_inputs is None and input_ids is not None:
            token_ple = self.embed_tokens_per_layer(input_ids)
            token_ple = token_ple * self.config.hidden_size_per_layer_input ** 0.5
            token_ple = token_ple.view(
                *input_ids.shape,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            )
            per_layer_inputs = token_ple
        if per_layer_inputs is not None:
            per_layer_inputs = (per_layer_inputs + context_ple) * (2.0 ** -0.5)
        else:
            per_layer_inputs = context_ple
        shared_kv = {}
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, per_layer_inputs[:, :, i, :], shared_kv)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)

    def forward(self, input_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                per_layer_inputs: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
        )
        logits = torch.matmul(hidden_states, self.model.embed_tokens.weight.t())
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def load_weights(self, model_path: str) -> None:
        from .checkpoint import load_gemma4_text_safetensors
        load_gemma4_text_safetensors(self, model_path)


class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.norm = Gemma4RMSNorm(vision_hidden_size, 1e-6, with_scale=False)
        self.embedding_projection = nn.Linear(
            vision_hidden_size, text_hidden_size, bias=False
        )

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.norm(vision_embeddings))


class Gemma4ForConditionalGeneration(nn.Module):
    """Gemma 4 text decoder with image soft-token insertion."""

    def __init__(self, text_config, vision_config=None, audio_config=None):
        super().__init__()
        from .gemma4_vision import Gemma4VisionModel
        from .gemma4_audio import Gemma4AudioModel

        self.config = text_config
        self.model = Gemma4TextModel(text_config)
        self.vision_tower = Gemma4VisionModel(vision_config) if vision_config else None
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_config.hidden_size, text_config.hidden_size
        ) if vision_config else None
        self.audio_tower = Gemma4AudioModel(audio_config) if audio_config else None
        self.embed_audio = Gemma4MultimodalEmbedder(
            audio_config.output_proj_dims, text_config.hidden_size
        ) if audio_config else None

    @classmethod
    def from_pretrained(cls, model_path: str, dtype: torch.dtype = torch.bfloat16,
                        device: str | torch.device | None = None):
        import json
        from pathlib import Path
        from .gemma4_audio import Gemma4AudioConfig
        from .gemma4_vision import Gemma4VisionConfig

        config_path = Path(model_path) / "config.json"
        text_config = Gemma4TextConfig.from_json(config_path)
        vision_config = Gemma4VisionConfig.from_json(config_path)
        audio_config = Gemma4AudioConfig.from_json(config_path)
        previous_dtype = torch.get_default_dtype()
        previous_device = torch.get_default_device()
        torch.set_default_dtype(dtype)
        if device is not None:
            torch.set_default_device(device)
        try:
            model = cls(text_config, vision_config, audio_config)
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.set_default_device(previous_device)
        model.load_weights(model_path)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        image_patches: torch.Tensor | None = None,
        pixel_position_ids: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        audio_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids) * self.config.hidden_size ** 0.5
        if image_patches is not None:
            if pixel_position_ids is None or image_token_mask is None:
                raise ValueError("image positions and image token mask are required")
            image_embeddings = self.vision_tower(image_patches, pixel_position_ids)
            image_embeddings = self.embed_vision(image_embeddings).reshape(-1, self.config.hidden_size)
            indices = image_token_mask.reshape(-1).nonzero(as_tuple=False).flatten()
            if indices.numel() != image_embeddings.shape[0]:
                raise ValueError("image token count does not match image soft-token count")
            flat = inputs_embeds.reshape(-1, self.config.hidden_size).clone()
            flat[indices] = image_embeddings
            inputs_embeds = flat.view_as(inputs_embeds)
        if audio_features is not None:
            if self.audio_tower is None or audio_token_mask is None:
                raise ValueError("audio tower and audio token mask are required")
            audio_embeddings = self.audio_tower(audio_features)
            audio_embeddings = self.embed_audio(audio_embeddings).reshape(-1, self.config.hidden_size)
            indices = audio_token_mask.reshape(-1).nonzero(as_tuple=False).flatten()
            if indices.numel() != audio_embeddings.shape[0]:
                raise ValueError("audio token count does not match audio soft-token count")
            flat = inputs_embeds.reshape(-1, self.config.hidden_size).clone()
            flat[indices] = audio_embeddings
            inputs_embeds = flat.view_as(inputs_embeds)
        per_layer_inputs = None
        if image_token_mask is not None or audio_token_mask is not None:
            # Gemma 4 computes PLE token identity from PAD at multimodal
            # placeholder positions, not from the special media token IDs.
            ple_ids = input_ids.clone()
            if image_token_mask is not None:
                ple_ids = torch.where(image_token_mask, self.config.pad_token_id, ple_ids)
            if audio_token_mask is not None:
                ple_ids = torch.where(audio_token_mask, self.config.pad_token_id, ple_ids)
            per_layer_inputs = self.model.embed_tokens_per_layer(ple_ids)
            per_layer_inputs = per_layer_inputs * self.config.hidden_size_per_layer_input ** 0.5
            per_layer_inputs = per_layer_inputs.view(
                *ple_ids.shape, self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            )
            hidden_states = self.model(
                inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
            )
        else:
            hidden_states = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
        logits = torch.matmul(hidden_states, self.model.embed_tokens.weight.t())
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def load_weights(self, model_path: str) -> None:
        from .checkpoint import load_gemma4_multimodal_safetensors
        load_gemma4_multimodal_safetensors(self, model_path)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, **media) -> torch.Tensor:
        """Greedy generation using full-prefix recomputation."""
        generated = input_ids
        image_mask = media.pop("image_token_mask", None)
        audio_mask = media.pop("audio_token_mask", None)
        for _ in range(max_new_tokens):
            logits = self(
                generated,
                image_token_mask=image_mask,
                audio_token_mask=audio_mask,
                **media,
            )
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if image_mask is not None:
                image_mask = torch.cat(
                    (image_mask, torch.zeros_like(next_token, dtype=torch.bool)), dim=-1
                )
            if audio_mask is not None:
                audio_mask = torch.cat(
                    (audio_mask, torch.zeros_like(next_token, dtype=torch.bool)), dim=-1
                )
        return generated
