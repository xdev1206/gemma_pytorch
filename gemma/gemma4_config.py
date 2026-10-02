# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for the dense Gemma 4 E2B text backbone."""

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Gemma4TextConfig:
    vocab_size: int = 262144
    pad_token_id: int = 0
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    max_position_embeddings: int = 131072
    sliding_window: int = 512
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    hidden_size_per_layer_input: int = 256
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    layer_types: tuple[str, ...] = dataclasses.field(default_factory=lambda: (
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "full_attention",
    ))

    @classmethod
    def from_json(cls, path: str | Path) -> "Gemma4TextConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))["text_config"]
        return cls(
            vocab_size=data["vocab_size"],
            pad_token_id=data.get("pad_token_id", 0),
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            global_head_dim=data["global_head_dim"],
            max_position_embeddings=data["max_position_embeddings"],
            sliding_window=data["sliding_window"],
            rms_norm_eps=data["rms_norm_eps"],
            final_logit_softcapping=data["final_logit_softcapping"],
            hidden_size_per_layer_input=data["hidden_size_per_layer_input"],
            num_kv_shared_layers=data["num_kv_shared_layers"],
            use_double_wide_mlp=data["use_double_wide_mlp"],
            layer_types=tuple(data["layer_types"]),
        )

    def layer_head_dim(self, layer_index: int) -> int:
        return (self.global_head_dim
                if self.layer_types[layer_index] == "full_attention"
                else self.head_dim)
