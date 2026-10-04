# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint loading helpers for Hugging Face model artifacts."""

import torch

from ..checkpoint_utils import (
    direct_assign,
    open_safetensors,
    resolve_safetensors_path,
)

def load_gemma3_text_safetensors(model: torch.nn.Module, model_path: str) -> None:
    """Loads a Gemma 3 text checkpoint into the local PyTorch model.

    The local implementation stores Q, K, and V in one projection while the
    Hugging Face checkpoint stores them separately. All trainable parameters
    must be mapped; silently accepting a partial checkpoint would invalidate
    numerical alignment.
    """
    model_path = resolve_safetensors_path(model_path)

    with open_safetensors(model_path) as reader:
        available = set(reader.keys())
        mapped: set[str] = set()

        def add(local_name: str, source_name: str) -> None:
            if source_name not in available:
                raise KeyError(f"Missing checkpoint key: {source_name}")
            direct_assign(model, local_name, reader.get_tensor(source_name))
            mapped.add(local_name)

        add("embedder.weight", "model.embed_tokens.weight")
        add("model.norm.weight", "model.norm.weight")

        layer_count = len(model.model.layers)
        for layer_index in range(layer_count):
            local = f"model.layers.{layer_index}"
            source = f"model.layers.{layer_index}"
            q = reader.get_tensor(f"{source}.self_attn.q_proj.weight")
            k = reader.get_tensor(f"{source}.self_attn.k_proj.weight")
            v = reader.get_tensor(f"{source}.self_attn.v_proj.weight")
            direct_assign(
                model,
                f"{local}.self_attn.qkv_proj.weight",
                torch.cat((q, k, v), dim=0),
            )
            mapped.add(f"{local}.self_attn.qkv_proj.weight")
            for suffix in (
                "o_proj",
                "q_norm",
                "k_norm",
            ):
                target_suffix = {
                    "q_norm": "query_norm",
                    "k_norm": "key_norm",
                }.get(suffix, suffix)
                add(
                    f"{local}.self_attn.{target_suffix}.weight",
                    f"{source}.self_attn.{suffix}.weight",
                )
            for suffix in (
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                add(
                    f"{local}.mlp.{suffix}.weight",
                    f"{source}.mlp.{suffix}.weight",
                )
            for suffix in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                add(f"{local}.{suffix}.weight", f"{source}.{suffix}.weight")

        parameter_names = {name for name, _ in model.named_parameters()}
        missing = sorted(parameter_names - mapped)
        if missing:
            raise RuntimeError(
                "Incomplete Gemma 3 safetensors mapping; missing local parameters: "
                + ", ".join(missing)
            )



