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

def load_gemma4_text_safetensors(
        model: torch.nn.Module, model_path: str, target_prefix: str = "model."
) -> None:
    """Loads the Gemma 4 E2B dense language weights into the text backbone."""
    model_path = resolve_safetensors_path(model_path)
    with open_safetensors(model_path) as reader:
        mapped: set[str] = set()
        local_targets = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))

        def add(local_name: str, source_name: str) -> None:
            # Gemma 4 checkpoints retain duplicated K/V tensors for shared
            # layers, while the local architecture intentionally omits them.
            if local_name not in local_targets:
                return
            direct_assign(model, local_name, reader.get_tensor(source_name))
            mapped.add(local_name)

        prefix = "model.language_model."
        add(target_prefix + "embed_tokens.weight", prefix + "embed_tokens.weight")
        add(target_prefix + "embed_tokens_per_layer.weight", prefix + "embed_tokens_per_layer.weight")
        add(target_prefix + "per_layer_model_projection.weight", prefix + "per_layer_model_projection.weight")
        add(target_prefix + "per_layer_projection_norm.weight", prefix + "per_layer_projection_norm.weight")
        add(target_prefix + "norm.weight", prefix + "norm.weight")
        for i in range(model.config.num_hidden_layers):
            local = target_prefix + f"layers.{i}"
            source = f"{prefix}layers.{i}"
            for suffix in (
                "input_layernorm", "post_attention_layernorm",
                "pre_feedforward_layernorm", "post_feedforward_layernorm",
            ):
                add(f"{local}.{suffix}.weight", f"{source}.{suffix}.weight")
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                add(f"{local}.mlp.{suffix}.weight", f"{source}.mlp.{suffix}.weight")
            for suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add(f"{local}.self_attn.{suffix}.weight", f"{source}.self_attn.{suffix}.weight")
            for suffix in ("q_norm", "k_norm"):
                add(f"{local}.self_attn.{suffix}.weight", f"{source}.self_attn.{suffix}.weight")
            add(f"{local}.per_layer_input_gate.weight", f"{source}.per_layer_input_gate.weight")
            add(f"{local}.per_layer_projection.weight", f"{source}.per_layer_projection.weight")
            add(f"{local}.post_per_layer_input_norm.weight", f"{source}.post_per_layer_input_norm.weight")
            add(f"{local}.layer_scalar", f"{source}.layer_scalar")
        expected = {
            name for name, _ in model.named_parameters()
            if name.startswith(target_prefix)
        }
        missing = sorted(expected - mapped)
        if missing:
            raise RuntimeError("Incomplete Gemma 4 mapping: " + ", ".join(missing))


def load_gemma4_vision_safetensors(
        model: torch.nn.Module, model_path: str, target_prefix: str = ""
) -> None:
    """Loads the Gemma 4 E2B vision tower weights."""
    model_path = resolve_safetensors_path(model_path)
    with open_safetensors(model_path) as reader:
        mapped: set[str] = set()

        def add(local_name: str, source_name: str) -> None:
            direct_assign(model, local_name, reader.get_tensor(source_name))
            mapped.add(local_name)

        add(target_prefix + "patch_embedder.input_proj.weight", "model.vision_tower.patch_embedder.input_proj.weight")
        add(target_prefix + "patch_embedder.position_embedding_table", "model.vision_tower.patch_embedder.position_embedding_table")
        vision_config = (
            model.vision_tower.config
            if target_prefix
            else model.config
        )
        for i in range(vision_config.num_hidden_layers):
            local = target_prefix + f"layers.{i}"
            source = f"model.vision_tower.encoder.layers.{i}"
            for suffix in (
                "input_layernorm", "post_attention_layernorm",
                "pre_feedforward_layernorm", "post_feedforward_layernorm",
            ):
                add(f"{local}.{suffix}.weight", f"{source}.{suffix}.weight")
            for suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add(f"{local}.self_attn.{suffix}.linear.weight",
                    f"{source}.self_attn.{suffix}.linear.weight")
                for limit in ("input_min", "input_max", "output_min", "output_max"):
                    add(f"{local}.self_attn.{suffix}.{limit}",
                        f"{source}.self_attn.{suffix}.{limit}")
            for suffix in ("q_norm", "k_norm"):
                add(f"{local}.self_attn.{suffix}.weight",
                    f"{source}.self_attn.{suffix}.weight")
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                add(f"{local}.mlp.{suffix}.linear.weight",
                    f"{source}.mlp.{suffix}.linear.weight")
                for limit in ("input_min", "input_max", "output_min", "output_max"):
                    add(f"{local}.mlp.{suffix}.{limit}",
                        f"{source}.mlp.{suffix}.{limit}")
        expected = {
            name for name, _ in model.named_parameters()
            if name.startswith(target_prefix)
        }
        missing = sorted(expected - mapped)
        if missing:
            raise RuntimeError("Incomplete Gemma 4 vision mapping: " + ", ".join(missing))


def load_gemma4_multimodal_safetensors(
        model: torch.nn.Module, model_path: str
) -> None:
    """Loads Gemma 4 language, vision, and image projection weights."""
    load_gemma4_text_safetensors(model, model_path, target_prefix="model.")
    if model.vision_tower is not None:
        load_gemma4_vision_safetensors(model, model_path, target_prefix="vision_tower.")
    if model.audio_tower is not None:
        load_gemma4_audio_safetensors(model.audio_tower, model_path)
    model_path = resolve_safetensors_path(model_path)
    with open_safetensors(model_path) as reader:
        projection = reader.get_tensor("model.embed_vision.embedding_projection.weight")
    if model.embed_vision is not None:
        model.load_state_dict(
            {"embed_vision.embedding_projection.weight": projection}, strict=False
        )
    if model.embed_audio is not None:
        with open_safetensors(model_path) as reader:
            audio_projection = reader.get_tensor(
                "model.embed_audio.embedding_projection.weight"
            )
        model.load_state_dict(
            {"embed_audio.embedding_projection.weight": audio_projection},
            strict=False,
        )


def load_gemma4_audio_safetensors(
        model: torch.nn.Module, model_path: str
) -> None:
    """Loads the Gemma 4 E2B audio tower weights."""
    model_path = resolve_safetensors_path(model_path)
    with open_safetensors(model_path) as reader:
        mapped: set[str] = set()

        def add(local_name: str, source_name: str) -> None:
            direct_assign(model, local_name, reader.get_tensor(source_name))
            mapped.add(local_name)

        root = "model.audio_tower."
        for name in (
            "subsample_conv_projection.layer0.conv.weight",
            "subsample_conv_projection.layer0.norm.weight",
            "subsample_conv_projection.layer1.conv.weight",
            "subsample_conv_projection.layer1.norm.weight",
            "subsample_conv_projection.input_proj_linear.weight",
            "output_proj.weight", "output_proj.bias",
        ):
            add(name, root + name)
        for i in range(model.config.num_hidden_layers):
            local = f"layers.{i}"
            source = f"{root}layers.{i}"
            for branch in ("feed_forward1", "feed_forward2"):
                for linear in ("ffw_layer_1", "ffw_layer_2"):
                    for limit in ("input_min", "input_max", "output_min", "output_max"):
                        add(f"{local}.{branch}.{linear}.{limit}",
                            f"{source}.{branch}.{linear}.{limit}")
                    add(f"{local}.{branch}.{linear}.linear.weight",
                        f"{source}.{branch}.{linear}.linear.weight")
                for norm in ("pre_layer_norm", "post_layer_norm"):
                    add(f"{local}.{branch}.{norm}.weight", f"{source}.{branch}.{norm}.weight")
            for linear in ("linear_start", "linear_end"):
                for limit in ("input_min", "input_max", "output_min", "output_max"):
                    add(f"{local}.lconv1d.{linear}.{limit}",
                        f"{source}.lconv1d.{linear}.{limit}")
                add(f"{local}.lconv1d.{linear}.linear.weight",
                    f"{source}.lconv1d.{linear}.linear.weight")
            add(f"{local}.lconv1d.depthwise_conv1d.weight",
                f"{source}.lconv1d.depthwise_conv1d.weight")
            for norm in ("pre_layer_norm", "conv_norm"):
                add(f"{local}.lconv1d.{norm}.weight",
                    f"{source}.lconv1d.{norm}.weight")
            for norm in ("norm_pre_attn", "norm_post_attn", "norm_out"):
                add(f"{local}.{norm}.weight", f"{source}.{norm}.weight")
            for linear in ("q_proj", "k_proj", "v_proj", "post"):
                for limit in ("input_min", "input_max", "output_min", "output_max"):
                    add(f"{local}.self_attn.{linear}.{limit}",
                        f"{source}.self_attn.{linear}.{limit}")
                add(f"{local}.self_attn.{linear}.linear.weight",
                    f"{source}.self_attn.{linear}.linear.weight")
            add(f"{local}.self_attn.relative_k_proj.weight",
                f"{source}.self_attn.relative_k_proj.weight")
            add(f"{local}.self_attn.per_dim_scale",
                f"{source}.self_attn.per_dim_scale")
        expected = {name for name, _ in model.named_parameters()}
        missing = sorted(expected - mapped)
        if missing:
            raise RuntimeError("Incomplete Gemma 4 audio mapping: " + ", ".join(missing))

