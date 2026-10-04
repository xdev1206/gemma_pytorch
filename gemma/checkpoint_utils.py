# Copyright 2026 Google LLC
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

"""Public helpers shared by safetensors checkpoint loaders."""

import os

import torch

__all__ = [
    "open_safetensors",
    "direct_assign",
    "resolve_safetensors_path",
]


def open_safetensors(path: str):
    """Open a safetensors file on CPU with a clear dependency error."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "Loading .safetensors checkpoints requires the safetensors package."
        ) from exc
    return safe_open(path, framework="pt", device="cpu")


def direct_assign(model: torch.nn.Module, local_name: str, value: torch.Tensor) -> None:
    """Copy a checkpoint tensor into a named parameter or buffer."""
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    if local_name not in targets:
        raise KeyError(f"Unknown local checkpoint target: {local_name}")
    with torch.no_grad():
        targets[local_name].copy_(value.to(device=targets[local_name].device,
                                           dtype=targets[local_name].dtype))


def resolve_safetensors_path(path: str) -> str:
    """Resolve a checkpoint directory to its default safetensors file."""
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path
