"""Compatibility exports for version-specific checkpoint loaders."""

from .gemma3.checkpoint import load_gemma3_text_safetensors
from .gemma4.checkpoint import (
    load_gemma4_audio_safetensors,
    load_gemma4_multimodal_safetensors,
    load_gemma4_text_safetensors,
    load_gemma4_vision_safetensors,
)

__all__ = [
    "load_gemma3_text_safetensors",
    "load_gemma4_audio_safetensors",
    "load_gemma4_multimodal_safetensors",
    "load_gemma4_text_safetensors",
    "load_gemma4_vision_safetensors",
]
