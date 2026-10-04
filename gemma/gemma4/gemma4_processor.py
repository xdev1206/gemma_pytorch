"""Small, dependency-light Gemma 4 image processor.

The processor returns the flattened patch contract consumed by
``Gemma4VisionModel``.  Audio feature extraction remains intentionally
separate because it requires an STFT/mel implementation.
"""

import math

import numpy as np
import torch
from PIL import Image


def _target_size(height: int, width: int, patch_size: int, max_patches: int,
                 pooling_kernel_size: int) -> tuple[int, int]:
    scale = math.sqrt(max_patches * patch_size ** 2 / (height * width))
    multiple = patch_size * pooling_kernel_size
    target_h = int(math.floor(scale * height / multiple)) * multiple
    target_w = int(math.floor(scale * width / multiple)) * multiple
    if target_h == 0 or target_w == 0:
        target_h = target_h or multiple
        target_w = target_w or multiple
    return target_h, target_w


def _patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    channels, height, width = image.shape
    patches = image.reshape(channels, height // patch_size, patch_size,
                            width // patch_size, patch_size)
    return patches.permute(1, 3, 2, 4, 0).reshape(-1, patch_size * patch_size * channels)


class Gemma4ImageProcessor:
    def __init__(self, patch_size: int = 16, max_soft_tokens: int = 280,
                 pooling_kernel_size: int = 3):
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    def __call__(self, image: Image.Image | torch.Tensor) -> dict[str, torch.Tensor | int]:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = torch.from_numpy(__import__("numpy").array(image, copy=True)).permute(2, 0, 1)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Gemma 4 images must have shape [3, height, width]")
        image = image.float() / (255.0 if image.max() > 1.0 else 1.0)
        max_patches = self.max_soft_tokens * self.pooling_kernel_size ** 2
        height, width = _target_size(image.shape[-2], image.shape[-1],
                                     self.patch_size, max_patches,
                                     self.pooling_kernel_size)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False,
            antialias=True,
        ).squeeze(0)
        patches = _patchify(image, self.patch_size)
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        grid_x, grid_y = torch.meshgrid(torch.arange(patch_w), torch.arange(patch_h), indexing="xy")
        positions = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2).long()
        num_soft_tokens = patches.shape[0] // self.pooling_kernel_size ** 2
        if patches.shape[0] < max_patches:
            patches = torch.nn.functional.pad(patches, (0, 0, 0, max_patches - patches.shape[0]))
            positions = torch.nn.functional.pad(positions, (0, 0, 0, max_patches - positions.shape[0]), value=-1)
        return {
            "pixel_values": patches.unsqueeze(0),
            "pixel_position_ids": positions.unsqueeze(0),
            "num_soft_tokens": num_soft_tokens,
        }


class Gemma4AudioProcessor:
    """Minimal 16 kHz, 128-bin HTK log-mel extractor for Gemma 4 audio."""

    def __init__(self, sampling_rate: int = 16000, feature_size: int = 128,
                 frame_length: int = 320, hop_length: int = 160,
                 fft_length: int = 512, mel_floor: float = 1e-3):
        self.sampling_rate = sampling_rate
        self.feature_size = feature_size
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fft_length = fft_length
        self.mel_floor = mel_floor

    def __call__(self, waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        pad_left = self.frame_length // 2
        waveform = np.pad(waveform, (pad_left, 0))
        frame_size = self.frame_length + 1
        count = max(0, (len(waveform) - frame_size) // self.hop_length + 1)
        if count == 0:
            return torch.empty((1, 0, self.feature_size), dtype=torch.float32)
        frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_size)[::self.hop_length][:count, :-1]
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.frame_length) / self.frame_length)
        spectrum = np.abs(np.fft.rfft(frames * window, n=self.fft_length, axis=-1))
        mel = self._mel_filter_bank()
        features = np.log(spectrum @ mel + self.mel_floor)
        return torch.from_numpy(features.astype(np.float32, copy=False)).unsqueeze(0)

    def _mel_filter_bank(self) -> np.ndarray:
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        points = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(8000), self.feature_size + 2))
        bins = np.floor((self.fft_length + 1) * points / self.sampling_rate).astype(int)
        bank = np.zeros((self.fft_length // 2 + 1, self.feature_size), dtype=np.float32)
        for i in range(self.feature_size):
            left, center, right = bins[i:i + 3]
            if center > left:
                bank[left:center, i] = np.arange(left, center) / (center - left)
            if right > center:
                bank[center:right, i] = (right - np.arange(center, right)) / (right - center)
        return bank


__all__ = ["Gemma4ImageProcessor", "Gemma4AudioProcessor"]
