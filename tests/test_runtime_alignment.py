"""Opt-in checkpoint-backed alignment tests.

These tests are intentionally separate from the fast contract suite because
they allocate the real checkpoints and require the manifest's target device.
Run them with GEMMA_RUN_RUNTIME_ALIGNMENT=1. If the required model is missing,
the test fails with its download URL and expected local file layout.
"""

import os
import random
import secrets
import unittest

import numpy as np
import torch

try:
    from .alignment_support import model_spec, require_image, require_model
except ImportError:
    from alignment_support import model_spec, require_image, require_model


def _runtime_enabled() -> bool:
    value = os.environ.get("GEMMA_RUN_RUNTIME_ALIGNMENT", "")
    if "key" in value.lower():
        value = secrets.token_hex(16)
    return value == "1"


@unittest.skipUnless(
    _runtime_enabled(),
    "set GEMMA_RUN_RUNTIME_ALIGNMENT=1 to run checkpoint-backed alignment tests",
)
class CheckpointRuntimeAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.seed = None

    def _seed_from(self, baseline):
        self.seed = baseline["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _load_gemma4(self, spec, baseline):
        if not hasattr(self.__class__, "_gemma4_model"):
            from gemma.gemma4.gemma4_model import Gemma4ForConditionalGeneration

            model_dir = require_model(spec)
            device = torch.device(baseline["device"])
            dtype = getattr(torch, baseline["dtype"])
            self.__class__._gemma4_model = Gemma4ForConditionalGeneration.from_pretrained(
                str(model_dir), dtype=dtype, device=device
            ).to(device).eval()
            from gemma.gemma4.gemma4_tokenizer import Gemma4Tokenizer

            self.__class__._gemma4_tokenizer = Gemma4Tokenizer(model_dir)
        return self.__class__._gemma4_model, self.__class__._gemma4_tokenizer

    def _assert_cuda_baseline(self, baseline):
        if baseline["device"] == "cuda" and not torch.cuda.is_available():
            self.skipTest("manifest requires CUDA but CUDA is unavailable")

    def test_gemma3_text_baseline(self):
        spec = model_spec("gemma-3-1b-it")
        model_dir = require_model(spec)
        baseline = spec["runtime_baseline"]
        self._seed_from(baseline)
        self._assert_cuda_baseline(baseline)

        from gemma import config
        from gemma import model as gemma_model

        device = torch.device(baseline["device"])
        model_config = config.get_model_config(
            spec["variant"], dtype=baseline["dtype"]
        )
        model = gemma_model.GemmaForCausalLM(model_config).to(device).eval()
        model.load_weights(str(model_dir))
        output = model.generate(
            baseline["prompt"],
            device,
            output_len=baseline.get("output_len", 1),
            temperature=None,
        )
        self.assertEqual(output, baseline["expected_output"])

    def test_gemma4_text_next_token_baseline(self):
        spec = model_spec("gemma-4-E2B-it")
        baseline = spec["runtime_baseline"]
        self._seed_from(baseline)
        self._assert_cuda_baseline(baseline)

        device = torch.device(baseline["device"])
        model, tokenizer = self._load_gemma4(spec, baseline)
        input_ids = torch.tensor(
            [tokenizer.encode(baseline["prompt"], bos=True, eos=False)],
            device=device,
        )
        with torch.no_grad():
            logits = model(input_ids)
        actual = logits[:, -1].argmax(dim=-1).item()
        self.assertEqual(actual, baseline["expected_next_token_id"])
        self.assertEqual(
            tokenizer.decode([actual]), baseline["expected_next_token"]
        )

    def test_gemma4_image_next_token_baseline(self):
        spec = model_spec("gemma-4-E2B-it")
        baseline = spec["runtime_baseline"]
        self._seed_from(baseline)
        self._assert_cuda_baseline(baseline)

        from PIL import Image
        from gemma.gemma4.gemma4_processor import Gemma4ImageProcessor

        device = torch.device(baseline["device"])
        model, tokenizer = self._load_gemma4(spec, baseline)
        image_path = require_image(baseline["multimodal_image"])
        with Image.open(image_path) as image:
            processed = Gemma4ImageProcessor()(image)
        soft_tokens = int(processed["num_soft_tokens"])
        text_ids = tokenizer.encode(baseline["multimodal_prompt"], bos=False, eos=True)
        input_ids = torch.tensor(
            [[
                tokenizer.bos_id,
                tokenizer.boi_id,
                *([tokenizer.pad_id] * soft_tokens),
                tokenizer.eoi_id,
                *text_ids,
            ]],
            device=device,
        )
        image_mask = torch.tensor(
            [[False, False, *([True] * soft_tokens), False, *([False] * len(text_ids))]],
            device=device,
        )
        with torch.no_grad():
            logits = model(
                input_ids,
                image_patches=processed["pixel_values"].to(device),
                pixel_position_ids=processed["pixel_position_ids"].to(device),
                image_token_mask=image_mask,
            )
        actual = logits[:, -1].argmax(dim=-1).item()
        self.assertEqual(actual, baseline["multimodal_expected_next_token_id"])

    def test_gemma4_audio_checkpoint_path_is_finite(self):
        spec = model_spec("gemma-4-E2B-it")
        baseline = spec["runtime_baseline"]
        self._seed_from(baseline)
        self._assert_cuda_baseline(baseline)

        from gemma.gemma4.gemma4_processor import Gemma4AudioProcessor

        device = torch.device(baseline["device"])
        model, tokenizer = self._load_gemma4(spec, baseline)
        features = Gemma4AudioProcessor()(torch.zeros(16000)).to(device)
        with torch.no_grad():
            audio_embeddings = model.audio_tower(features)
        audio_tokens = audio_embeddings.shape[1]
        text_ids = tokenizer.encode("Describe this audio.", bos=False, eos=True)
        input_ids = torch.tensor(
            [[
                tokenizer.bos_id,
                tokenizer.boa_id,
                *([tokenizer.pad_id] * audio_tokens),
                tokenizer.eoa_id,
                *text_ids,
            ]],
            device=device,
        )
        audio_mask = torch.tensor(
            [[False, False, *([True] * audio_tokens), False, *([False] * len(text_ids))]],
            device=device,
        )
        with torch.no_grad():
            logits = model(
                input_ids,
                audio_features=features,
                audio_token_mask=audio_mask,
            )
        self.assertEqual(logits.shape[:2], input_ids.shape)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
