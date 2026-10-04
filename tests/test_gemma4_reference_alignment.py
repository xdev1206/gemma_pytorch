"""Gemma 4 numerical and behavioral comparisons with Transformers reference."""

import unittest

import torch

try:
    from .alignment_support import model_spec, require_image, require_model
    from .test_runtime_alignment import _runtime_enabled
except ImportError:
    from alignment_support import model_spec, require_image, require_model
    from test_runtime_alignment import _runtime_enabled


try:
    from transformers.models.gemma4 import Gemma4ForConditionalGeneration
except ImportError:
    Gemma4ForConditionalGeneration = None


@unittest.skipUnless(
    _runtime_enabled(),
    "set GEMMA_RUN_RUNTIME_ALIGNMENT=1 to run checkpoint-backed alignment tests",
)
class Gemma4ReferenceAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Gemma4ForConditionalGeneration is None:
            raise AssertionError(
                "install transformers>=5.18.0 for the Gemma 4 reference model"
            )
        spec = model_spec("gemma-4-E2B-it")
        cls.model_dir = require_model(spec)
        baseline = spec["runtime_baseline"]
        cls.reference_config = baseline["reference"]
        cls.max_error = cls.reference_config["max_abs_error"]
        cls.mean_error = cls.reference_config["mean_abs_error"]
        cls.device = torch.device(baseline["device"])
        cls.dtype = getattr(torch, baseline["dtype"])
        cls.reference = Gemma4ForConditionalGeneration.from_pretrained(
            str(cls.model_dir),
            torch_dtype=cls.dtype,
            device_map=str(cls.device),
            attn_implementation=cls.reference_config["attention_backend"],
            local_files_only=True,
        ).eval()
        backend = cls.reference_config["attention_backend"]
        if cls.reference.config._attn_implementation != backend:
            raise AssertionError(
                "Gemma 4 reference backend does not match manifest"
            )
        if cls.reference.model.language_model.config._attn_implementation != backend:
            raise AssertionError(
                "Gemma 4 language model reference backend does not match manifest"
            )
        from gemma.gemma4.gemma4_model import Gemma4ForConditionalGeneration as LocalModel
        from gemma.gemma4.gemma4_tokenizer import Gemma4Tokenizer

        cls.local = LocalModel.from_pretrained(
            str(cls.model_dir), dtype=cls.dtype, device=cls.device
        ).eval()
        cls.tokenizer = Gemma4Tokenizer(cls.model_dir)

    def _text_ids(self, prompt):
        return torch.tensor(
            [self.tokenizer.encode(prompt, bos=True, eos=False)],
            device=self.device,
        )

    def _assert_close(self, actual, expected, label):
        difference = (actual.float() - expected.float()).abs()
        self._alignment_metrics = getattr(self, "_alignment_metrics", {})
        self._alignment_metrics[label] = {
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "max_abs_threshold": self.max_error,
            "mean_abs_threshold": self.mean_error,
        }
        self.assertLessEqual(
            difference.max().item(), self.max_error,
            f"{label} max absolute error: {difference.max().item()}",
        )
        self.assertLessEqual(
            difference.mean().item(), self.mean_error,
            f"{label} mean absolute error: {difference.mean().item()}",
        )

    def test_text_logits_match_reference(self):
        spec = model_spec("gemma-4-E2B-it")
        for prompt in spec["text_prompts"]:
            with self.subTest(prompt=prompt):
                input_ids = self._text_ids(prompt)
                with torch.no_grad():
                    reference = self.reference(input_ids=input_ids).logits
                    actual = self.local(input_ids)
                self._assert_close(actual, reference, f"text logits: {prompt}")

    def test_text_greedy_generation_matches_reference(self):
        spec = model_spec("gemma-4-E2B-it")
        input_ids = self._text_ids(spec["text_prompts"][1])
        with torch.no_grad():
            reference = self.reference.generate(
                input_ids, max_new_tokens=4, do_sample=False
            )
            actual = self.local.generate(input_ids, max_new_tokens=4)
        self.assertTrue(torch.equal(actual, reference))

    def test_vision_features_match_reference(self):
        from PIL import Image
        from gemma.gemma4.gemma4_processor import Gemma4ImageProcessor

        image_path = require_image("golden_test_image")
        with Image.open(image_path) as image:
            processed = Gemma4ImageProcessor()(image)
        with torch.no_grad():
            actual = self.local.vision_tower(
                processed["pixel_values"].to(self.device),
                processed["pixel_position_ids"].to(self.device),
            )
            actual = self.local.embed_vision(actual)
            reference = self.reference.model.get_image_features(
                pixel_values=processed["pixel_values"].to(self.device),
                image_position_ids=processed["pixel_position_ids"].to(self.device),
            ).pooler_output[0]
        self._assert_close(actual, reference, "Gemma 4 vision features")

    def test_audio_position_embeddings_match_reference(self):
        features = torch.zeros(1, 99, 128, device=self.device)
        with torch.no_grad():
            local_input = self.local.audio_tower.subsample_conv_projection(features)
            reference_input = self.reference.model.audio_tower.subsample_conv_projection(
                features
            )[0]
            actual = self.local.audio_tower.rel_pos_enc(local_input)
            reference = self.reference.model.audio_tower.rel_pos_enc(reference_input)
        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, reference.dtype)
        self.assertEqual(local_input.dtype, reference_input.dtype)
        self.assertTrue(torch.equal(local_input, reference_input))
        self.assertEqual(
            self.local.audio_tower.rel_pos_enc.context_size,
            self.reference.model.audio_tower.rel_pos_enc.context_size,
        )
        difference = (actual.float() - reference.float()).abs()
        self.assertLessEqual(
            difference.max().item(), 0.0,
            f"position max absolute error: {difference.max().item()}, "
            f"local buffer dtype={self.local.audio_tower.rel_pos_enc.inv_timescales.dtype}, "
            f"reference buffer dtype={self.reference.model.audio_tower.rel_pos_enc.inv_timescales.dtype}, "
            f"buffer max absolute error={(self.local.audio_tower.rel_pos_enc.inv_timescales.float() - self.reference.model.audio_tower.rel_pos_enc.inv_timescales.float()).abs().max().item()}",
        )

    def test_image_text_logits_match_reference(self):
        from PIL import Image
        from gemma.gemma4.gemma4_processor import Gemma4ImageProcessor

        spec = model_spec("gemma-4-E2B-it")["runtime_baseline"]
        image_path = require_image(spec["multimodal_image"])
        with Image.open(image_path) as image:
            processed = Gemma4ImageProcessor()(image)
        soft_tokens = int(processed["num_soft_tokens"])
        text_ids = self.tokenizer.encode(spec["multimodal_prompt"], bos=False, eos=True)
        local_ids = torch.tensor(
            [[
                self.tokenizer.bos_id,
                self.tokenizer.boi_id,
                *([self.tokenizer.pad_id] * soft_tokens),
                self.tokenizer.eoi_id,
                *text_ids,
            ]],
            device=self.device,
        )
        local_mask = torch.tensor(
            [[False, False, *([True] * soft_tokens), False, *([False] * len(text_ids))]],
            device=self.device,
        )
        reference_ids = local_ids.masked_fill(
            local_mask, self.reference.config.image_token_id
        )
        with torch.no_grad():
            actual = self.local(
                local_ids,
                image_patches=processed["pixel_values"].to(self.device),
                pixel_position_ids=processed["pixel_position_ids"].to(self.device),
                image_token_mask=local_mask,
            )
            reference = self.reference(
                input_ids=reference_ids,
                pixel_values=processed["pixel_values"].to(self.device),
                image_position_ids=processed["pixel_position_ids"].to(self.device),
            ).logits
        self._assert_close(actual, reference, "Gemma 4 image-text logits")

    def test_audio_features_match_reference(self):
        from gemma.gemma4.gemma4_processor import Gemma4AudioProcessor

        spec = model_spec("gemma-4-E2B-it")
        samples = spec["runtime_baseline"]["audio_samples"]
        features = Gemma4AudioProcessor()(torch.zeros(samples)).to(self.device)
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            actual = self.local.audio_tower(features)
            reference = self.reference.model.audio_tower(
                features, attention_mask=mask
            ).last_hidden_state
        self._assert_close(actual, reference, "Gemma 4 audio features")

    def test_audio_text_logits_match_reference(self):
        spec = model_spec("gemma-4-E2B-it")
        baseline = spec["runtime_baseline"]
        from gemma.gemma4.gemma4_processor import Gemma4AudioProcessor

        features = Gemma4AudioProcessor()(torch.zeros(baseline["audio_samples"])).to(
            self.device
        )
        with torch.no_grad():
            local_audio = self.local.audio_tower(features)
        audio_tokens = local_audio.shape[1]
        text_ids = self.tokenizer.encode(
            baseline["audio_prompt"], bos=False, eos=True
        )
        local_ids = torch.tensor(
            [[
                self.tokenizer.bos_id,
                self.tokenizer.boa_id,
                *([self.tokenizer.pad_id] * audio_tokens),
                self.tokenizer.eoa_id,
                *text_ids,
            ]],
            device=self.device,
        )
        local_mask = torch.tensor(
            [[
                False,
                False,
                *([True] * audio_tokens),
                False,
                *([False] * len(text_ids)),
            ]],
            device=self.device,
        )
        reference_audio_id = getattr(self.reference.config, "audio_token_id", None)
        if reference_audio_id is None:
            reference_audio_id = self.reference.config.get_text_config().audio_token_id
        reference_ids = local_ids.masked_fill(local_mask, reference_audio_id)
        feature_mask = torch.ones(
            features.shape[:2], dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            actual = self.local(
                local_ids,
                audio_features=features,
                audio_token_mask=local_mask,
            )
            reference = self.reference(
                input_ids=reference_ids,
                input_features=features,
                input_features_mask=feature_mask,
            ).logits
        self._assert_close(actual, reference, "Gemma 4 audio-text logits")


if __name__ == "__main__":
    unittest.main()
