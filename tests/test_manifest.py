import unittest
from pathlib import Path

try:
    from .alignment_support import (
        ROOT,
        load_manifest,
        model_download_message,
        required_model_files,
    )
except ImportError:
    from alignment_support import (
        ROOT,
        load_manifest,
        model_download_message,
        required_model_files,
    )


class AlignmentManifestTest(unittest.TestCase):
    def test_manifest_schema_and_resources(self):
        manifest = load_manifest()
        self.assertEqual(
            {item["id"] for item in manifest["models"]},
            {"gemma-3-1b-it", "gemma-4-E2B-it"},
        )
        for model in manifest["models"]:
            for field in ("id", "path", "variant", "checkpoint", "tokenizer", "status"):
                self.assertIn(field, model, model.get("id"))
            model_dir = ROOT / model["path"]
            self.assertFalse(Path(model["path"]).is_absolute())
            missing = []
            if not model_dir.is_dir():
                missing.append("model directory")
            else:
                missing.extend(
                    name for name in required_model_files(model)
                    if not (model_dir / name).is_file()
                )
            if missing:
                self.fail(model_download_message(model, model_dir, missing))
            baseline = model["runtime_baseline"]
            self.assertIn("device", baseline)
            self.assertIn("dtype", baseline)
            self.assertIn("seed", baseline)
            self.assertIn("prompt", baseline)
        for image in manifest["images"]:
            self.assertTrue((ROOT / image["path"]).is_file(), image["path"])

    def test_runtime_baselines_have_comparable_fields(self):
        manifest = load_manifest()
        gemma4 = next(item for item in manifest["models"] if item["id"] == "gemma-4-E2B-it")
        baseline = gemma4["runtime_baseline"]
        for field in (
            "expected_next_token_id",
            "expected_next_token",
            "multimodal_expected_next_token_id",
            "multimodal_expected_next_token",
            "multimodal_image",
        ):
            self.assertIn(field, baseline, field)
        self.assertEqual(
            baseline["reference"]["attention_backend"], "eager"
        )
        for field in ("package", "attention_backend", "max_abs_error", "mean_abs_error"):
            self.assertIn(field, baseline["reference"], field)
        self.assertIn("audio_prompt", baseline)
        self.assertGreater(baseline["audio_samples"], 0)


if __name__ == "__main__":
    unittest.main()
