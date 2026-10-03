import unittest
from pathlib import Path

try:
    from .alignment_support import ROOT, load_manifest
except ImportError:
    from alignment_support import ROOT, load_manifest


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
            self.assertTrue(model_dir.is_dir(), f"download model into {model_dir}")
            self.assertTrue(
                (model_dir / model["checkpoint"]).is_file(),
                f"missing checkpoint for {model['id']}; download the model first",
            )
            self.assertTrue(
                (model_dir / model["tokenizer"]).is_file(),
                f"missing tokenizer for {model['id']}; download the model first",
            )
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
            "reference_backend",
            "reference_max_abs_logit_error",
            "reference_mean_abs_logit_error",
        ):
            self.assertIn(field, baseline, field)


if __name__ == "__main__":
    unittest.main()
