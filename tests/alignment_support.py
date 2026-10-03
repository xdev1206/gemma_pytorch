"""Shared manifest and resource helpers for alignment tests."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "data" / "alignment_cases.json"


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required = {"models", "images", "multimodal_prompts"}
    missing = required - manifest.keys()
    if missing:
        raise AssertionError(f"alignment manifest is missing sections: {sorted(missing)}")
    return manifest


def model_spec(model_id: str) -> dict:
    matches = [item for item in load_manifest()["models"] if item["id"] == model_id]
    if len(matches) != 1:
        raise AssertionError(f"manifest must contain one model entry for {model_id}")
    return matches[0]


def model_path(spec: dict) -> Path:
    return ROOT / spec["path"]


def require_model(spec: dict) -> Path:
    path = model_path(spec)
    checkpoint = path / spec["checkpoint"]
    if not path.is_dir() or not checkpoint.is_file():
        raise unittest.SkipTest(
            f"{path} or {checkpoint.name} is unavailable; download the alignment model first"
        )
    return path


def require_image(image_id: str) -> Path:
    matches = [item for item in load_manifest()["images"] if item["id"] == image_id]
    if len(matches) != 1:
        raise AssertionError(f"manifest must contain one image entry for {image_id}")
    path = ROOT / matches[0]["path"]
    if not path.is_file():
        raise unittest.SkipTest(f"{path} is unavailable; restore the test image first")
    return path
