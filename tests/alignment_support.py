"""Shared manifest and resource helpers for alignment tests."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "data" / "alignment_cases.json"

MODEL_URLS = {
    "gemma-3-1b-it": "https://huggingface.co/google/gemma-3-1b-it",
    "gemma-4-E2B-it": "https://huggingface.co/google/gemma-4-E2B-it",
}


class ModelUnavailableError(FileNotFoundError):
    """Raised when a required alignment model is not installed."""


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


def required_model_files(spec: dict) -> list[str]:
    required = [spec["checkpoint"], spec["tokenizer"]]
    if spec["id"] == "gemma-4-E2B-it":
        required.insert(0, "config.json")
    return required


def model_download_message(spec: dict, path: Path, missing: list[str]) -> str:
    model_id = spec["id"]
    layout = "\n".join(
        f"    ├── {name}" for name in required_model_files(spec)
    )
    return (
        f"Model {model_id} is unavailable.\n"
        f"Download: {MODEL_URLS[model_id]}\n"
        "Expected local layout:\n"
        "  models/ (symlink)\n"
        f"  └── {model_id}/\n"
        f"{layout}\n"
        f"Missing: {', '.join(missing)}\n"
        f"Resolved model directory: {path}"
    )


def require_model(spec: dict) -> Path:
    path = model_path(spec)
    missing = []
    if not path.is_dir():
        missing.append("model directory")
    else:
        missing.extend(
            name for name in required_model_files(spec)
            if not (path / name).is_file()
        )
    if missing:
        raise ModelUnavailableError(
            model_download_message(spec, path, missing)
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
