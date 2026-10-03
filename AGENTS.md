# Repository Guidelines

## Project Structure & Module Organization

This is a Python implementation of Gemma text and multimodal models. Core
model, configuration, tokenizer, Gemma 4 processor, preprocessing, and SigLIP
vision modules are under `gemma/`. Runnable inference examples are in `scripts/`: `run.py` for
text, `run_multimodal.py` for image-plus-text, and `run_xla.py` for
PyTorch/XLA. Tokenizer files live in `tokenizer/`, sample images in
`scripts/images/`, and container recipes in `docker/`.

## Build, Test, and Development Commands

Install the pinned Python dependencies and package locally with:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

For an end-to-end smoke test, follow `README.md` to build the appropriate
Docker image and run an inference script, supplying `--ckpt` and `--variant`.
Use `--device=cuda` for GPU inference. XLA runs require the matching Docker
recipe and `PJRT_DEVICE=CPU`, `TPU`, or `CUDA`.

Tests use Python's standard `unittest` runner. Run the fast manifest, contract,
and checkpoint-loader checks with
`python -m unittest tests.test_manifest tests.test_alignment tests.test_checkpoint_loading -v`.
Run real checkpoint baselines explicitly with
`GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m unittest tests.test_runtime_alignment -v`.
Gemma 4 reference acceptance additionally requires `requirements-alignment.txt`
and `GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m unittest tests.test_gemma4_reference_alignment -v`.
Validate changes with the relevant inference example and compile checks such as
`python -m compileall gemma scripts tests`. There is no configured formatter or
linter. The `safetensors` and `tokenizers` dependencies are required for
Hugging Face checkpoint and Gemma 4 tokenizer loading; the Gemma 4 processor
uses NumPy for audio feature extraction.

## Coding Style & Naming Conventions

Use Python 3.11+ with four-space indentation, descriptive `snake_case`
functions and variables, and `PascalCase` classes. Keep model tensor shapes,
device handling, and dtype behavior explicit. Match the surrounding style and
avoid unrelated refactors; preserve the Apache license headers in Python files.

## Testing Guidelines

When adding behavior, add focused tests if a test framework is introduced;
otherwise document the exact inference command used for validation. Include
CPU coverage where practical, and exercise CUDA/XLA-specific paths on their
corresponding hardware.

## Commit & Pull Request Guidelines

Use short imperative commit subjects consistent with existing history, such as
`Add Gemma3` or `Run gemma3 on gpus`. Pull requests should explain the change,
identify validation commands and hardware, link relevant issues, and include
sample output or screenshots when behavior is user-visible. All submissions
require review and the Google CLA; see `CONTRIBUTING.md`.
