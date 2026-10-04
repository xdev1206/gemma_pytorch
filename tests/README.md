# Alignment tests

## Test architecture

`data/alignment_cases.json` defines model resources, prompts, audio inputs,
device/dtype, reference backend, and numeric thresholds. `alignment_support.py`
validates and resolves those resources. Tests are separated into contract,
local runtime smoke, and Gemma 4 reference comparison layers; `run_alignment.py`
selects them with explicit profiles and writes the final report.

| Profile | Scope | Strict skip handling |
| --- | --- | --- |
| `contract` | Manifest, component, and checkpoint-loader contracts | Skip allowed only for optional dependencies |
| `runtime-smoke` | Real local checkpoint behavior | Any skip is failure |
| `gemma4-reference` | Transformers numerical and behavioral comparison | Any skip is failure |
| `all` | Contract plus runtime and reference layers | Use `--strict` for acceptance |

The reference layer compares text logits and generation, vision features,
audio features, audio-text logits, and image-text logits. Report records contain
the case status, duration, thresholds, and measured max/mean absolute errors.

Run the manifest/resource checks with:

```bash
python -m unittest tests.test_manifest -v
```

Run component contract and checkpoint-loader checks after installing
`requirements.txt`:

```bash
python -m unittest tests.test_alignment tests.test_checkpoint_loading -v
```

Run the real checkpoint-backed baselines explicitly on the device and dtype
declared in `data/alignment_cases.json`:

```bash
GEMMA_RUN_RUNTIME_ALIGNMENT=1 \
  python -m unittest tests.test_runtime_alignment -v
```

Run the contract profile and write `tests/reports/alignment_report.md` plus the
machine-readable `alignment_report.json`:

```bash
python -m tests.run_alignment --profile contract
```

Run strict checkpoint-backed profiles. A skipped test, unavailable device, or
missing model is a failure in these profiles:

```bash
GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m tests.run_alignment --profile runtime-smoke
GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m tests.run_alignment --profile gemma4-reference

# Complete strict acceptance and one combined report
GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m tests.run_alignment --profile all --strict
```

Complete Gemma 4 acceptance also requires the local Transformers reference
implementation used by `test_gemma4_reference_alignment.py`:

```bash
python -m pip install -r requirements-alignment.txt
GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m unittest \
  tests.test_gemma4_reference_alignment -v
```

The runner uses explicit profiles rather than mixing contract and acceptance
tests. It records status, duration, skip reason, environment, thresholds, and
measured max/mean absolute errors, then returns non-zero when the profile is
not successful. Reports are ignored by Git.

The manifest in `data/alignment_cases.json` is the source of truth for Gemma 4
model, prompt, image, device, dtype, and tolerance inputs. Runtime acceptance
must include text logits and greedy generation, vision features, audio features,
audio-text logits, image next-token behavior, and the reference comparison. The
reference tests use the same local checkpoint through Transformers, read prompts
and thresholds from the manifest, and enforce max and mean absolute error
thresholds; a next-token match alone is insufficient.

If a required model is missing, the test run fails (it is not silently skipped)
and prints its Hugging Face download URL and expected
`models/<model-id>/` contents. Gemma 3 requires `model.safetensors` and
`tokenizer.model`; Gemma 4 requires `config.json`, `model.safetensors`, and
`tokenizer.json`.

音频 tower、音频文本 logits 和对应阈值现在都纳入正式 reference case；如果
后续 attention 层再次出现数值分叉，严格 profile 会直接失败并在报告中保留误差指标。

Gemma 3 remains a separate pending item: its text smoke baseline exists, but
multimodal behavior and numerical tolerance comparisons are not yet covered.

Checkpoint comparison tests should write temporary outputs outside the
repository and record model ID, device, dtype, generation length, and numeric
tolerance in the test report.
