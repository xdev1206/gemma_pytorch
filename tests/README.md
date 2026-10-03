# Alignment tests

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

Run the complete suite and write `tests/reports/alignment_report.md` plus the
machine-readable `alignment_report.json`:

```bash
GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m tests.run_alignment
```

Complete Gemma 4 acceptance also requires the local Transformers reference
implementation used by `test_gemma4_reference_alignment.py`:

```bash
python -m pip install -r requirements-alignment.txt
GEMMA_RUN_RUNTIME_ALIGNMENT=1 python -m unittest \
  tests.test_gemma4_reference_alignment -v
```

The runner records each test's status, duration, skip reason, and failure or
error details, then returns a non-zero exit code when the suite is not
successful. Reports are ignored by Git.

The manifest in `data/alignment_cases.json` is the source of truth for Gemma 4
model, prompt, image, device, dtype, and tolerance inputs. Runtime acceptance
must include text logits and greedy generation, vision features, audio features,
image next-token behavior, and the reference comparison. The reference tests
use the same local checkpoint through Transformers and enforce max and mean
absolute error thresholds; a next-token match alone is insufficient.

当前音频对齐仍需继续修复：音频模型已统一生成并传递 blocked 5D attention mask，且补齐层末端 clipping；第 0 层 attention 仍存在数值分叉，未通过验收。

Gemma 3 remains a separate pending item: its text smoke baseline exists, but
multimodal behavior and numerical tolerance comparisons are not yet covered.

Checkpoint comparison tests should write temporary outputs outside the
repository and record model ID, device, dtype, generation length, and numeric
tolerance in the test report.
