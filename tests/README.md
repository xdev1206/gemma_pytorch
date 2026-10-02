# Alignment tests

Run the dependency-light manifest checks with:

```bash
python -m unittest tests.test_alignment.AlignmentDataTest -v
```

Run all deterministic preprocessing checks after installing
`requirements.txt`:

```bash
python -m unittest tests.test_alignment -v
```

The manifest in `data/alignment_cases.json` is the source of truth for model,
prompt, image, and runtime baseline inputs. Gemma 3 1B has a safetensors loader
and a recorded CUDA smoke baseline. Gemma 4 E2B has text, vision, audio, and
tokenizer and image-processor adapters, and a recorded CUDA next-token baseline;
audio processor integration and full multimodal numerical alignment are still pending.
The fixed image+text CUDA/bfloat16 case now has matching top-1/top-5 tokens
with a recorded mean absolute logits error of about `0.21`; broader tolerance
coverage is still pending.

Checkpoint comparison tests should write temporary outputs outside the
repository and record model ID, device, dtype, generation length, and numeric
tolerance in the test report.
