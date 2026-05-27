# Patch Notes — v14.3.2a

## Fixed

- `training/eval.py` now has a real CLI `main()`.
- `python training/eval.py ...` works from inside the `tovah_v14` directory.
- `python -m tovah_v14.training.eval ...` works from the package parent.
- Eval writes output JSON to `--out`.
- The paradox corpus generator accepts `--emit-uap-profiles` and `--emit-validation-shards` as compatibility flags.

## Hardened

- Shadow-depth eval now reports metric provenance.
- Source-text fallback is clearly marked as label/provenance validation.
- Optional model-generated Shadow-depth probing is available via `--max-examples-shadow-model`.
- Added tests for eval CLI and provenance warnings.

## Validation run in patch environment

```text
python -m compileall -q .
# passed

python -m pytest tests/test_v14_3_2a_eval_hardening.py -q
# 3 passed

python -m pytest tests/test_v14_3_2_shadow_depth.py -q
# 4 passed

python -m pytest tests/test_high_glut_training.py -q
# 13 passed

python -m pytest tests/test_training_pipeline.py -q
# 19 passed
```

A full all-tests run was not completed in the sandbox because some historical CPU-heavy tests exceeded the interactive timeout. The targeted v14.3.2a/eval/training smoke path passed.
