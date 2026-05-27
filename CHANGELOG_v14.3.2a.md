# TOVAH v14.3.2a — Eval CLI and Shadow-depth hardening

## Purpose

v14.3.2a hardens the v14.3.2 Shadow-depth milestone. It does not reframe TOVAH as competing against AdamW. AdamW remains the classicalized floor/projection of the richer UAP ShadowOptimizer token ontology.

## Applied changes

- Added a real `main()` CLI entrypoint to `training/eval.py`.
- Added direct-script support for `python training/eval.py` from inside the `tovah_v14` folder.
- Added package/module support for `python -m tovah_v14.training.eval` from the package parent.
- Added `tovah-eval = tovah_v14.training.eval:main` to `pyproject.toml`.
- Added combined eval JSON writing via `--out`.
- Added checkpoint/model/tokenizer loading to eval CLI.
- Added optional model-generated Shadow-depth probing with `--max-examples-shadow-model`.
- Added metric provenance warnings to `training/shadow_depth_eval.py` so label-derived source-text scoring is not mistaken for proof of learned UAP geometry.
- Added compatibility flags to `tools/generate_paradox_corpus.py`:
  - `--emit-uap-profiles`
  - `--emit-validation-shards`
- Added tests for eval CLI, provenance warnings, and disabled model-generated Shadow-depth default behavior.
- Updated package metadata to `14.3.2a0` and runtime constants to `14.3.2a`.

## Correct eval commands

From inside the `tovah_v14` folder:

```powershell
python .\training\eval.py `
  --checkpoint .\checkpoints\tovah_v14_3_2_heavy_uapshadow_bpe_0100.pt `
  --shard-dir .\tovah_corpus\paradox_v14_3_2 `
  --tokenizer .\tokenizer.json `
  --out .\runs\tovah_v14_3_2a_shadow_depth_eval.json `
  --n-gen-samples 0
```

From the parent folder:

```powershell
python -m tovah_v14.training.eval `
  --checkpoint .\tovah_v14\checkpoints\tovah_v14_3_2_heavy_uapshadow_bpe_0100.pt `
  --shard-dir .\tovah_v14\tovah_corpus\paradox_v14_3_2 `
  --tokenizer .\tovah_v14\tokenizer.json `
  --out .\tovah_v14\runs\tovah_v14_3_2a_shadow_depth_eval.json `
  --n-gen-samples 0
```

Optional harder generated-continuation probe, CPU-small:

```powershell
python .\training\eval.py `
  --checkpoint .\checkpoints\tovah_v14_3_2_heavy_uapshadow_bpe_0100.pt `
  --shard-dir .\tovah_corpus\paradox_v14_3_2 `
  --tokenizer .\tokenizer.json `
  --out .\runs\tovah_v14_3_2a_shadow_depth_eval_generated_probe.json `
  --n-gen-samples 0 `
  --max-examples-shadow-model 16 `
  --shadow-model-max-gen-tokens 80 `
  --shadow-model-temperature 0.0
```

## Interpretation hardening

v14.3.2a explicitly separates:

1. **label/provenance Shadow-depth validation** — verifies that corpus UAP labels and preservation metrics are internally coherent; and
2. **model-generated Shadow-depth probing** — asks whether the model can generate continuations that preserve UAP profiles.

This prevents perfect label-derived metrics from being oversold as evidence of learned ShadowHoTT geometry.
