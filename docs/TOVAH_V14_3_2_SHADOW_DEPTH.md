# TOVAH v14.3.2 — Shadow-depth / UAP token-profile update

v14.3.2 shifts the evaluation target. The point is not to stage another AdamW-vs-ShadowOptimizer horse race. AdamW is treated as the classicalized projection/floor of the richer UAP ShadowOptimizer. The update asks whether TOVAH preserves a richer token ontology: truth-support, falsity-support, glut, gap, obstruction residue, collapse pressure, and classicalization depth.

## What this overlay adds

- `tools/uap_shadow_profiles.py` — deterministic UAP token-profile labeler and Shadow-depth metric primitives.
- `tools/generate_uap_shadow_corpus.py` — corpus generator with UAP profile labels, held-out paradox families, and validation shards.
- `training/shadow_depth_eval.py` — eval harness for contradiction retention, noncollapse under gluts, gap recognition, obstruction preservation, residue preservation, and related metrics.
- `training/uap_aux_losses.py` — auxiliary loss scaffold to add to cross entropy once model heads/probes expose UAP profile predictions.
- `tools/tokenizer_shadow_diagnostics.py` — tokenizer fragmentation diagnostics over glut/gap/residue regions.
- `tests/test_v14_3_2_shadow_depth.py` — smoke tests for labels, corpus generation, and eval.

## Generate v14.3.2 corpus

```powershell
python .\tools\generate_uap_shadow_corpus.py `
  --out .\tovah_corpus\uap_shadow_depth_v14_3_2 `
  --n 20000 `
  --shard-size 5000
```

## Run Shadow-depth corpus/eval smoke

```powershell
python .\training\shadow_depth_eval.py `
  .\tovah_corpus\uap_shadow_depth_v14_3_2 `
  --out .\runs\tovah_v14_3_2_shadow_depth_eval.json
```

## Run tokenizer diagnostics

```powershell
python .\tools\tokenizer_shadow_diagnostics.py `
  .\tovah_corpus\uap_shadow_depth_v14_3_2 `
  --tokenizer .\tokenizer.json `
  --out .\runs\tovah_v14_3_2_tokenizer_shadow_diagnostics.json
```

## Training objective direction

The v14.3.2 objective scaffold is:

```text
cross_entropy
+ collapse_penalty
+ residue_preservation_loss
+ glut_retention_loss
+ gap_recognition_loss
+ local_global_obstruction_loss
+ support_profile_consistency_loss
```

Use `training/uap_aux_losses.py` when the model has auxiliary heads/probes for `t_support`, `f_support`, `glut_mass`, `gap_mass`, `obstruction_residue`, `collapse_pressure`, and `classicalization_depth`.

## Correct interpretation

A good result is not “ShadowOptimizer beats AdamW.” A good result is “TOVAH preserves richer token structure while retaining the classical floor.” Cross-entropy remains necessary, but it is not sufficient for this ontology.
