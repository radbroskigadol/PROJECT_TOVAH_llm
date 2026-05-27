# Applied update report — TOVAH v14.3.2

## Scope applied

- Added UAP/ShadowHoTT token-profile labeler and scalar profile target extraction.
- Added v14.3.2 UAP shadow-depth corpus generator with held-out paradox-family validation shards.
- Updated the existing paradox corpus generator so old commands now emit `uap_profile`, `uap_token_profiles`, `uap_profile_targets`, and `uap_schema_version` while preserving legacy `bilateral_t`, `bilateral_f`, `kind`, `domain`, and `paraconsistent_class` fields.
- Updated dataset collation to carry `uap_profile_targets` tensors into training batches.
- Added auxiliary UAP loss wiring for collapse pressure, residue preservation, glut retention, gap recognition, local/global obstruction, and support-profile consistency.
- Wired derived semantic-support auxiliary outputs into pretraining behind `--uap-aux-weight`.
- Added Shadow-depth eval and connected it to `run_full_eval`.
- Added tokenizer diagnostics for high glut/gap/residue regions.
- Added v14.3.2 docs, changelog, smoke script, and tests.

## Framing preserved

This update does not frame ShadowOptimizer as competing with AdamW. AdamW remains the classicalized floor/projection. v14.3.2 evaluates whether TOVAH preserves richer UAP token structure: truth-support, falsity-support, glut, gap, obstruction residue, collapse pressure, local-global noncollapse, backliftability, and classicalization depth.

## Validation performed in sandbox

```text
python -m compileall -q .
# passed with no compile errors

python -m pytest tests/test_v14_3_2_shadow_depth.py -q
# 4 passed

python -m pytest tests/test_v14_3_2_shadow_depth.py tests/test_high_glut_training.py tests/test_training_pipeline.py -q
# 36 passed

python tools/generate_paradox_corpus.py --out /tmp/tovah_v1432_train_smoke --n 8 --shard-size 100
python run_tovah.py --pretrain --shard-dir /tmp/tovah_v1432_train_smoke --profile debug --optimizer shadow --tokenizer byte --epochs 1 --batch-size 1 --grad-accum-steps 1 --max-steps 2 --device cpu --uap-aux-weight 0.05
# completed 2 training steps and saved checkpoint
```

## Note

The v14.3.2 auxiliary losses currently derive UAP predictions from the existing T/F semantic supports. That makes the update operational immediately while leaving a clean path for future dedicated UAP auxiliary heads.
