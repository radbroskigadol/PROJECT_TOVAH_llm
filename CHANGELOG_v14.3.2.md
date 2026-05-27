# CHANGELOG v14.3.2 — Shadow-depth / UAP token-profile update

## Framing

v14.3.2 does not treat ShadowOptimizer as an AdamW competitor. AdamW remains the classicalized projection/floor. The new work asks whether TOVAH learns and preserves richer token ontology: truth-support, falsity-support, glut, gap, obstruction residue, collapse pressure, local/global noncollapse, and classicalization depth.

## Added

- UAP token-profile labeler and profile target extraction.
- `tools/generate_uap_shadow_corpus.py` with held-out paradox-family transfer shards.
- Existing `tools/generate_paradox_corpus.py` now emits v14.3.2 `uap_profile`, `uap_token_profiles`, and scalar profile targets while preserving legacy fields.
- Dataset collation now carries `uap_profile_targets` tensors into training batches.
- `training/uap_aux_losses.py` with collapse, residue, glut, gap, obstruction, and support-consistency loss scaffold.
- Pretraining now wires derived semantic-support auxiliary losses via `--uap-aux-weight`.
- `training/shadow_depth_eval.py` and full eval integration for contradiction retention, noncollapse under gluts, gap recognition, obstruction preservation, residue preservation, transfer, stabilization, loop/drift, backliftability, collapse pressure, and support consistency.
- Tokenizer diagnostics for high glut/gap/residue regions.
- v14.3.2 smoke tests and PowerShell smoke runner.

## Important limitation

The auxiliary losses use derived outputs from the current T/F semantic supports. They are intentionally ready for replacement or augmentation by dedicated learned UAP heads later.
