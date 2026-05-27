# Patch notes — TOVAH v14.3.2 Shadow-depth update

## Added

1. UAP token-profile labels for synthetic paradox training records.
2. Held-out validation shards for contradiction transfer, collapse preservation, obstruction retention, gap recognition, adversarial paraphrase, and tokenizer diagnostics.
3. Shadow-depth evaluation metrics:
   - contradiction retention
   - noncollapse under gluts
   - gap recognition
   - truth/falsity calibration
   - local-global obstruction preservation
   - unseen paradox-family transfer
   - stabilization depth
   - loop/drift behavior
   - backliftability diagnostics
   - collapse pressure
   - support profile consistency
   - residue preservation
4. Auxiliary UAP loss scaffold for model-head integration.
5. Tokenizer diagnostics for high glut/gap/residue regions.
6. Tests covering profile labeling, corpus generation, and eval smoke behavior.

## Framing correction

This patch does not frame ShadowOptimizer as a competitor to AdamW. AdamW remains the classicalized projection/floor. v14.3.2 measures whether the Shadow/UAP layer preserves structure that ordinary next-token evaluation cannot see.

## Known limitation

This overlay is conservative. It adds the v14.3.2 modules without destructively rewriting your working `pretrain.py` or `run_tovah.py`. Wire `training.uap_aux_losses` into the active training loop after confirming the current model exposes the auxiliary heads/probes.
