# TOVAH v14.2.2 — High-Glut Gradient Flow Patch

This maintenance release applies the high-glut live-backprop patches requested after the v14.2.1 HoTT frontier fix pack.

## Changes

1. **Metadata-aware training loss** — both `train_shadow_step()` and production `pretrain()` now consume `bilateral_t`/`bilateral_f` corpus metadata when present.
2. **Mean-glut phase detection** — the old shape-dependent `sum(K) > 5 and h == 0` rule is replaced by mean predicted K/G plus corpus metadata, so genuinely high-glut batches enter `Collapse-Resistant Paradox`.
3. **Lane routing gradients** — K-heavy examples softly route toward semantic lane B; G-heavy examples softly route toward lane C; classical true-only/false-only examples route toward lane A. Lane D remains a forced-totalization/readout lane, not a default training target.
4. **Contradiction/gap-preserving metadata budgets** — known K/G examples receive relaxed contradiction/gap budgets, so the training objective preserves real gluts/gaps rather than blindly collapsing them.
5. **High-glut throttle on AdamW path** — AdamW now honors ShadowHoTT phase signals by lowering LR and tightening gradient clipping for `Collapse-Resistant Paradox` batches.
6. **Live corpus records** — `train_shadow_step()` accepts legacy strings and metadata-bearing dict records.
7. **Regression coverage** — added tests for lane routing, metadata semantic loss, mean-glut phase detection, live metadata training, pretrain batch metadata use, and AdamW collapse throttling.

## Status

This is still a prototype verifier/runtime layer, not a complete dependent type checker or a frontier-scale training proof. The patch makes high-glut gradient behavior explicit and test-covered without removing the full-vocab semantic objective.
