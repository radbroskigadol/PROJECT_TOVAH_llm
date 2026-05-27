# CHANGELOG v14.3.4 — Frontier Hardening

- Fixed `ScalableBilateralCore` initialization: embeddings and linear weights now use std=0.02.
- Added residual-output rescaling by `1/sqrt(2*n_blocks)`.
- Resolved frontier bilateral semantics: shared vocabulary projection, compact per-token semantic T/F supports as primary paraconsistent heads.
- Replaced hard K/G budget walls with calibration/metadata-target losses.
- Compacted `ShadowOptimizer` state to four persistent buffers per parameter: `m`, `rms2`, `K_glut_q`, `R_obs_q`.
- Added old-checkpoint compaction for legacy `T_sup/F_sup/K_glut/R_obs` optimizer state.
- Removed training-time logits concentration penalty; loop control remains decode/eval/reward-side.
- Made `lane_mixture()` exclude forced-totalization lane D unless explicitly requested.
- Stabilized `bilateral_or()` evidence accumulation.
- Added differentiable paraconsistent K/G/A/B surrogates.
- Added `FormalHoTTChecker` reward scaffold over Π/Σ/Id tasks.
- Added optional QLoRA/DoRA bilateral adapter scaffold.
- Added Muon-style optimizer wrapper and made frontier profiles default to Muon unless overridden.
- Added kernel decomposition role-module scaffolds.
