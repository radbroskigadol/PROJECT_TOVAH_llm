# TOVAH v14.3.1 — UAP ShadowOptimizer Upgrade

## Core change

`ShadowOptimizer` is now a UAP/ShadowHoTT optimizer with AdamW-classicalized optimizer hygiene built in.

## Added

- Bilateral support geometry:
  - `T_sup`, `F_sup`
  - `M_sup`, `D_det`
  - `K_glut`, `G_gap`
  - `R_obs`, `C_collapse`
- AdamW-like optimizer maturity inside ShadowOptimizer:
  - first moment momentum
  - second moment scaling
  - bias correction
  - decoupled weight decay
  - global gradient clipping
  - update trust-ratio / magnitude control
- `UAPGeometryGate`, an 8-float controller for the classicalization-vs-ShadowHoTT split.
- CLI controls:
  - `--uap-classical-floor`
  - `--uap-classical-ceiling`
  - `--uap-geometry-lr`
  - `--uap-weight-decay`
  - `--hybrid-gate-lr`
  - `--hybrid-min-adamw-weight`
- Metrics for UAP geometry and internal hybrid behavior.
- Test coverage for UAP ShadowOptimizer state and geometry metrics.

## Changed

- `--optimizer shadow` now uses the UAP-upgraded `ShadowOptimizer` while preserving the public class name and v14.2.x state compatibility.
- `--optimizer hybrid` now mixes AdamW with the upgraded UAP ShadowOptimizer and uses slower, safer gate defaults.

## Validation

- `python -m compileall -q .` passed.
- `pytest tests/test_v14_2_8_training_speed_fixes.py -q` passed with 5 tests.
- Debug smoke runs passed for both `--optimizer shadow` and `--optimizer hybrid`.
