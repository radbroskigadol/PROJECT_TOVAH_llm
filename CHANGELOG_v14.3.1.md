# TOVAH v14.3.1 — UAP ShadowOptimizer magnitude-control fix

v14.3.0 correctly introduced the UAP/ShadowHoTT optimizer shape, but its
stable magnitude-control layer used an **absolute tensor-norm cap**. For large
parameter matrices, that cap divided AdamW-like sign updates by roughly
`sqrt(numel)`, making Shadow-only training effectively immobile on BPE/heavy
runs.

v14.3.1 fixes this without abandoning the UAP optimizer:

- Replaces the default absolute tensor-norm cap with a size-invariant
  `max_update_rms` cap.
- Disables LAMB-style trust-ratio suppression by default (`uap_trust_clip=0`).
- Keeps optional trust ratio available as an explicit flag.
- Aligns UAP ShadowOptimizer default base LR with AdamW (`3e-4`) so the
  classicalized projection can actually recover AdamW-like scale.
- Adds metrics:
  - `uap_update_rms_mean`
  - `uap_effective_scale_mean`
  - `uap_max_update_rms`
- Adds CLI flags:
  - `--uap-max-update-rms`
  - `--uap-trust-clip`

Validation:

- `python -m compileall -q .`
- `pytest tests/test_v14_2_8_training_speed_fixes.py -q` → 5 passed
- Debug Shadow-only smoke moved loss from ~9.78 to ~8.32 over 5 updates,
  confirming the optimizer is no longer update-scale frozen.
