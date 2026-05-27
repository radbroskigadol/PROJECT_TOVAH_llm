# TOVAH v14.3.3 — Loop-Stability / Support-Profile Hardening

## Summary

v14.3.3 hardens the generated-continuation failure mode found after the v14.3.2a real-mixed 2500-step run.  The update preserves the framing that TOVAH is a ShadowHoTT/UAP token-ontology trainer, not an AdamW-vs-ShadowOptimizer benchmark.

## Code changes

- Added `training/loop_stability.py` with explicit 2/3/4/5-gram repetition diagnostics, longest-run detection, and an optional logits-concentration auxiliary penalty.
- Added `training/shadow_profile_objectives.py` for canonical support-profile target inference and auxiliary profile losses.
- Wired the v14.3.3 loop diagnostic into `tools/uap_shadow_profiles.py`, so generated Shadow-depth loop scores now use the stronger multi-ngram loop detector.
- Wired generated-continuation loop summaries into `training/shadow_depth_eval.py` and `training/eval.py`.
- Added optional `uap_loop_penalty_weight` to `training/pretrain.py` and `--uap-loop-penalty-weight` to `run_tovah.py`. Default is `0.0` to avoid perturbing known-good training runs unless explicitly enabled.
- Added probe-check/rerun tools for the stalled 96/160 generated probes.
- Added held-out/adversarial family split tooling so `unseen_paradox_family_transfer` can become meaningful.

## Operational note

For the immediate diagnostic, first run the probe checker. Rerun missing or partial 96/160 outputs with the sequential runner. Only enable a nonzero loop penalty after confirming drift persists across the length curve.
