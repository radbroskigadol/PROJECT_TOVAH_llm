# TOVAH v14.3.3 — Loop-Stability and Support-Profile Hardening

## Decision

v14.3.3 should not be a longer-training-only update.  The 2500-step real-mixed run preserved UAP/Shadow-depth geometry, but generated continuation still exposed loop/support drift.  The next patch therefore hardens generation dynamics while preserving the successful ShadowOptimizer scale controls.

## What v14.3.3 changes

### 1. Loop-stability layer

Adds `training/loop_stability.py`, which measures:

- unique-token ratio
- repeated 2/3/4/5-gram fractions
- longest same-token run
- loop-drift behavior score
- degeneracy warnings

This is intentionally separate from cross-entropy.  The model may legitimately repeat terms such as truth-support/falsity-support in paradoxical discourse; v14.3.3 only penalizes low-information attractor loops.

### 2. Support-profile layer

Adds `training/shadow_profile_objectives.py`, which canonically represents:

```text
T_support
F_support
glut_mass
gap_mass
obstruction_residue
collapse_pressure
classicalization_depth
```

The helper can infer profile targets from older records with `bilateral_t`, `bilateral_f`, `kind`, `family`, or explicit `uap_profile` fields.  This keeps v14.3.2a corpora usable while enabling real auxiliary heads later.

### 3. Probe status tooling

Adds `tools/check_probe_outputs_v14_3_3.py` and `tools/run_shadow_probe_lengths_v14_3_3.py`.

These exist because the 96/160 probes can look frozen on CPU.  The runner executes one length at a time, while the checker distinguishes complete JSON, missing output, empty output, and partial/invalid JSON.

### 4. Held-out/adversarial corpus tooling

Adds `tools/generate_heldout_adversarial_split_v14_3_3.py`.

This makes `unseen_paradox_family_transfer` meaningful by ensuring selected families do not appear in the train split.  It also creates a conservative adversarial validation shard that changes surface/domain framing while preserving UAP labels.

## What this intentionally does not do

- It does not compare ShadowOptimizer to AdamW.
- It does not change `uap_max_update_rms=1.0` or `uap_trust_clip=0.0`.
- It does not force anti-repetition at decoding time as a substitute for training/eval diagnostics.
- It does not claim source-text Shadow-depth is learned model geometry.

## Recommended next run

1. Check whether 96 and 160 outputs exist and are valid.
2. If missing/incomplete, rerun the generated probes one length at a time.
3. Generate held-out/adversarial split.
4. Train with the same known-good UAP ShadowOptimizer settings plus auxiliary profile/loop gates when wired into `training/pretrain.py`.

## Success criteria

```text
shadow_depth_mean >= 0.94
collapse_pressure <= 0.02
support_profile_consistency >= 0.94
loop_drift_behavior >= 0.82 at 48 tokens
loop_drift_behavior degrades mildly, not sharply, at 96 and 160 tokens
noncollapse_under_gluts >= 0.95 overall
```
