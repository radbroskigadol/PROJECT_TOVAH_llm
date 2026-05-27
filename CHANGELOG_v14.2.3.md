# TOVAH v14.2.3 — Explicit Lane B/C Semantic Matching

This maintenance release completes the high-glut gradient routing introduced in v14.2.2.

## Main change

v14.2.2 routed K-heavy examples toward Lane B and G-heavy examples toward Lane C, and relaxed contradiction/gap budgets when metadata said those regimes were genuine. That was useful, but it did not yet force the model's predicted semantic mass to match the corpus metadata.

v14.2.3 adds explicit differentiable matching regularizers:

- `Lane B`: matches predicted contradiction mass `K_pred = mean(min(T,F))` to metadata `K_meta = min(bilateral_t,bilateral_f)`.
- `Lane C`: matches predicted gap mass `G_pred = mean(1-max(T,F))` to metadata `G_meta = min(1-bilateral_t,1-bilateral_f)`.

Both regularizers use a smooth-L1 penalty with a small tolerance dead-zone so mini-batches are not forced into brittle exact equality.

## Loss wiring

The metadata path now includes:

```text
L_total += 0.15 * metadata_weighted_semantic_loss
L_total += 0.05 * lane_routing_loss
L_total += 0.10 * lane_semantic_matching_loss
```

where:

```text
lane_semantic_matching_loss = mean(
    K_meta * _compute_lane_b_regularizer(...)
  + G_meta * _compute_lane_c_regularizer(...)
)
```

Lane D remains a forced-totalization/readout lane and is not targeted by ordinary training.

## Files changed

- `neural/training.py`
  - added `_per_example_semantic_masses`
  - added `_mean_gate_probs`
  - added `_smooth_match_penalty`
  - added `_compute_lane_b_regularizer`
  - added `_compute_lane_c_regularizer`
  - added `lane_semantic_matching_loss`
  - wired matching loss into `train_shadow_step`
- `training/pretrain.py`
  - wired matching loss into `_train_one_batch`
- `tests/test_high_glut_training.py`
  - added Lane B/C semantic matching regression tests
- version constants bumped to `14.2.3`

## Verification

Targeted verification performed in the patch environment:

```text
compileall: passed
all non-test modules import: passed
pytest collection: 477 tests collected
high-glut suite: 13 passed
targeted HoTT/scaling/training suite: 114 passed
selected version/kernel smoke tests: 4 passed
```

A full monolithic pytest run still exceeds the sandbox time limit because several autonomy/research paths are long-running. The changed gradient-routing, HoTT, scaling, pretrain, and version surfaces are covered by targeted tests.
