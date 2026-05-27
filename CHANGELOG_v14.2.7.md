# TOVAH v14.2.7 — Audit Fixes

Five fixes derived from the v14.2.6 audit. Each lands behind tests; the
full pre-existing test suite still passes (515/515 green).

## Summary

| # | Area | File(s) | Tests | Risk |
|---|------|---------|-------|------|
| 1 | Reservoir shuffle in streaming dataset | `training/dataset.py` | `TestReservoirShuffle` | low |
| 2 | RC-1 sovereign-default inversion | `mutation/promotion_ladder.py`, `kernel/kernel.py` | `TestRC1Inversion` | medium |
| 3 | Persist-on-evict for rolling buffers | `debug/trace_writer.py` (new), `mutation/promotion_ladder.py`, `modules/registry.py`, `modules/bus_contracts.py` | `TestPersistOnEvict` | low |
| 4 | CalibrationProfile constant lift (Phase 1) | `modules/calibration.py` (new), `modules/registry.py`, `mutation/promotion_ladder.py` | `TestCalibrationProfile` | low (no numerical drift) |
| 5 | Cellular sheaf observer | `modules/sheaf_observer.py` (new) | `TestSheafObserver` | none (read-only) |

## 1. Reservoir shuffle (`training/dataset.py`)

The v14.2.6 `CorpusShardDataset` partitioned shards round-robin among
workers and then read each worker's shards top-to-bottom — no within-worker
shuffle. Gradient updates saw systematic file-level ordering bias.

**Fix:** added a reservoir-sampling buffer inside the worker loop. New
`reservoir_size` constructor parameter (default 8192). Set to 0 or 1 to
restore the legacy deterministic order (used for ablation / eval).

The reservoir uses an evict-and-emit scheme that gives an exact uniform
mix over a sliding window of `cap` examples, with O(1) memory per item.

## 2. RC-1 sovereign-default inversion (`mutation/promotion_ladder.py`)

`_source_context` in v14.2.6 returned `source_role='main'`,
`trust_level='sovereign'`, `risk_class='low'` whenever a patch had no
registered `source_metadata`. `_adaptive_gate_checks` exempted
sovereign-main from cooldown, success-rate, budget, and failure-weight
checks. Result: any code path that staged a patch name without
explicit `set_source_metadata()` got full bypass — fail-OPEN.

**Fix:** inverted the default. Empty metadata now yields
`source_role='subkernel'`, `trust_level='provisional'`,
`risk_class='medium'`. A warning is logged on each unaccounted call
so missing registrations surface in operational logs.

All legitimate sovereign paths (`direct_inject_method`,
`_stage_patch_proposal`) already set sovereign metadata before the gate.
The kernel's `stage_patch` route was the one un-attested caller in the
v14.2.6 tree; it now explicitly registers
`source_role='main', trust_level='sovereign'` for internal staging
so legitimate kernel-initiated patches retain their authority.

**Test-suite impact:** three pre-existing tests
(`test_kernel_apply_through_ladder`, `test_promotion_stage_ordering`,
`test_full_promotion_lifecycle`) and the `test_hott_promotion_wiring`
helper relied on the implicit-sovereign default. Each was updated to
declare sovereign metadata explicitly. This is the desired surfacing:
the bug was that these paths were granted authority silently.

## 3. Persist-on-evict (`debug/trace_writer.py` + four call sites)

The v14.2.6 rolling buffers (`gate_log[-200:]`, `proposal_history[-500:]`,
`message_log[-200:]`, `evidence_log[patch][-100:]`, `history[-500:]`)
dropped oldest entries silently. Under high-velocity execution this
discarded root-cause evidence before observers/critics could consume it.

**Fix:** added `tovah_v14/debug/trace_writer.py`. Each truncation site
now persists the soon-to-be-dropped records to an append-only NDJSON
trace under `tovah_traces/` before truncating. Caps use a cushion
(e.g. cap=200 + cushion=50 → batch evict 50 at once) so disk I/O is
amortized rather than per-record.

**Invariant tested:** every record is either in memory or persisted,
never both lost. See `test_bus_log_persists_overflow_with_no_gap`.

Persisted traces:

- `tovah_traces/promotion_history.ndjson`
- `tovah_traces/promotion_evidence.ndjson` (each record tagged with `patch_name`)
- `tovah_traces/promotion_gate_log.ndjson`
- `tovah_traces/module_proposal_history.ndjson`
- `tovah_traces/module_bus_log.ndjson`
- `tovah_traces/sheaf_observer_findings.ndjson` (from §5)

`TraceWriter` failures are swallowed and warning-throttled — a failure
to persist a trace MUST NOT crash the kernel.

## 4. CalibrationProfile Phase 1 (`modules/calibration.py`)

The v14.2.6 feedback and gate code contained ~30 numeric literals (the
audit cited 0.55, 0.45, 0.80; the full count is closer to 30) embedded
inline. This made the constants un-auditable and un-tunable.

**Fix — Phase 1 (this commit):** lifted every numeric literal into
named fields on two dataclasses:

- `ModuleFeedbackCalibration` — coefficients for ModuleRegistry feedback,
  family carry-over, and quality scoring.
- `AdaptiveGateCalibration` — thresholds for the PromotionLadder
  adaptive gates (per-stage `required_evidence`, `min_success_rate`,
  `max_budget_pressure`, `max_recent_failure_weight`).

`ModuleRegistry._evidence_quality_total` and
`PromotionLadder._adaptive_gate_checks` now consume these profiles.
The default profile values are **bit-identical** to v14.2.6 numerics
(locked by `test_evidence_quality_total_parity_with_explicit_default`
and the explicit-constants tests in `TestCalibrationProfile`).

**Out of scope for this commit (Phase 2/3):** binding profile fields
to module-family bilateral state. The full bilateral parameterization
requires care around feedback dynamics (a module that performs well
once should not capture a multiplier that helps it perform "well" again)
and is staged separately. Phase 1 is the refactor-without-behavior-change
that makes the bilateral binding tractable and reviewable.

## 5. Cellular sheaf observer (`modules/sheaf_observer.py`)

A read-only diagnostic module that models the `ModuleRegistry` dependency
graph as a discrete cellular sheaf. Each module is a vertex; stalks hold
a triple `(t, f, q)` derived from operational metrics; restriction maps
are built from interface-contract overlap. The global obstruction is the
sum of squared disagreements across edges.

**Behavior:**

- `assess()` computes the obstruction and classifies it as
  `ok` / `drift` / `glut` (latter = concentrated divergence on few edges).
- Findings persist to `tovah_traces/sheaf_observer_findings.ndjson`.
- Glut findings emit a WARNING log line identifying the hot edges.
- The observer subscribes to the ladder's `on_gate_decision` hook (if
  bound) to reassess after each gate decision.

**Critically: the observer is strictly diagnostic.** It does not call
`set_source_metadata`, `record_evidence`, or any gate. Its presence
must not change any gate decision — locked by
`test_observer_does_not_influence_gate`.

**Provenance:** Bodnar et al. (2022), *Neural Sheaf Diffusion*; Hansen
& Gebhart (2020), *Sheaf Neural Networks*. The TOVAH specialization is
that stalks are bilateral `(t, f)` pairs (plus an evidence-quality
channel) and restriction maps are derived from interface overlap rather
than learned. A learned-restriction-map version is future work.

## Test results

```
$ python -m pytest tovah_v14/tests/
515 passed in 43.63s
```

- 496 pre-existing tests still green (4 test-helpers updated to declare
  metadata explicitly; see §2).
- 19 new tests in `tests/test_audit_v14_2_7.py` covering all five fixes.

## Files changed

```
new:    tovah_v14/debug/trace_writer.py
new:    tovah_v14/modules/calibration.py
new:    tovah_v14/modules/sheaf_observer.py
new:    tovah_v14/tests/test_audit_v14_2_7.py
new:    tovah_v14/CHANGELOG_v14.2.7.md
mod:    tovah_v14/training/dataset.py       (reservoir shuffle)
mod:    tovah_v14/mutation/promotion_ladder.py  (RC-1, persist-on-evict, calibration)
mod:    tovah_v14/modules/registry.py       (persist-on-evict, calibration)
mod:    tovah_v14/modules/bus_contracts.py  (persist-on-evict)
mod:    tovah_v14/kernel/kernel.py          (stage_patch metadata registration)
mod:    tovah_v14/tests/test_mutation.py    (updated for RC-1 inversion)
mod:    tovah_v14/tests/test_hott_promotion_wiring.py  (updated for RC-1 inversion)
```
