"""
TOVAH v14 tests/test_runtime_core.py — Baseline semantic preservation tests.

These tests verify that the core semantic layer produces IDENTICAL results
to v13 for all formulas. If any of these fail, the refactor has broken
the runtime contract.

Run with: python -m pytest tovah_v14/tests/test_runtime_core.py -v
Or: python tovah_v14/tests/test_runtime_core.py
"""
from __future__ import annotations

import math
import sys
import os

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.core.primitives import (
    BilateralValue,
    bilateral_or,
    bilateral_recover,
    coerce_bilateral_value,
)
from tovah_v14.core.lanes import (
    lane_project,
    lane_project_A, lane_project_B, lane_project_C, lane_project_D,
    lane_mixture, lane_divergence,
)
from tovah_v14.core.state import CarrierState, ProvenanceState, ShadowState
from tovah_v14.core.cache import gamma_cache, refresh_state, is_cache_coherent
from tovah_v14.core.determinization import determinize_value, readout_state
from tovah_v14.core.contracts import (
    CONTRACT_REGISTRY, ALLOWED_PATCH_TARGETS, PROTECTED_METHODS,
    verify_patch_contract,
)
from tovah_v14.core.runtime_interface import (
    capture_runtime_view, capture_determinized_view, make_fresh_state, verify_replay,
)
from tovah_v14.core.updates_gate import gate_accumulate, gate_recover, gate_weaken
from tovah_v14.core.updates_measurement import (
    measurement_set, measurement_reset, measurement_determinize, measurement_confidence,
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ============================================================
# BilateralValue tests
# ============================================================
def test_bilateral_value_basic():
    v = BilateralValue(0.7, 0.3)
    assert _close(v.t, 0.7)
    assert _close(v.f, 0.3)
    assert _close(v.glut, 0.3)
    assert _close(v.gap, 0.3)
    assert _close(v.delta, 0.4)


def test_bilateral_value_clamp():
    v = BilateralValue(1.5, -0.3).clamp()
    assert _close(v.t, 1.0)
    assert _close(v.f, 0.0)


def test_bilateral_value_nan_safety():
    v = BilateralValue(float("nan"), float("inf")).clamp()
    assert _close(v.t, 0.0)
    assert _close(v.f, 0.0)


def test_bilateral_value_repr():
    v = BilateralValue(0.8, 0.2)
    r = repr(v)
    assert "BV(" in r
    assert "0.800" in r


def test_bilateral_value_eq():
    a = BilateralValue(0.5, 0.3)
    b = BilateralValue(0.5, 0.3)
    c = BilateralValue(0.5, 0.4)
    assert a == b
    assert a != c


# ============================================================
# bilateral_or tests — exact formula verification
# ============================================================
def test_bilateral_or_formula():
    a = BilateralValue(0.6, 0.3)
    b = BilateralValue(0.4, 0.5)
    r = bilateral_or(a, b)
    # t = 0.6 + 0.4 - 0.6*0.4 = 0.76
    # f = 0.3 + 0.5 - 0.3*0.5 = 0.65
    assert _close(r.t, 0.76)
    assert _close(r.f, 0.65)


def test_bilateral_or_identity():
    a = BilateralValue(0.5, 0.5)
    z = BilateralValue(0.0, 0.0)
    r = bilateral_or(a, z)
    assert _close(r.t, 0.5)
    assert _close(r.f, 0.5)


def test_bilateral_or_saturation():
    a = BilateralValue(1.0, 1.0)
    b = BilateralValue(0.5, 0.5)
    r = bilateral_or(a, b)
    assert _close(r.t, 1.0)
    assert _close(r.f, 1.0)


# ============================================================
# bilateral_recover tests — exact formula verification
# ============================================================
def test_bilateral_recover_formula():
    v = BilateralValue(0.4, 0.6)
    r = bilateral_recover(v, truth_gain=0.3, falsity_decay=0.2)
    # t = 0.4 + 0.3 - 0.4*0.3 = 0.58
    # f = 0.6 * max(0, 1 - 0.2) = 0.6 * 0.8 = 0.48
    assert _close(r.t, 0.58)
    assert _close(r.f, 0.48)


def test_bilateral_recover_zero():
    v = BilateralValue(0.5, 0.5)
    r = bilateral_recover(v, truth_gain=0.0, falsity_decay=0.0)
    assert _close(r.t, 0.5)
    assert _close(r.f, 0.5)


# ============================================================
# Lane projection tests — exact formula verification
# ============================================================
def test_lane_A():
    t, f = lane_project_A(0.8, 0.3)
    assert _close(t, 0.8 * 0.7)  # 0.56
    assert _close(f, 0.3 * 0.2)  # 0.06


def test_lane_B():
    t, f = lane_project_B(0.8, 0.3)
    assert _close(t, max(0.8, 0.3))  # 0.8
    assert _close(f, 0.3 * 0.2)      # 0.06


def test_lane_C():
    t, f = lane_project_C(0.8, 0.3)
    assert _close(t, 0.8 * 0.7)                          # 0.56
    assert _close(f, max(0.3, 0.2) * (1.0 - 0.8 * 0.3)) # 0.3 * 0.76 = 0.228


def test_lane_D():
    t, f = lane_project_D(0.8, 0.3)
    assert _close(t, 0.8)
    assert _close(f, 0.2)


def test_lane_project_dispatch():
    """Verify lane_project dispatches to the same formulas."""
    for lane, fn in [("A", lane_project_A), ("B", lane_project_B),
                     ("C", lane_project_C), ("D", lane_project_D)]:
        r1 = lane_project(0.7, 0.4, lane)
        r2 = fn(0.7, 0.4)
        assert _close(r1[0], r2[0]) and _close(r1[1], r2[1]), f"lane {lane} mismatch"


def test_lane_divergence():
    d = lane_divergence(0.5, 0.5)
    assert "A" in d and "B" in d and "C" in d and "D" in d and "spread" in d


# ============================================================
# Gamma cache tests
# ============================================================
def test_gamma_cache_classifications():
    beta = {
        "high_t": BilateralValue(0.8, 0.1),   # T
        "high_f": BilateralValue(0.1, 0.8),   # F
        "both":   BilateralValue(0.8, 0.8),   # B
        "gap":    BilateralValue(0.1, 0.1),   # G
    }
    nu = gamma_cache(beta)
    assert nu["high_t"] == "T"
    assert nu["high_f"] == "F"
    assert nu["both"] == "B"
    assert nu["gap"] == "G"


def test_gamma_threshold_boundary():
    beta = {
        "exact": BilateralValue(0.55, 0.55),
    }
    nu = gamma_cache(beta, theta_t=0.55, theta_f=0.55)
    assert nu["exact"] == "B"  # both at threshold -> B


# ============================================================
# refresh_state / coherence tests
# ============================================================
def test_refresh_coherence():
    s = ShadowState(
        c=CarrierState(),
        beta={"k1": BilateralValue(0.8, 0.1)},
        nu={},
        pi=ProvenanceState(),
    )
    refresh_state(s)
    assert is_cache_coherent(s)
    assert s.nu["k1"] == "T"


def test_refresh_coerces_dicts():
    """v13 migration: beta may contain plain dicts instead of BilateralValues."""
    s = ShadowState(
        c=CarrierState(),
        beta={"k1": {"t": 0.9, "f": 0.05}},  # type: ignore
        nu={},
        pi=ProvenanceState(),
    )
    refresh_state(s)
    assert isinstance(s.beta["k1"], BilateralValue)
    assert _close(s.beta["k1"].t, 0.9)


def test_refresh_increments_provenance():
    s = make_fresh_state(["a", "b"])
    old_count = s.pi.refresh_count
    refresh_state(s)
    assert s.pi.refresh_count == old_count + 1


def test_incoherent_after_manual_mutation():
    s = make_fresh_state(["k"])
    assert is_cache_coherent(s)
    s.beta["k"] = BilateralValue(0.99, 0.01)
    # nu is now stale
    assert not is_cache_coherent(s)
    refresh_state(s)
    assert is_cache_coherent(s)


# ============================================================
# Coerce tests (migration safety)
# ============================================================
def test_coerce_from_dict():
    v = coerce_bilateral_value({"t": 0.7, "f": 0.2})
    assert _close(v.t, 0.7)
    assert _close(v.f, 0.2)


def test_coerce_from_bilateral():
    orig = BilateralValue(0.6, 0.4)
    v = coerce_bilateral_value(orig)
    assert _close(v.t, 0.6)


def test_coerce_from_garbage():
    v = coerce_bilateral_value("not a number", default_t=0.3, default_f=0.1)
    assert _close(v.t, 0.3)
    assert _close(v.f, 0.1)


def test_coerce_nan():
    v = coerce_bilateral_value({"t": float("nan"), "f": 0.5})
    assert math.isfinite(v.t)
    assert _close(v.f, 0.5)


# ============================================================
# Contract registry tests
# ============================================================
def test_all_patch_targets_have_contracts():
    for target in ALLOWED_PATCH_TARGETS:
        assert target in CONTRACT_REGISTRY, f"missing contract: {target}"


def test_protected_not_patchable():
    for method in PROTECTED_METHODS:
        assert method not in ALLOWED_PATCH_TARGETS, f"protected method in patch targets: {method}"


def test_verify_patch_contract_basic():
    code = """
def research_topic(self, topic, context=""):
    self.state.beta["research.novelty"] = bilateral_recover(
        self.state.beta.get("research.novelty", BilateralValue()), truth_gain=0.1)
    refresh_state(self.state)
    return []
"""
    ok, errors = verify_patch_contract("research_topic", code)
    assert ok, f"unexpected errors: {errors}"


def test_verify_patch_contract_forbidden():
    code = """
def research_topic(self, topic, context=""):
    self.tool_layer.search(topic)
    return []
"""
    ok, errors = verify_patch_contract("research_topic", code)
    assert not ok
    assert any("forbidden" in e for e in errors)


def test_verify_patch_contract_missing_refresh():
    code = """
def _classify_query_intent(self, text):
    self.state.beta["intent"] = BilateralValue(0.5, 0.5)
    return "broad_research"
"""
    ok, errors = verify_patch_contract("_classify_query_intent", code)
    assert not ok
    assert any("refresh_state" in e for e in errors)


# ============================================================
# Gate-like vs measurement-like update tests
# ============================================================
def test_gate_accumulate():
    r = gate_accumulate(BilateralValue(0.5, 0.2), BilateralValue(0.3, 0.1))
    assert r.t > 0.5  # accumulated
    assert r.f > 0.2


def test_gate_weaken():
    r = gate_weaken(BilateralValue(0.8, 0.2), truth_decay=0.5, falsity_gain=0.3)
    assert r.t < 0.8
    assert r.f > 0.2


def test_measurement_set():
    r = measurement_set(0.9, 0.1)
    assert _close(r.t, 0.9)
    assert _close(r.f, 0.1)


def test_measurement_determinize():
    v = BilateralValue(0.8, 0.2)
    d = measurement_determinize(v)
    # 0.5 + 0.5 * 0.6 = 0.8
    assert _close(d, 0.8)


def test_measurement_confidence():
    v = BilateralValue(0.9, 0.1)
    c = measurement_confidence(v)
    assert c > 0.5  # high delta, low glut -> high confidence


# ============================================================
# Determinization tests
# ============================================================
def test_determinize_preserves_bilateral():
    """Determinization must not change the stored state."""
    s = make_fresh_state(["k"])
    s.beta["k"] = BilateralValue(0.8, 0.3)
    refresh_state(s)
    view = readout_state(s)
    # bilateral state unchanged
    assert _close(s.beta["k"].t, 0.8)
    assert _close(s.beta["k"].f, 0.3)


# ============================================================
# Runtime interface tests
# ============================================================
def test_capture_views():
    s = make_fresh_state(["a", "b", "c"])
    rv = capture_runtime_view(s)
    assert rv.coherent
    assert len(rv.beta_snapshot) == 3

    dv = capture_determinized_view(s)
    assert len(dv.confidence_map) == 3


def test_verify_replay_coherent():
    s = make_fresh_state(["a"])
    before = capture_runtime_view(s)
    s.beta["a"] = BilateralValue(0.9, 0.1)
    refresh_state(s)
    after = capture_runtime_view(s)
    report = verify_replay(before, after, "test_update")
    assert report["ok"]
    assert report["provenance_advanced"]


# ============================================================
# State snapshot roundtrip
# ============================================================
def test_state_snapshot_roundtrip():
    s = make_fresh_state(["x", "y"])
    s.beta["x"] = BilateralValue(0.7, 0.2)
    s.beta["y"] = BilateralValue(0.3, 0.8)
    refresh_state(s)
    snap = s.snapshot()

    # Reconstruct
    from tovah_v14.core.state import ShadowState, CarrierState, ProvenanceState
    s2 = ShadowState(
        c=CarrierState(**snap["c"]),
        beta={k: coerce_bilateral_value(v) for k, v in snap["beta"].items()},
        nu=snap["nu"],
        pi=ProvenanceState(**snap["pi"]),
    )
    assert _close(s2.beta["x"].t, 0.7)
    assert _close(s2.beta["y"].f, 0.8)
    assert is_cache_coherent(s2)


# ============================================================
# Runner
# ============================================================
def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
