"""
TOVAH v14 tests/test_invariants.py — Invariant, report, and conformance tests.

Verifies:
- InvariantEngine build_report preserves v13 shape
- StateReport includes lane divergence and determinized views
- TraceAnalyzer detects shocks
- ComparisonReport detects regressions
- CertificationLayer issues and checks all cert kinds
- ContradictionDiagnostic classifies correctly
- GlutHygieneReport structure
- Conformance fixtures all pass
- Regression suite passes
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState, CarrierState, ProvenanceState
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.runtime_interface import make_fresh_state
from tovah_v14.invariants.state_invariants import InvariantEngine, InvariantReport
from tovah_v14.invariants.schemas import StateReport, TraceReport, ComparisonReport, ReportProfile
from tovah_v14.invariants.trace_invariants import TraceAnalyzer
from tovah_v14.invariants.comparison_invariants import compare_state_reports
from tovah_v14.invariants.certification import CertificationLayer
from tovah_v14.invariants.contradiction import (
    classify_contradiction, recommend_action, diagnose_contradictions,
    build_hygiene_report, ContradictionDiagnostic,
)
from tovah_v14.conformance.regression import run_regression_suite


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def _make_state(beta_dict):
    s = ShadowState(c=CarrierState(), beta=dict(beta_dict), nu={}, pi=ProvenanceState())
    return refresh_state(s)


# ============================================================
# InvariantEngine tests (v13 compat)
# ============================================================
def test_build_report_shape():
    engine = InvariantEngine()
    s = _make_state({"k": BilateralValue(0.8, 0.2)})
    report = engine.build_report(s, [1.0, 0.8, 0.6])
    assert isinstance(report, InvariantReport)
    assert report.coherent is True
    assert "T" in report.cache_histogram
    assert isinstance(report.trajectory_signature, dict)
    assert "loss_mean_8" in report.trajectory_signature


def test_build_report_glut_detection():
    engine = InvariantEngine()
    s = _make_state({"contradicted": BilateralValue(0.8, 0.8)})
    report = engine.build_report(s, [])
    assert "contradicted" in report.contradiction_keys


def test_build_report_gap_detection():
    engine = InvariantEngine()
    s = _make_state({"uncertain": BilateralValue(0.1, 0.1)})
    report = engine.build_report(s, [])
    assert "uncertain" in report.gap_keys


# ============================================================
# StateReport tests (v14 enriched)
# ============================================================
def test_state_report_includes_lanes():
    engine = InvariantEngine()
    s = _make_state({"a": BilateralValue(0.7, 0.3), "b": BilateralValue(0.5, 0.5)})
    report = engine.build_state_report(s)
    assert isinstance(report, StateReport)
    assert "mean_spread" in report.lane_divergence_summary
    assert len(report.determinized_summary) > 0


def test_state_report_custom_profile():
    engine = InvariantEngine()
    profile = ReportProfile(glut_critical_threshold=0.2)
    s = _make_state({"mild_glut": BilateralValue(0.6, 0.3)})
    report = engine.build_state_report(s, profile=profile)
    # 0.3 glut > 0.2 threshold
    assert "mild_glut" in report.contradiction_keys


# ============================================================
# TraceAnalyzer tests
# ============================================================
def test_trace_basic():
    ta = TraceAnalyzer()
    s = _make_state({"k": BilateralValue(0.5, 0.5)})
    ta.record_step(s, loss=1.0)
    ta.record_step(s, loss=0.8)
    report = ta.build_report("test_trace")
    assert isinstance(report, TraceReport)
    assert report.step_count == 2
    assert len(report.loss_trajectory) == 2


def test_trace_shock_detection():
    ta = TraceAnalyzer(profile=ReportProfile(shock_threshold=0.1))
    s1 = _make_state({"k": BilateralValue(0.5, 0.5)})
    ta.record_step(s1)
    # Big shift
    s2 = _make_state({"k": BilateralValue(0.95, 0.05)})
    ta.record_step(s2)
    report = ta.build_report()
    assert len(report.shocks) >= 1


# ============================================================
# Comparison tests
# ============================================================
def test_comparison_regression():
    engine = InvariantEngine()
    s1 = _make_state({"k": BilateralValue(0.8, 0.1)})
    s2 = _make_state({"k": BilateralValue(0.4, 0.7)})  # degraded
    r1 = engine.build_state_report(s1)
    r2 = engine.build_state_report(s2)
    comp = compare_state_reports(r1, r2)
    assert isinstance(comp, ComparisonReport)
    assert comp.glut_delta > 0 or comp.gap_delta > 0


def test_comparison_improvement():
    engine = InvariantEngine()
    s1 = _make_state({"k": BilateralValue(0.4, 0.6)})
    s2 = _make_state({"k": BilateralValue(0.9, 0.1)})  # improved
    r1 = engine.build_state_report(s1)
    r2 = engine.build_state_report(s2)
    comp = compare_state_reports(r1, r2)
    assert comp.improvement_detected or comp.glut_delta <= 0


# ============================================================
# Certification tests
# ============================================================
def test_certify_state():
    certs = CertificationLayer()
    s = _make_state({"k": BilateralValue(0.8, 0.1)})
    cert = certs.certify_state(s)
    ok, msg = certs.check(cert)
    assert ok, msg


def test_certify_report():
    engine = InvariantEngine()
    certs = CertificationLayer()
    s = _make_state({"k": BilateralValue(0.8, 0.1)})
    report = engine.build_report(s, [])
    cert = certs.certify_report(report)
    ok, msg = certs.check(cert)
    assert ok, msg


def test_certify_patch_contract():
    certs = CertificationLayer()
    cert = certs.certify_patch_contract("research_topic", True, True, [])
    ok, msg = certs.check(cert)
    assert ok, msg

    cert_bad = certs.certify_patch_contract("research_topic", True, False, ["error"])
    ok2, msg2 = certs.check(cert_bad)
    assert not ok2


# ============================================================
# Contradiction diagnostics
# ============================================================
def test_classify_destabilizing():
    v = BilateralValue(0.8, 0.8)  # high glut, near-zero delta
    assert classify_contradiction("k", v) == "destabilizing"


def test_classify_informative():
    v = BilateralValue(0.7, 0.4)  # moderate glut, clear delta
    assert classify_contradiction("k", v) == "informative"


def test_classify_transient():
    v = BilateralValue(0.6, 0.1)  # low glut
    assert classify_contradiction("k", v) == "transient"


def test_recommend_escalate_runtime():
    assert recommend_action("destabilizing", "runtime.stability") == "escalate"


def test_recommend_dampen_non_critical():
    assert recommend_action("destabilizing", "tool.search_efficacy") == "dampen"


def test_diagnose_contradictions():
    s = _make_state({
        "clean": BilateralValue(0.9, 0.1),
        "contradicted": BilateralValue(0.8, 0.8),
        "uncertain": BilateralValue(0.1, 0.1),
    })
    diags = diagnose_contradictions(s, glut_threshold=0.25, gap_threshold=0.30)
    assert any(d.key == "contradicted" for d in diags)
    # uncertain has gap=0.9 which is > 0.30
    assert any(d.key == "uncertain" for d in diags)
    # clean should not appear
    assert not any(d.key == "clean" for d in diags)


def test_hygiene_report():
    s = _make_state({
        "good": BilateralValue(0.9, 0.1),
        "bad": BilateralValue(0.8, 0.8),
    })
    report = build_hygiene_report(s)
    assert report.total_keys == 2
    assert report.glut_keys >= 1
    assert report.destabilizing_contradictions >= 1


# ============================================================
# Conformance regression suite
# ============================================================
def test_conformance_regression():
    passed, total, details = run_regression_suite()
    failed = [d for d in details if not d["passed"]]
    if failed:
        for f in failed:
            print(f"  FIXTURE FAIL: {f['fixture']}: {f['errors']}")
    assert passed == total, f"{passed}/{total} fixtures passed"


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
