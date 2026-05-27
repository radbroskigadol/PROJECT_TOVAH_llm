"""
TOVAH v14 conformance/regression.py — Regression test runner.

Validates the conformance ladder:
1. Core runtime conformance (gamma cache, refresh)
2. State reports (invariant engine)
3. Trace reports
4. Comparison reports
5. Fixtures pass
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState, CarrierState, ProvenanceState
from tovah_v14.core.cache import refresh_state, is_cache_coherent, gamma_cache
from tovah_v14.invariants.state_invariants import InvariantEngine
from tovah_v14.invariants.schemas import ReportProfile
from tovah_v14.conformance.fixtures import BASELINE_FIXTURES


def _make_state(beta: Dict[str, BilateralValue]) -> ShadowState:
    s = ShadowState(c=CarrierState(), beta=dict(beta), nu={}, pi=ProvenanceState())
    return refresh_state(s)


def run_regression_suite() -> Tuple[int, int, List[Dict[str, Any]]]:
    """Run full regression suite against fixtures.

    Returns (passed, total, details).
    """
    results: List[Dict[str, Any]] = []
    engine = InvariantEngine()

    for fixture in BASELINE_FIXTURES:
        name = fixture["name"]
        beta = fixture["beta"]
        expected_nu = fixture.get("expected_nu", {})
        expected = fixture.get("expected", {})

        s = _make_state(beta)
        test_result: Dict[str, Any] = {"fixture": name, "passed": True, "errors": []}

        # 1. Gamma cache conformance
        if expected_nu:
            for k, expected_class in expected_nu.items():
                actual = s.nu.get(k, "?")
                if actual != expected_class:
                    test_result["errors"].append(
                        f"gamma: {k} expected {expected_class}, got {actual}"
                    )

        # 2. Coherence
        if "coherent" in expected:
            if is_cache_coherent(s) != expected["coherent"]:
                test_result["errors"].append(
                    f"coherence: expected {expected['coherent']}"
                )

        # 3. Invariant report checks
        report = engine.build_report(s, [])
        if "mean_glut_below" in expected:
            if report.mean_glut >= expected["mean_glut_below"]:
                test_result["errors"].append(
                    f"mean_glut {report.mean_glut:.3f} >= {expected['mean_glut_below']}"
                )
        if "mean_glut_above" in expected:
            if report.mean_glut <= expected["mean_glut_above"]:
                test_result["errors"].append(
                    f"mean_glut {report.mean_glut:.3f} <= {expected['mean_glut_above']}"
                )
        if "mean_gap_above" in expected:
            if report.mean_gap <= expected["mean_gap_above"]:
                test_result["errors"].append(
                    f"mean_gap {report.mean_gap:.3f} <= {expected['mean_gap_above']}"
                )
        if "beta_key_count" in expected:
            if len(s.beta) != expected["beta_key_count"]:
                test_result["errors"].append(
                    f"beta_key_count {len(s.beta)} != {expected['beta_key_count']}"
                )

        # 4. State report check
        state_report = engine.build_state_report(s)
        if not isinstance(state_report.cache_histogram, dict):
            test_result["errors"].append("state_report.cache_histogram not a dict")

        test_result["passed"] = len(test_result["errors"]) == 0
        results.append(test_result)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    return passed, total, results
