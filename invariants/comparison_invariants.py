"""
TOVAH v14 invariants/comparison_invariants.py — Comparison analysis.

Compares two StateReports or TraceReports to detect:
- regressions (things got worse)
- improvements (things got better)
- key-level changes
- coherence changes

Used for A/B testing of patches and before/after deployment analysis.
"""
from __future__ import annotations

from typing import Dict

from tovah_v14.invariants.schemas import StateReport, ComparisonReport, ReportProfile


def compare_state_reports(
    a: StateReport,
    b: StateReport,
    profile: ReportProfile | None = None,
) -> ComparisonReport:
    """Compare two state reports.

    a is 'before', b is 'after'.
    Positive deltas mean b is larger than a.
    """
    if profile is None:
        profile = ReportProfile()

    glut_delta = b.mean_glut - a.mean_glut
    gap_delta = b.mean_gap - a.mean_gap
    mean_delta_change = b.mean_delta - a.mean_delta

    coherence_change = None
    if a.coherent != b.coherent:
        coherence_change = b.coherent  # True = improved, False = degraded

    # Key-level changes for keys present in both
    key_changes: Dict[str, Dict[str, float]] = {}
    a_det = a.determinized_summary or {}
    b_det = b.determinized_summary or {}
    common_keys = set(a_det.keys()) & set(b_det.keys())
    for k in common_keys:
        diff = b_det[k] - a_det[k]
        if abs(diff) > profile.numeric_tolerance:
            key_changes[k] = {"before": a_det[k], "after": b_det[k], "change": diff}

    regression = (
        glut_delta > profile.glut_warning_threshold
        or gap_delta > profile.gap_warning_threshold
        or coherence_change is False
    )
    improvement = (
        glut_delta < -0.05
        or gap_delta < -0.05
        or coherence_change is True
    )

    notes = []
    if regression:
        notes.append("REGRESSION detected")
    if improvement:
        notes.append("improvement detected")

    return ComparisonReport(
        report_a_id=f"{a.profile_id}:{a.timestamp}",
        report_b_id=f"{b.profile_id}:{b.timestamp}",
        glut_delta=glut_delta,
        gap_delta=gap_delta,
        mean_delta_change=mean_delta_change,
        coherence_change=coherence_change,
        key_changes=key_changes,
        regression_detected=regression,
        improvement_detected=improvement,
        profile_id=profile.profile_id,
        notes=notes,
    )
