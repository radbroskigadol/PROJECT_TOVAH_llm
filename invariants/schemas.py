"""
TOVAH v14 invariants/schemas.py — Typed report and certificate schemas.

These are BRIDGE objects that cross the boundary between
the bilateral evidence layer and the report/observation layer.
They are plain data — no side effects, no mutation.

SEMANTIC PRESERVATION:
  Certificate and InvariantReport shapes are identical to v13.
  Added: StateReport, TraceReport, ComparisonReport, ReportProfile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Certificate:
    """Certification record. Shape preserved from v13."""
    cert_kind: str
    version: str
    domain_id: str
    witness: Dict[str, Any]
    context: Dict[str, Any]
    created_at: str


@dataclass
class StateReport:
    """Full state report at a point in time.

    Includes bilateral diagnostics, cache histogram, contradiction/gap analysis,
    and determinized views. Pure data — computed by InvariantEngine.
    """
    timestamp: str
    coherent: bool
    cache_histogram: Dict[str, int]
    beta_key_count: int
    mean_glut: float
    mean_gap: float
    mean_delta: float
    contradiction_keys: List[str]
    gap_keys: List[str]
    lane_divergence_summary: Dict[str, float] = field(default_factory=dict)
    determinized_summary: Dict[str, float] = field(default_factory=dict)
    profile_id: str = "default"
    notes: List[str] = field(default_factory=list)


@dataclass
class TraceReport:
    """Report over a sequence of state transitions.

    Records trajectory signatures, shock detection, and
    bilateral mass evolution.
    """
    trace_id: str
    start_timestamp: str
    end_timestamp: str
    step_count: int
    glut_trajectory: List[float] = field(default_factory=list)
    gap_trajectory: List[float] = field(default_factory=list)
    delta_trajectory: List[float] = field(default_factory=list)
    coherence_trajectory: List[bool] = field(default_factory=list)
    shocks: List[Dict[str, Any]] = field(default_factory=list)
    loss_trajectory: List[float] = field(default_factory=list)
    profile_id: str = "default"
    notes: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comparison between two state reports or trace reports.

    Used for A/B testing of patches, before/after analysis,
    and regression detection.
    """
    report_a_id: str
    report_b_id: str
    glut_delta: float = 0.0
    gap_delta: float = 0.0
    mean_delta_change: float = 0.0
    coherence_change: Optional[bool] = None  # None=same, True=improved, False=degraded
    key_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regression_detected: bool = False
    improvement_detected: bool = False
    profile_id: str = "default"
    notes: List[str] = field(default_factory=list)


@dataclass
class ReportProfile:
    """Defines tolerances and thresholds for report generation.

    Different profiles allow different sensitivity levels.
    The 'default' profile uses v13 thresholds.
    """
    profile_id: str = "default"
    glut_warning_threshold: float = 0.25
    glut_critical_threshold: float = 0.45
    gap_warning_threshold: float = 0.25
    gap_critical_threshold: float = 0.45
    shock_threshold: float = 0.3  # delta change per step that counts as shock
    numeric_tolerance: float = 1e-9  # for comparison reports
