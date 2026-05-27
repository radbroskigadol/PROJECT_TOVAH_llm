"""
TOVAH v14 invariants/trace_invariants.py — Trace-level invariant analysis.

Tracks state evolution over sequences of transitions.
Detects shocks (large sudden changes), trajectory trends,
and coherence stability.

Pure analysis — no state mutation.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Sequence

from tovah_v14.core.state import ShadowState
from tovah_v14.core.cache import is_cache_coherent
from tovah_v14.invariants.schemas import TraceReport, ReportProfile


class TraceAnalyzer:
    """Analyzes sequences of state snapshots."""

    def __init__(self, profile: ReportProfile | None = None):
        self.profile = profile or ReportProfile()
        self._glut_history: List[float] = []
        self._gap_history: List[float] = []
        self._delta_history: List[float] = []
        self._coherence_history: List[bool] = []
        self._loss_history: List[float] = []
        self._shocks: List[Dict[str, Any]] = []
        self._step_count = 0
        self._start_time = dt.datetime.now().isoformat(timespec="seconds")

    def record_step(self, s: ShadowState, loss: float = 0.0) -> None:
        """Record one state snapshot in the trace."""
        n = max(1, len(s.beta))
        mean_glut = sum(v.glut for v in s.beta.values()) / n if s.beta else 0.0
        mean_gap = sum(v.gap for v in s.beta.values()) / n if s.beta else 0.0
        mean_delta = sum(v.delta for v in s.beta.values()) / n if s.beta else 0.0

        self._glut_history.append(mean_glut)
        self._gap_history.append(mean_gap)
        self._delta_history.append(mean_delta)
        self._coherence_history.append(is_cache_coherent(s))
        self._loss_history.append(loss)

        # Shock detection
        if len(self._delta_history) >= 2:
            change = abs(self._delta_history[-1] - self._delta_history[-2])
            if change > self.profile.shock_threshold:
                self._shocks.append({
                    "step": self._step_count,
                    "delta_change": change,
                    "from_delta": self._delta_history[-2],
                    "to_delta": self._delta_history[-1],
                })

        self._step_count += 1

    def build_report(self, trace_id: str = "") -> TraceReport:
        """Build trace report from recorded steps."""
        if not trace_id:
            trace_id = f"trace_{int(dt.datetime.now().timestamp())}"

        notes: List[str] = []
        if self._shocks:
            notes.append(f"{len(self._shocks)} shocks detected")
        incoherent_count = sum(1 for c in self._coherence_history if not c)
        if incoherent_count > 0:
            notes.append(f"{incoherent_count} incoherent steps")

        return TraceReport(
            trace_id=trace_id,
            start_timestamp=self._start_time,
            end_timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            step_count=self._step_count,
            glut_trajectory=list(self._glut_history),
            gap_trajectory=list(self._gap_history),
            delta_trajectory=list(self._delta_history),
            coherence_trajectory=list(self._coherence_history),
            shocks=list(self._shocks),
            loss_trajectory=list(self._loss_history),
            profile_id=self.profile.profile_id,
            notes=notes,
        )
