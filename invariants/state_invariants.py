"""
TOVAH v14 invariants/state_invariants.py — State-level invariant analysis.

SEMANTIC PRESERVATION:
  InvariantReport shape is identical to v13.
  InvariantEngine.build_report produces identical output for identical input.

v14 ADDITIONS:
  - build_state_report() produces the richer StateReport
  - lane_divergence_summary computed per key
  - determinized_summary included
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState
from tovah_v14.core.cache import is_cache_coherent, gamma_cache
from tovah_v14.core.lanes import lane_divergence
from tovah_v14.core.determinization import determinize_value
from tovah_v14.invariants.schemas import StateReport, ReportProfile


@dataclass
class InvariantReport:
    """v13-compatible invariant report. Shape preserved exactly."""
    timestamp: str
    coherent: bool
    support_profile: Dict[str, Dict[str, float]]
    cache_histogram: Dict[str, int]
    mean_glut: float
    mean_gap: float
    mean_delta: float
    contradiction_keys: List[str]
    gap_keys: List[str]
    trajectory_signature: Dict[str, float]
    notes: List[str] = field(default_factory=list)


class InvariantEngine:
    """Computes state-level invariant reports.

    SEMANTIC PRESERVATION: build_report is identical to v13.
    """

    def build_report(
        self, s: ShadowState, recent_losses: Sequence[float]
    ) -> InvariantReport:
        """Build v13-compatible invariant report."""
        hist: Dict[str, int] = {"T": 0, "F": 0, "B": 0, "G": 0}
        gluts, gaps, deltas = [], [], []
        contradiction_keys, gap_keys = [], []
        for k, v in s.beta.items():
            hist[s.nu.get(k, "G")] += 1
            gluts.append(v.glut)
            gaps.append(v.gap)
            deltas.append(v.delta)
            if v.glut >= 0.45:
                contradiction_keys.append(k)
            if v.gap >= 0.45:
                gap_keys.append(k)
        notes: List[str] = []
        if not is_cache_coherent(s):
            notes.append("cache incoherent")
        if contradiction_keys:
            notes.append("high contradiction mass present")
        if gap_keys:
            notes.append("high gap mass present")
        return InvariantReport(
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            coherent=is_cache_coherent(s),
            support_profile={k: asdict(v) for k, v in s.beta.items()},
            cache_histogram=hist,
            mean_glut=float(sum(gluts) / max(1, len(gluts))),
            mean_gap=float(sum(gaps) / max(1, len(gaps))),
            mean_delta=float(sum(deltas) / max(1, len(deltas))),
            contradiction_keys=contradiction_keys[:20],
            gap_keys=gap_keys[:20],
            trajectory_signature={
                "loss_mean_8": float(
                    sum(recent_losses[-8:]) / max(1, len(recent_losses[-8:]))
                )
                if recent_losses
                else 0.0,
                "cache_mass": float(sum(hist.values())),
                "support_variation": float(
                    sum(abs(x) for x in deltas) / max(1, len(deltas))
                )
                if deltas
                else 0.0,
            },
            notes=notes,
        )

    def build_state_report(
        self,
        s: ShadowState,
        recent_losses: Sequence[float] = (),
        profile: ReportProfile | None = None,
    ) -> StateReport:
        """Build v14 enriched state report.

        Includes lane divergence summary and determinized values.
        """
        if profile is None:
            profile = ReportProfile()

        hist: Dict[str, int] = {"T": 0, "F": 0, "B": 0, "G": 0}
        gluts, gaps, deltas_list = [], [], []
        contradiction_keys, gap_keys = [], []

        for k, v in s.beta.items():
            hist[s.nu.get(k, "G")] += 1
            gluts.append(v.glut)
            gaps.append(v.gap)
            deltas_list.append(v.delta)
            if v.glut >= profile.glut_critical_threshold:
                contradiction_keys.append(k)
            if v.gap >= profile.gap_critical_threshold:
                gap_keys.append(k)

        notes: List[str] = []
        if not is_cache_coherent(s):
            notes.append("cache incoherent")
        if contradiction_keys:
            notes.append(f"{len(contradiction_keys)} keys above glut critical threshold")
        if gap_keys:
            notes.append(f"{len(gap_keys)} keys above gap critical threshold")

        # Lane divergence: average spread across all keys
        spreads = []
        for v in s.beta.values():
            ld = lane_divergence(v.t, v.f)
            spreads.append(ld.get("spread", 0.0))

        # Determinized summary: top keys by absolute delta
        sorted_keys = sorted(s.beta.keys(), key=lambda k: abs(s.beta[k].delta), reverse=True)
        det_summary = {k: determinize_value(s.beta[k]) for k in sorted_keys[:20]}

        n = max(1, len(gluts))
        return StateReport(
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            coherent=is_cache_coherent(s),
            cache_histogram=hist,
            beta_key_count=len(s.beta),
            mean_glut=float(sum(gluts) / n),
            mean_gap=float(sum(gaps) / n),
            mean_delta=float(sum(deltas_list) / n),
            contradiction_keys=contradiction_keys[:20],
            gap_keys=gap_keys[:20],
            lane_divergence_summary={
                "mean_spread": float(sum(spreads) / max(1, len(spreads))),
                "max_spread": float(max(spreads)) if spreads else 0.0,
            },
            determinized_summary=det_summary,
            profile_id=profile.profile_id,
            notes=notes,
        )
