"""
TOVAH v14 debug/observability.py — Per-cycle observability metrics.

CycleMetrics captures a full structured snapshot of kernel health
at each cycle. Used for dashboarding, anomaly detection, and
audit trails.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState
from tovah_v14.core.cache import is_cache_coherent


@dataclass
class CycleMetrics:
    """Structured observability snapshot for one kernel cycle."""
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)
    coherent: bool = True
    mean_glut: float = 0.0
    mean_gap: float = 0.0
    mean_delta: float = 0.0
    beta_key_count: int = 0
    cache_histogram: Dict[str, int] = field(default_factory=dict)
    active_goal: str = ""
    training_loss: float = 0.0
    training_phase: str = ""
    tools_used: int = 0
    patches_staged: int = 0
    patches_applied: int = 0
    memory_counts: Dict[str, int] = field(default_factory=dict)
    tasks_active: int = 0
    budget_usage: Dict[str, float] = field(default_factory=dict)
    degraded: bool = False
    notes: List[str] = field(default_factory=list)


def collect_cycle_metrics(
    state: ShadowState,
    *,
    training_loss: float = 0.0,
    training_phase: str = "",
    tools_used: int = 0,
    patches_staged: int = 0,
    patches_applied: int = 0,
    memory_counts: Dict[str, int] | None = None,
    tasks_active: int = 0,
    budget_usage: Dict[str, float] | None = None,
) -> CycleMetrics:
    """Collect metrics from current state."""
    n = max(1, len(state.beta))
    hist: Dict[str, int] = {"T": 0, "F": 0, "B": 0, "G": 0}
    for v in state.nu.values():
        hist[v] = hist.get(v, 0) + 1

    return CycleMetrics(
        cycle=state.c.cycle,
        coherent=is_cache_coherent(state),
        mean_glut=sum(v.glut for v in state.beta.values()) / n if state.beta else 0.0,
        mean_gap=sum(v.gap for v in state.beta.values()) / n if state.beta else 0.0,
        mean_delta=sum(v.delta for v in state.beta.values()) / n if state.beta else 0.0,
        beta_key_count=len(state.beta),
        cache_histogram=hist,
        active_goal=state.c.active_goal,
        training_loss=training_loss,
        training_phase=training_phase,
        tools_used=tools_used,
        patches_staged=patches_staged,
        patches_applied=patches_applied,
        memory_counts=memory_counts or {},
        tasks_active=tasks_active,
        budget_usage=budget_usage or {},
        degraded=getattr(state.c, "degraded", False),
    )
