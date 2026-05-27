"""
TOVAH v14 core/runtime_interface.py — Runtime semantic interface.

Provides a pure-function-style API for reasoning about state transitions.
Given the same state, same profile, and same operator inputs,
replay must produce the same semantic result.

This module exposes the ability to:
- describe current state
- apply a named operator
- verify deterministic replay
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tovah_v14.core.primitives import BilateralValue, bilateral_or, bilateral_recover
from tovah_v14.core.state import ShadowState, CarrierState, ProvenanceState
from tovah_v14.core.cache import refresh_state, is_cache_coherent, gamma_cache
from tovah_v14.core.determinization import determinize_beta, readout_state


@dataclass
class RuntimeStateView:
    """Immutable snapshot of runtime state for external consumption.

    This is a BRIDGE object. It crosses the layer boundary between
    the bilateral evidence layer and the semantic decision layer.
    """
    beta_snapshot: Dict[str, Dict[str, float]]  # key -> {t, f}
    nu_snapshot: Dict[str, str]
    carrier_snapshot: Dict[str, Any]
    provenance_snapshot: Dict[str, Any]
    coherent: bool
    profile_id: str = "default"


@dataclass
class DeterminizedStateView:
    """Classical view over bilateral state.

    This is an INTERFACE VIEW, not the core state.
    """
    confidence_map: Dict[str, float]  # key -> [0, 1]
    cache_histogram: Dict[str, int]
    profile_id: str = "default"


def capture_runtime_view(s: ShadowState, profile_id: str = "default") -> RuntimeStateView:
    """Capture an immutable view of current runtime state."""
    return RuntimeStateView(
        beta_snapshot={k: {"t": v.t, "f": v.f} for k, v in s.beta.items()},
        nu_snapshot=dict(s.nu),
        carrier_snapshot={
            "active_goal": s.c.active_goal,
            "cycle": s.c.cycle,
            "mode": s.c.mode,
            "paused": s.c.paused,
            "degraded": getattr(s.c, "degraded", False),
        },
        provenance_snapshot={
            "step": s.pi.step,
            "refresh_count": s.pi.refresh_count,
        },
        coherent=is_cache_coherent(s),
        profile_id=profile_id,
    )


def capture_determinized_view(s: ShadowState, profile_id: str = "default") -> DeterminizedStateView:
    """Capture a classical determinized view of current state."""
    hist: Dict[str, int] = {"T": 0, "F": 0, "B": 0, "G": 0}
    for v in s.nu.values():
        hist[v] = hist.get(v, 0) + 1
    return DeterminizedStateView(
        confidence_map=determinize_beta(s.beta),
        cache_histogram=hist,
        profile_id=profile_id,
    )


def make_fresh_state(beta_keys: list[str]) -> ShadowState:
    """Create a fresh ShadowState with the given beta keys at default values.

    Useful for testing and for boot-time state creation.
    """
    beta = {k: BilateralValue(0.5, 0.0) for k in beta_keys}
    s = ShadowState(
        c=CarrierState(),
        beta=beta,
        nu={},
        pi=ProvenanceState(),
    )
    return refresh_state(s)


def verify_replay(
    state_before: RuntimeStateView,
    state_after: RuntimeStateView,
    operator_name: str,
) -> Dict[str, Any]:
    """Verify that a state transition is consistent.

    Checks:
    - provenance advanced
    - cache coherent after
    - no keys disappeared unexpectedly

    Returns a report dict.
    """
    report: Dict[str, Any] = {
        "operator": operator_name,
        "coherent_before": state_before.coherent,
        "coherent_after": state_after.coherent,
        "provenance_advanced": (
            state_after.provenance_snapshot.get("refresh_count", 0)
            > state_before.provenance_snapshot.get("refresh_count", 0)
        ),
        "keys_before": len(state_before.beta_snapshot),
        "keys_after": len(state_after.beta_snapshot),
    }

    lost_keys = set(state_before.beta_snapshot) - set(state_after.beta_snapshot)
    if lost_keys:
        report["lost_keys"] = sorted(lost_keys)
    report["ok"] = report["coherent_after"] and not lost_keys

    return report
