"""
TOVAH v14 core/lanes.py — Four-lane ShadowHoTT projection system.

Lanes are VIEWS over bilateral evidence, not replacements for the bilateral core.
The stored state is always (t, f). Lanes project that into different interpretive frames.

SEMANTIC PRESERVATION:
  lane_project formulas are identical to v13.
  Added: explicit per-lane functions, lane mixture, divergence metrics.

Lane semantics:
  A: Classical-clean attenuation — favor truth not contradicted, falsity not contradicted.
  B: Glut-as-true bias — contradiction leans toward assertability.
  C: Gap-as-false bias — underdetermination leans toward rejection.
  D: Forced totalization — produce a determinized answer.

CRITICAL RULE:
  Bilateral revision always remains in (t, f) space.
  Lanes are read-only views. Do not collapse stored bilateral state to serve one lane.
"""
from __future__ import annotations

from typing import Dict, Tuple


def lane_project_A(t: float, f: float) -> Tuple[float, float]:
    """Lane A: Classical-clean attenuation.
    Attenuate truth by falsity and vice versa.
    """
    return t * (1.0 - f), f * (1.0 - t)


def lane_project_B(t: float, f: float) -> Tuple[float, float]:
    """Lane B: Glut-as-true / paraconsistent bias.
    Contradiction mass gets counted as assertability.
    """
    return max(t, f), f * (1.0 - t)


def lane_project_C(t: float, f: float) -> Tuple[float, float]:
    """Lane C: Gap-as-false / paracomplete bias.
    Underdetermination leans toward rejection.
    """
    return t * (1.0 - f), max(f, 1.0 - t) * (1.0 - t * f)


def lane_project_D(t: float, f: float) -> Tuple[float, float]:
    """Lane D: Forced totalization / deterministic.
    Produces a classical view where f = 1 - t.
    """
    return t, 1.0 - t


# Dispatch table
_LANE_FNS = {
    "A": lane_project_A,
    "B": lane_project_B,
    "C": lane_project_C,
    "D": lane_project_D,
}


def lane_project(t: float, f: float, lane: str) -> Tuple[float, float]:
    """Project bilateral values through one of four semantic lanes.

    FORMULA (preserved exactly from v13):
      A: (t*(1-f), f*(1-t))
      B: (max(t,f), f*(1-t))
      C: (t*(1-f), max(f,1-t)*(1-t*f))
      D: (t, 1-t)

    Returns (projected_t, projected_f).
    """
    fn = _LANE_FNS.get(lane)
    if fn is not None:
        return fn(t, f)
    return t, f  # identity fallback


def lane_mixture(
    t: float,
    f: float,
    weights: Dict[str, float],
    *,
    include_d: bool = False,
) -> Tuple[float, float]:
    """Weighted mixture of lane projections without accidental D collapse.

    Lane D is a forced-totalization diagnostic: ``lane_project_D(t,f)`` discards
    independent falsity support by construction. The runtime mixture therefore
    excludes D unless ``include_d=True`` is explicitly requested.

    This pure function does NOT modify bilateral state.
    """
    usable = {k: v for k, v in weights.items() if include_d or k != "D"}
    total_w = sum(usable.values())
    if total_w < 1e-8:
        return t, f
    mix_t, mix_f = 0.0, 0.0
    for lane_name, w in usable.items():
        pt, pf = lane_project(t, f, lane_name)
        mix_t += w * pt
        mix_f += w * pf
    return mix_t / total_w, mix_f / total_w


def lane_divergence(t: float, f: float) -> Dict[str, float]:
    """Compute how much the four lanes disagree on this bilateral value.

    Returns per-lane projected delta (t-f) and the max spread.
    High divergence = the lanes disagree significantly about this proposition.

    This is a NEW v14 diagnostic. Pure function, no side effects.
    """
    deltas: Dict[str, float] = {}
    for name in ("A", "B", "C", "D"):
        pt, pf = lane_project(t, f, name)
        deltas[name] = pt - pf
    vals = list(deltas.values())
    deltas["spread"] = max(vals) - min(vals) if vals else 0.0
    return deltas
