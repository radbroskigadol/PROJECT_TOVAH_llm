"""
TOVAH v14 core/primitives.py — Bilateral evidence type and operators.

SEMANTIC PRESERVATION:
  Every formula here is identical to v13. No behavioral changes.
  Added: __repr__, NaN/inf safety in clamp, __eq__/__hash__ for testing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BilateralValue:
    """Core ShadowHoTT datum: independent truth and falsity channels.

    t: truth support in [0, 1]
    f: falsity support in [0, 1]

    These are INDEPENDENT. High t and high f simultaneously is a glut (contradiction).
    Low t and low f simultaneously is a gap (underdetermination).
    """
    t: float = 0.0
    f: float = 0.0

    def clamp(self) -> BilateralValue:
        """Clamp t and f to [0, 1]. NaN/inf become 0."""
        self.t = float(max(0.0, min(1.0, self.t if math.isfinite(self.t) else 0.0)))
        self.f = float(max(0.0, min(1.0, self.f if math.isfinite(self.f) else 0.0)))
        return self

    @property
    def glut(self) -> float:
        """Contradiction mass: min(t, f). High glut = both true and false."""
        return min(self.t, self.f)

    @property
    def gap(self) -> float:
        """Uncertainty mass: min(1-t, 1-f). High gap = neither true nor false."""
        return min(1.0 - self.t, 1.0 - self.f)

    @property
    def delta(self) -> float:
        """Net support: t - f. Positive = leans true."""
        return self.t - self.f

    def __repr__(self) -> str:
        return f"BV(t={self.t:.3f}, f={self.f:.3f}, Δ={self.delta:+.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BilateralValue):
            return NotImplemented
        return abs(self.t - other.t) < 1e-9 and abs(self.f - other.f) < 1e-9

    def __hash__(self) -> int:
        return hash((round(self.t, 9), round(self.f, 9)))


def _prob_or01(x: float, y: float) -> float:
    """Associative probabilistic OR on clamped [0,1] evidence."""
    x = max(0.0, min(1.0, x if math.isfinite(x) else 0.0))
    y = max(0.0, min(1.0, y if math.isfinite(y) else 0.0))
    if x >= 1.0 or y >= 1.0:
        return 1.0
    if x <= 0.0:
        return y
    if y <= 0.0:
        return x
    return max(0.0, min(1.0, -math.expm1(math.log1p(-x) + math.log1p(-y))))


def bilateral_or(a: BilateralValue, b: BilateralValue) -> BilateralValue:
    """Paraconsistent disjunction: independent channel accumulation.

    Probabilistic OR is applied independently to truth and falsity support.
    Inputs are clamped before accumulation and a stable log-space form is used,
    so repeated evidence accumulation is order-insensitive up to floating-point
    roundoff for all finite inputs.
    """
    return BilateralValue(
        _prob_or01(a.t, b.t),
        _prob_or01(a.f, b.f),
    )


def bilateral_recover(
    v: BilateralValue,
    truth_gain: float = 0.0,
    falsity_decay: float = 0.0,
) -> BilateralValue:
    """Asymmetric belief update: strengthen truth, weaken falsity.

    FORMULA (preserved exactly from v13):
      t_out = v.t + truth_gain - v.t * truth_gain
      f_out = v.f * max(0, 1 - falsity_decay)

    truth_gain uses probabilistic OR semantics (never exceeds 1).
    falsity_decay is multiplicative damping.
    """
    return BilateralValue(
        v.t + truth_gain - v.t * truth_gain,
        v.f * max(0.0, 1.0 - falsity_decay),
    ).clamp()


def coerce_bilateral_value(
    value: Any,
    default_t: float = 0.5,
    default_f: float = 0.1,
) -> BilateralValue:
    """Safely coerce anything into a BilateralValue.

    Handles: BilateralValue, dict with t/f keys, arbitrary objects.
    NaN/inf-safe: any non-finite value becomes the default.

    This is critical for v13 migration: old saved state may contain
    plain dicts, corrupted floats, or unexpected types.
    """
    def _num(x: Any, default: float) -> float:
        if isinstance(x, dict):
            for k in ("scalar", "value", "score", "t", "f"):
                if k in x:
                    try:
                        v = float(x[k])
                        return v if math.isfinite(v) else float(default)
                    except Exception:
                        pass
            return float(default)
        try:
            v = float(x)
            return v if math.isfinite(v) else float(default)
        except Exception:
            return float(default)

    if isinstance(value, BilateralValue):
        return BilateralValue(_num(value.t, default_t), _num(value.f, default_f)).clamp()
    if isinstance(value, dict):
        return BilateralValue(
            _num(value.get("t", default_t), default_t),
            _num(value.get("f", default_f), default_f),
        ).clamp()
    return BilateralValue(default_t, default_f).clamp()
