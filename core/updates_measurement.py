"""
TOVAH v14 core/updates_measurement.py — Measurement-like update semantics.

Measurement-like transitions involve RESET, NORMALIZATION, or CONTROLLED COLLAPSE:
- set a belief to a definite value based on observed outcome
- force a classical determination for an interface that needs one
- bounded reset after a definitive event

These are NOT structure-preserving. They impose a new state rather than
accumulating into the old one. The distinction from gate-like updates
matters for correctness and auditability.

NEW in v14: explicit distinction from gate-like updates.
"""
from __future__ import annotations

from tovah_v14.core.primitives import BilateralValue


def measurement_set(t: float, f: float) -> BilateralValue:
    """Direct measurement: set bilateral value from observed outcome.

    This is a measurement-like operation: the previous value is replaced,
    not accumulated into.

    Use for: definitive test results, regression outcomes, capability checks.
    """
    return BilateralValue(t, f).clamp()


def measurement_reset(
    current: BilateralValue,
    reset_toward_t: float = 0.5,
    reset_toward_f: float = 0.1,
    strength: float = 1.0,
) -> BilateralValue:
    """Partial reset toward a target value.

    strength=1.0 fully resets. strength=0.5 moves halfway.
    This is measurement-like because it can override accumulated evidence.

    Use for: stale belief cleanup, periodic recalibration.
    """
    s = max(0.0, min(1.0, strength))
    new_t = current.t + s * (reset_toward_t - current.t)
    new_f = current.f + s * (reset_toward_f - current.f)
    return BilateralValue(new_t, new_f).clamp()


def measurement_determinize(v: BilateralValue) -> float:
    """Produce a single classical float from bilateral value.

    This is the canonical determinization: delta clamped to [-1, 1],
    then mapped to [0, 1].

    CRITICAL: This is a VIEW operation. It does NOT modify the stored
    bilateral state. The returned float is for interfaces that require
    a scalar. The bilateral (t, f) remains authoritative.
    """
    return max(0.0, min(1.0, 0.5 + 0.5 * v.delta))


def measurement_confidence(v: BilateralValue) -> float:
    """Produce a confidence score from bilateral value.

    High confidence = strong delta and low glut.
    This is a VIEW operation, not a state mutation.
    """
    return max(0.0, min(1.0, abs(v.delta) * (1.0 - v.glut)))
