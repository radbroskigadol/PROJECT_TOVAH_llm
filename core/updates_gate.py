"""
TOVAH v14 core/updates_gate.py — Gate-like update semantics.

Gate-like transitions are STRUCTURE-PRESERVING:
- support transport
- exact propagation
- no normalization or collapse

These are used when the update should faithfully propagate evidence
without resetting or forcing a classical interpretation.

NEW in v14: explicit distinction from measurement-like updates.
The v13 code used bilateral_or and bilateral_recover everywhere without
distinguishing gate-like from measurement-like intent. This module makes
the distinction explicit for audit and correctness.
"""
from __future__ import annotations

from tovah_v14.core.primitives import BilateralValue, bilateral_or, bilateral_recover


def gate_accumulate(current: BilateralValue, evidence: BilateralValue) -> BilateralValue:
    """Accumulate new evidence into existing belief.

    This is a gate-like operation: evidence is added via bilateral_or
    without any normalization or collapse. The bilateral structure is preserved.

    Use for: tool results, research findings, advisor responses —
    anywhere you are ADDING information.
    """
    return bilateral_or(current, evidence)


def gate_recover(
    current: BilateralValue,
    truth_gain: float = 0.0,
    falsity_decay: float = 0.0,
) -> BilateralValue:
    """Asymmetric recovery update.

    This is a gate-like operation: truth is strengthened, falsity is damped,
    but the bilateral structure is preserved. No collapse to classical.

    Use for: successful operations, confidence building, gradual healing.
    """
    return bilateral_recover(current, truth_gain=truth_gain, falsity_decay=falsity_decay)


def gate_weaken(
    current: BilateralValue,
    truth_decay: float = 0.0,
    falsity_gain: float = 0.0,
) -> BilateralValue:
    """Symmetric counterpart to gate_recover: weaken truth, strengthen falsity.

    Use for: failures, errors, degradation signals.
    """
    new_t = current.t * max(0.0, 1.0 - truth_decay)
    new_f = current.f + falsity_gain - current.f * falsity_gain
    return BilateralValue(new_t, new_f).clamp()
