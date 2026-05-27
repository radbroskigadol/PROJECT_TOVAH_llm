"""
TOVAH v14 core/cache.py — Gamma cache computation, state refresh, coherence.

SEMANTIC PRESERVATION:
  gamma_cache, refresh_state, is_cache_coherent are identical to v13.
  No behavioral changes.
"""
from __future__ import annotations

from typing import Dict

from tovah_v14.core.primitives import BilateralValue, coerce_bilateral_value
from tovah_v14.core.state import ShadowState
from tovah_v14.config.constants import GAMMA_THETA_T, GAMMA_THETA_F


def gamma_cache(
    beta: Dict[str, BilateralValue],
    theta_t: float = GAMMA_THETA_T,
    theta_f: float = GAMMA_THETA_F,
) -> Dict[str, str]:
    """Compute four-valued classification for each belief key.

    FORMULA (preserved exactly from v13):
      T: t >= theta_t and f < theta_f       (true)
      F: t < theta_t and f >= theta_f       (false)
      B: t >= theta_t and f >= theta_f      (both / glut)
      G: t < theta_t and f < theta_f        (gap / neither)

    theta_t and theta_f default to 0.55.
    """
    out: Dict[str, str] = {}
    for k, raw_v in beta.items():
        v = coerce_bilateral_value(raw_v)
        if v.t >= theta_t and v.f >= theta_f:
            out[k] = "B"
        elif v.t >= theta_t:
            out[k] = "T"
        elif v.f >= theta_f:
            out[k] = "F"
        else:
            out[k] = "G"
    return out


def refresh_state(s: ShadowState) -> ShadowState:
    """Recompute gamma cache from current beta.

    MUST be called after any mutation to s.beta.

    BEHAVIOR (preserved exactly from v13):
      1. Coerce all beta values to valid BilateralValues.
      2. Recompute nu = gamma_cache(beta).
      3. Increment pi.refresh_count.
    """
    if not isinstance(s.beta, dict):
        s.beta = {}
    s.beta = {str(k): coerce_bilateral_value(v) for k, v in s.beta.items()}
    s.nu = gamma_cache(s.beta)
    s.pi.refresh_count += 1
    return s


def is_cache_coherent(s: ShadowState) -> bool:
    """Check that nu matches a fresh recomputation from beta.

    BEHAVIOR (preserved exactly from v13):
      Returns True iff s.nu == gamma_cache(s.beta).
    """
    return s.nu == gamma_cache(s.beta)
