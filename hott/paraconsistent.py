"""
TOVAH v14.2.0 hott/paraconsistent.py — Bilateral semantics for HoTT assertions.

This module is the bridge between the structural backbone (hott.core)
and the bilateral runtime (BilateralValue / Belnap four-valued logic).

The key conceptual move:
  In ordinary HoTT, "Id(A;a,b) is inhabited" is a Boolean assertion.
  In paraconsistent HoTT, that assertion is *bilateral*:
    - T-evidence: paths supporting a = b
    - F-evidence: paths supporting a ≠ b
    - both high → genuine paradox (K-class)
    - both low → no information yet (G-class)

This module provides:
  PIdJudgment       — a bilateral assertion about an identity-type's habitation
  bilateral_J       — J-induction that respects paraconsistent semantics
  bilateral_transport — transport that gates on classified evidence
  classify_path     — A/B/K/G classification of a path's bilateral evidence
  is_T_supported    — does the T-evidence dominate?

Public laws (preserved by tests):
  - classify_path(refl_a) == "A" (high T, low F)
  - bilateral_transport(P, refl_a, x).value == x
  - composing supportive paths yields supportive composite (under min/max)
  - composing a supportive path with a refuted path yields a K/B
    composite (paradox or refutation), NEVER silently 'wins for one side'
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott.core import (
    Type, Id, Path, refl, transport, J,
    compose, inverse, DependentFamily, TransportResult,
)


# --- Belnap classification of a Path's bilateral evidence -------------------

class IdentityClass(str, Enum):
    """ABKG class for a path's identity-evidence."""
    A = "A"  # Agreed: high T, low F. Supports identification.
    B = "B"  # Belied: low T, high F. Refutes identification.
    K = "K"  # Knot: high T AND high F. Contested / paradox.
    G = "G"  # Gap: low T, low F. No information.

    @classmethod
    def of(cls, t: float, f: float,
           theta_t: float = 0.55, theta_f: float = 0.55) -> "IdentityClass":
        t_high = t >= theta_t
        f_high = f >= theta_f
        if t_high and f_high:
            return cls.K
        if t_high and not f_high:
            return cls.A
        if not t_high and f_high:
            return cls.B
        return cls.G


def classify_path(p: Path,
                  theta_t: float = 0.55, theta_f: float = 0.55) -> IdentityClass:
    """Belnap class of a path's identity-evidence."""
    return IdentityClass.of(p.bilateral.t, p.bilateral.f, theta_t, theta_f)


def is_T_supported(p: Path, theta: float = 0.55) -> bool:
    """T-supported = A-class. Use to gate classical-lane operations."""
    return classify_path(p, theta_t=theta, theta_f=theta) == IdentityClass.A


# --- Paraconsistent J-judgment ---------------------------------------------

@dataclass
class PIdJudgment:
    """A bilateral judgment about whether Id(A;a,b) is inhabited.

    Attributes:
      id_type:      the Id-formation we're judging
      supporting:   paths we have FOR the identification
      refuting:     paths we have AGAINST the identification
      class_:       Belnap class summarising the overall judgment
      reason:       short explanation
    """
    id_type: Id
    supporting: List[Path] = field(default_factory=list)
    refuting: List[Path] = field(default_factory=list)
    class_: IdentityClass = IdentityClass.G
    reason: str = "no evidence"

    @property
    def best_t(self) -> float:
        return max((p.bilateral.t for p in self.supporting), default=0.0)

    @property
    def best_f(self) -> float:
        return max((p.bilateral.f for p in self.refuting), default=0.0)

    @property
    def bilateral(self) -> BilateralValue:
        return BilateralValue(self.best_t, self.best_f)

    def recompute_class(self,
                        theta_t: float = 0.55, theta_f: float = 0.55) -> None:
        self.class_ = IdentityClass.of(self.best_t, self.best_f, theta_t, theta_f)


def judge_identity(id_type: Id, supporting: List[Path], refuting: List[Path]
                   ) -> PIdJudgment:
    """Build a bilateral judgment from supporting / refuting paths.

    PARACONSISTENT INVARIANT: we do NOT collapse contradiction. If both
    pools are non-empty with high evidence, the result is K-class — a
    real paradox that survives transport and that downstream consumers
    can detect.
    """
    j = PIdJudgment(id_type=id_type,
                    supporting=list(supporting),
                    refuting=list(refuting))
    j.recompute_class()
    if j.class_ == IdentityClass.K:
        j.reason = f"contested (supported: {len(supporting)}, refuted: {len(refuting)})"
    elif j.class_ == IdentityClass.A:
        j.reason = f"supported (best_t={j.best_t:.2f})"
    elif j.class_ == IdentityClass.B:
        j.reason = f"refuted (best_f={j.best_f:.2f})"
    else:
        j.reason = "no decisive evidence"
    return j


# --- Bilateral J-induction --------------------------------------------------

def bilateral_J(C: Callable[[Any, Any, Path], Type],
                d: Callable[[Any], Any],
                judgment: PIdJudgment,
                *,
                gate_on: IdentityClass = IdentityClass.A
                ) -> Tuple[Optional[Any], PIdJudgment]:
    """J-induction that respects paraconsistent semantics.

    Standard J: given motive C and base d, eliminates over a path p.
    Bilateral J: given motive C and base d, eliminates over a *judgment*
    (a collection of supporting/refuting paths) and gates on Belnap class.

    Semantics:
      - judgment.class_ == A and gate_on >= A: eliminate using best supporting path
      - judgment.class_ == K: REFUSE to eliminate (return None) — return
        the judgment so caller can route to contradiction-handling
      - judgment.class_ == B and gate_on requires support: refuse
      - judgment.class_ == G: refuse, return judgment

    Returns (result, refreshed_judgment). result is None when elimination
    is gated.

    This is the operational heart of "identity-preserving transformation
    under paraconsistency": we only run the inductive principle when the
    bilateral evidence supports it, and we surface the paradox or gap
    instead of silently picking a side.
    """
    j = judgment
    j.recompute_class()

    # Gate logic.
    if j.class_ == IdentityClass.K:
        # Genuine paradox: refuse to eliminate, surface the contradiction.
        return None, j
    if j.class_ == IdentityClass.B:
        # Refutation: refuse to eliminate; the alleged identity is denied.
        return None, j
    if j.class_ == IdentityClass.G and gate_on == IdentityClass.A:
        # Gap: no information.
        return None, j

    # We have supporting evidence. Eliminate using the strongest path.
    if not j.supporting:
        return None, j
    best = max(j.supporting, key=lambda p: p.bilateral.t)
    result = J(C, d, best)
    return result, j


# --- Bilateral transport ----------------------------------------------------

def bilateral_transport(P: DependentFamily,
                        judgment: PIdJudgment,
                        x: Any,
                        *,
                        gate_on: IdentityClass = IdentityClass.A,
                        coerce: Optional[Callable[[Any, Type, Type], Any]] = None
                        ) -> Tuple[Optional[TransportResult], PIdJudgment]:
    """Transport that gates on the bilateral judgment's classification.

    Returns (TransportResult or None, refreshed judgment).
    - If the judgment is A-class (supported), transport along the
      strongest supporting path and return the result.
    - If K (paradox), B (refuted), or G (gap) and gate_on=A: refuse and
      return None — the value is not safely transportable.

    PARACONSISTENT GUARANTEE: contradiction is never silently collapsed.
    If both supporting and refuting evidence are strong, the consumer
    learns about the paradox by getting (None, K-class-judgment).
    """
    j = judgment
    j.recompute_class()

    if j.class_ == IdentityClass.K:
        return None, j
    if j.class_ == IdentityClass.B:
        return None, j
    if j.class_ == IdentityClass.G and gate_on == IdentityClass.A:
        return None, j

    if not j.supporting:
        return None, j
    best = max(j.supporting, key=lambda p: p.bilateral.t)
    result = transport(P, best, x, coerce=coerce)
    return result, j


# --- Aggregation: combining multiple judgments ------------------------------

def combine_judgments(judgments: List[PIdJudgment]) -> PIdJudgment:
    """Combine multiple identity-judgments about the same id_type.

    PARACONSISTENT: supporting and refuting pools are unioned. A high-T
    judgment and a high-F judgment together produce K-class, not 'one
    wins'.
    """
    if not judgments:
        raise ValueError("combine_judgments needs at least one judgment")
    first = judgments[0]
    if any(j.id_type != first.id_type for j in judgments):
        raise ValueError("combine_judgments: id_type mismatch across inputs")
    supp = []
    refu = []
    for j in judgments:
        supp.extend(j.supporting)
        refu.extend(j.refuting)
    return judge_identity(first.id_type, supp, refu)


# --- Path-coherence checks --------------------------------------------------

def check_refl_J_reduction(C: Callable[[Any, Any, Path], Type],
                           d: Callable[[Any], Any],
                           a: Any, A: Type) -> bool:
    """Verify the reflexivity computation rule for J:
       J(C, d, refl_a) reduces to d(a).

    This is the *defining* law of J-induction. We test it operationally
    by computing both sides and comparing.
    """
    r = refl(A, a)
    left = J(C, d, r)
    right = d(a)
    # J's refl-case returns d(a) directly per our implementation. We
    # check via callable signature for the dataclass / scalar cases.
    return left == right or repr(left) == repr(right)


def check_transport_along_refl(P: DependentFamily, a: Any, x: Any, A: Type
                               ) -> bool:
    """Verify: transport(P, refl_a, x).value == x with bilateral (1, 0)."""
    r = refl(A, a)
    res = transport(P, r, x)
    return (res.value == x
            and res.bilateral.t == 1.0
            and res.bilateral.f == 0.0)


def check_compose_associativity(p: Path, q: Path, r: Path) -> bool:
    """Verify: compose(compose(p, q), r) ~ compose(p, compose(q, r))
    on endpoints and on bilateral evidence (up to min/max combination).

    In real HoTT this is associativity-up-to-2-path. We only check the
    *truncated* statement (endpoints + bilateral agree), which is what
    we need for the verifier layer to function correctly.
    """
    try:
        lhs = compose(compose(p, q), r)
        rhs = compose(p, compose(q, r))
    except Exception:
        return False
    return (lhs.source == rhs.source
            and lhs.target == rhs.target
            and abs(lhs.bilateral.t - rhs.bilateral.t) < 1e-9
            and abs(lhs.bilateral.f - rhs.bilateral.f) < 1e-9)
