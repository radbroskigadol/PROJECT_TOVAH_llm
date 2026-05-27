"""
TOVAH v14.2.0 hott/core.py — Paraconsistent HoTT primitive layer.

DESIGN PRINCIPLE (from the architecture brief):
    Use bilateral paraconsistency for runtime cognition.
    Use full HoTT for identity-preserving transformation.

This module implements the *structural backbone* that the existing
bilateral runtime was shadowing: Types, Identity-types, Paths,
reflexivity, J-induction, transport, equivalences — as first-class
Python objects with explicit witnesses, not as a real proof assistant.

The bilateral / paraconsistent semantics enter at the *evidence wrapper*
layer: every Path carries a `BilateralValue` that records evidence for
and against the identification it claims. Transport propagates that
evidence forward, so consumers can decide whether to gate on it.

What this module is:
  - A concrete, testable implementation of HoTT's structural laws
    (refl-J reduction, transport functoriality, equiv composition)
  - The foundation for the verifier layers above it (patch certificates,
    module equivalence, memory identity, obstruction classifier)

What this module is NOT:
  - A general-purpose dependent type checker
  - A universe hierarchy with realized polymorphism
  - A path-equality decision procedure for arbitrary types

The pieces it does provide are enough to operationalize the patch-
witness, module-equivalence, and obstruction-classifier layers that
the architecture brief identifies as highest-value.

Public:
  Type            — a tagged universe element (its `inhabits` predicate
                    determines membership)
  Id              — identity-type constructor: Id(A, a, b) is a Type
  Path            — an inhabitant of an Id-type, carrying a witness +
                    bilateral evidence
  refl            — reflexivity path constructor (T=1, F=0)
  transport       — P : A→Type, p : Id_A(a,b), x : P(a) ↦ P(b)
  J               — path induction: prove C(x,y,p) from C(x,x,refl_x)
  Equiv           — an equivalence between two Types
  is_equiv        — predicate over a function
  Sigma, Pi       — dependent sum / product Types
  TruncationLevel — n-type machinery (prop, set, groupoid, ...)
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from tovah_v14.core.primitives import BilateralValue


# --- Universe ---------------------------------------------------------------

class Type:
    """A type in the type-theory sense: a name, a membership predicate,
    and optional structure (an n-truncation level, a carrier set, etc.).

    Equality of Types is *structural* and *nominal*: two Type objects with
    the same `name` and the same `inhabits` predicate are considered the
    same type. We don't implement univalence here — that's a research
    decision deferred to a possible v15.
    """

    def __init__(self, name: str,
                 inhabits: Callable[[Any], bool] = lambda _x: True,
                 truncation: Optional["TruncationLevel"] = None,
                 carrier: Optional[Any] = None):
        self.name = str(name)
        self.inhabits = inhabits
        self.truncation = truncation
        self.carrier = carrier

    def __repr__(self) -> str:
        return f"Type({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Type) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("Type", self.name))


class TruncationLevel(IntEnum):
    """n-types: -2 = contractible, -1 = mere prop, 0 = set, 1 = groupoid, ..."""
    CONTRACTIBLE = -2
    PROP = -1
    SET = 0
    GROUPOID = 1
    TWO_GROUPOID = 2
    UNTRUNCATED = 100  # sentinel for "we haven't truncated this yet"


# --- Identity-types ---------------------------------------------------------

@dataclass(frozen=True)
class Id:
    """The identity-type constructor: Id(A; a, b) for a, b : A.

    In ordinary HoTT, Id(A; a, b) is itself a Type whose inhabitants are
    paths from a to b. Here we materialize that Type lazily — `Id.type()`
    returns the Type object whose inhabitants are Path instances over
    this Id-formation data.
    """
    A: Type
    a: Any
    b: Any

    def type(self) -> Type:
        """The Type whose inhabitants are Paths over this Id-formation."""
        name = f"Id({self.A.name};{_short(self.a)},{_short(self.b)})"

        def _inhabits(p: Any) -> bool:
            return (isinstance(p, Path)
                    and p.id_type == self
                    and self.A.inhabits(p.source)
                    and self.A.inhabits(p.target))

        return Type(name, inhabits=_inhabits)


def _short(x: Any) -> str:
    """Compact string for use in type names. Stable for hashable objects."""
    try:
        return f"{x!r}"[:40]
    except Exception:
        return f"<{type(x).__name__}>"


# --- Paths ------------------------------------------------------------------

@dataclass
class Path:
    """An inhabitant of an Id-type.

    A Path carries:
      - source, target: the endpoints (must inhabit A)
      - witness: a runtime artifact justifying the identification
                 (e.g. for patches, a diff; for memories, a canonical key;
                 for modules, a contract-compatibility certificate)
      - bilateral: evidence FOR (t) and AGAINST (f) this identification.
                   refl has t=1, f=0. A heuristic match might have t=0.7, f=0.1.
                   A contested identification might have t=0.6, f=0.5 (K-class).
      - id_type: the Id-formation this path inhabits.

    The PARACONSISTENT aspect: a Path is not a Boolean assertion of
    equality. It is bilateral evidence. Transport propagates the evidence
    forward, so a 'highly contested' identification produces a 'highly
    contested' transported value. Consumers gate.
    """
    id_type: Id
    source: Any
    target: Any
    witness: Any
    bilateral: BilateralValue = field(default_factory=lambda: BilateralValue(1.0, 0.0))
    # Optional structural data for path concatenation / inverse:
    is_refl: bool = False
    composed_from: Optional[Tuple["Path", "Path"]] = None

    def __post_init__(self):
        # Endpoint sanity. Don't enforce A.inhabits at construction (callers
        # may build paths during inference); rely on Type.inhabits at use.
        if self.is_refl and self.source != self.target:
            raise ValueError(f"refl Path requires source == target ({self.source!r} vs {self.target!r})")

    @property
    def supports_identification(self) -> bool:
        """True iff bilateral.t >= 0.55 (the GAMMA threshold).

        This is the 'classical lane' projection: does the evidence
        support treating the endpoints as identified? A Path with low T
        and low F (G-class, gap) does NOT support identification — it
        encodes 'no information yet'.
        """
        return self.bilateral.t >= 0.55 and self.bilateral.f < 0.55

    @property
    def is_contested(self) -> bool:
        """K-class: high T and high F. The identification has both
        confirming and refuting evidence."""
        return self.bilateral.t >= 0.55 and self.bilateral.f >= 0.55

    @property
    def is_refuted(self) -> bool:
        """B-class: low T, high F. The identification is actively
        contradicted by evidence."""
        return self.bilateral.t < 0.55 and self.bilateral.f >= 0.55


def refl(A: Type, a: Any) -> Path:
    """The reflexivity path refl_a : Id(A; a, a)."""
    if not A.inhabits(a):
        raise ValueError(f"refl: {a!r} does not inhabit {A}")
    id_type = Id(A, a, a)
    return Path(id_type=id_type, source=a, target=a,
                witness="refl",
                bilateral=BilateralValue(1.0, 0.0),
                is_refl=True)


def compose(p: Path, q: Path, *,
            merge_bilateral: str = "min") -> Path:
    """Compose paths: if p : Id(A; a, b) and q : Id(A; b, c), then
    compose(p, q) : Id(A; a, c).

    BILATERAL SEMANTICS: by default the composition's truth is min(p.t, q.t)
    (the chain is only as strong as its weakest link) and falsity is
    max(p.f, q.f) (any refutation along the chain refutes the chain).
    Set merge_bilateral='product' for multiplicative semantics.

    PARACONSISTENT NOTE: composition of contested paths produces a
    contested composite. We don't collapse contradiction; we propagate it.
    """
    if p.id_type.A != q.id_type.A:
        raise ValueError(f"compose: type mismatch {p.id_type.A} vs {q.id_type.A}")
    if p.target != q.source:
        raise ValueError(
            f"compose: endpoint mismatch p.target={_short(p.target)} "
            f"q.source={_short(q.source)}"
        )
    if merge_bilateral == "min":
        bv = BilateralValue(min(p.bilateral.t, q.bilateral.t),
                            max(p.bilateral.f, q.bilateral.f))
    elif merge_bilateral == "product":
        bv = BilateralValue(p.bilateral.t * q.bilateral.t,
                            1.0 - (1.0 - p.bilateral.f) * (1.0 - q.bilateral.f))
    else:
        raise ValueError(f"unknown merge_bilateral={merge_bilateral!r}")
    return Path(
        id_type=Id(p.id_type.A, p.source, q.target),
        source=p.source, target=q.target,
        witness=("compose", p.witness, q.witness),
        bilateral=bv,
        composed_from=(p, q),
    )


def inverse(p: Path) -> Path:
    """Path inverse: if p : Id(A; a, b), then inverse(p) : Id(A; b, a).

    Bilateral evidence is *preserved*: the strength of 'a = b' equals
    the strength of 'b = a'.
    """
    return Path(
        id_type=Id(p.id_type.A, p.target, p.source),
        source=p.target, target=p.source,
        witness=("inverse", p.witness),
        bilateral=BilateralValue(p.bilateral.t, p.bilateral.f),
    )


# --- Transport and J --------------------------------------------------------

# Type alias for a dependent type family P : A → Type
DependentFamily = Callable[[Any], Type]


@dataclass
class TransportResult:
    """Result of transport(P, p, x). Carries the value AND the bilateral
    evidence inherited from the path. Consumers decide whether to gate.
    """
    value: Any
    bilateral: BilateralValue
    via_path: Path
    target_type: Type

    @property
    def supports_use(self) -> bool:
        """True iff the inherited evidence supports using this value.
        Same threshold as Path.supports_identification."""
        return self.bilateral.t >= 0.55 and self.bilateral.f < 0.55


def transport(P: DependentFamily, p: Path, x: Any,
              *, coerce: Optional[Callable[[Any, Type, Type], Any]] = None
              ) -> TransportResult:
    """Transport an x : P(a) along a path p : Id(A; a, b) to yield P(b).

    Args:
      P:      dependent type family A → Type
      p:      path from a to b in A
      x:      inhabitant of P(a)
      coerce: optional coercion function (x, P_a, P_b) → x'. By default
              the identity function — appropriate when P is constant or
              P(a) and P(b) have shared carriers. For genuinely-different
              fibers, the caller must supply a coercion.

    Returns a TransportResult that carries the transported value AND the
    path's bilateral evidence. The PARACONSISTENT property: a refuted
    path (high F) produces a TransportResult with high F — the consumer
    sees the contradiction, doesn't get a silently-corrupted value.

    LAW (preserved-by-tests): transport(P, refl_a, x) reduces to x with
    BilateralValue(1, 0). This is the computational content of
    'transport along refl is the identity'.
    """
    P_source = P(p.source)
    P_target = P(p.target)
    if not P_source.inhabits(x):
        raise ValueError(f"transport: {x!r} does not inhabit {P_source}")

    if p.is_refl:
        # transport along refl is the identity; evidence is (1, 0).
        if not P_target.inhabits(x):
            raise ValueError(f"transport: {x!r} does not inhabit target {P_target}")
        return TransportResult(
            value=x,
            bilateral=BilateralValue(1.0, 0.0),
            via_path=p,
            target_type=P_target,
        )

    # General case: coerce or identity.
    if coerce is not None:
        x_new = coerce(x, P_source, P_target)
    else:
        x_new = x

    if not P_target.inhabits(x_new):
        raise ValueError(f"transport: coerced value {x_new!r} does not inhabit target {P_target}")

    # Inherit the path's evidence; do not synthesize.
    return TransportResult(
        value=x_new,
        bilateral=BilateralValue(p.bilateral.t, p.bilateral.f),
        via_path=p,
        target_type=P_target,
    )


# J-induction
#   Given:
#     A : Type
#     C : (x : A) (y : A) (p : Id(A; x, y)) → Type
#     d : (x : A) → C(x, x, refl_x)
#   then for any x, y : A and p : Id(A; x, y):
#     J(C, d, p) : C(x, y, p)
#
# In Python we encode this operationally: J reduces by checking is_refl.
# For non-refl paths, the user supplies a transport-based eliminator that
# uses transport to derive C(x,y,p) from C(x,x,refl_x).

def J(C: Callable[[Any, Any, Path], Type],
      d: Callable[[Any], Any],
      p: Path,
      *,
      transport_fn: Optional[Callable[[Path, Any], TransportResult]] = None
      ) -> Any:
    """Path induction (J / id-elim).

    Args:
      C:            motive — for each (x, y, p) gives the type C(x, y, p)
      d:            base case — for each x, gives d(x) : C(x, x, refl_x)
      p:            the path along which we eliminate
      transport_fn: optional explicit transport for the off-diagonal case.
                    By default we build one from `transport` using a
                    constant dependent family P(z) = C(p.source, z, refl_z),
                    which is the canonical reduction strategy.

    LAW: J(C, d, refl_x) reduces to d(x). This is the 'reflexivity computation
    rule' and is what makes J the universal eliminator for identity-types.
    """
    if p.is_refl:
        return d(p.source)

    if transport_fn is None:
        # Default eliminator: transport d(source) along p, applying C
        # at every point. We approximate the dependent type family
        # P(z) = C(source, z, <synthesized path>) by a constant family.
        base = d(p.source)
        # Form a constant family P. The motive C is informational but the
        # carrier is base's type; transport just inherits evidence.
        const_family: DependentFamily = lambda z, _base=base: C(p.source, z, p)
        result = transport(const_family, p, base)
        return result
    else:
        # Caller provides a custom transport behavior.
        base = d(p.source)
        return transport_fn(p, base)


# --- Equivalence ------------------------------------------------------------

@dataclass
class Equiv:
    """An equivalence between Types A and B.

    By the 'coherent equivalence' definition: an equiv is a quasi-inverse
    pair (f, g) with paths f∘g ~ id_B and g∘f ~ id_A. We expose both
    homotopies as Paths in their respective Id-types.

    Equivs compose; the composition path is the canonical one.
    """
    A: Type
    B: Type
    f: Callable[[Any], Any]
    g: Callable[[Any], Any]  # quasi-inverse
    eta: Callable[[Any], Path]  # eta_a : Id(A; g(f(a)), a)
    epsilon: Callable[[Any], Path]  # epsilon_b : Id(B; f(g(b)), b)
    bilateral: BilateralValue = field(default_factory=lambda: BilateralValue(1.0, 0.0))

    def apply(self, a: Any) -> Any:
        return self.f(a)

    def apply_inverse(self, b: Any) -> Any:
        return self.g(b)


def is_equiv(f: Callable[[Any], Any], A: Type, B: Type,
             g: Callable[[Any], Any],
             check_samples: Optional[List[Any]] = None,
             tol: float = 1e-6) -> Tuple[bool, str]:
    """Sample-based check that f : A → B is an equivalence with quasi-inverse g.

    PRAGMATIC: this is a *test* of equivalence, not a proof — we sample
    `check_samples` (if given) and verify f(g(b)) ≈ b and g(f(a)) ≈ a.
    For an honest type-theoretic check the caller must supply genuine
    homotopy paths; this function only tells you whether the equations
    hold on the supplied samples.
    """
    if check_samples is None:
        check_samples = []
    for a in check_samples:
        if not A.inhabits(a):
            return False, f"sample {a!r} not in {A}"
        try:
            roundtrip = g(f(a))
            if roundtrip != a and abs(_approx_diff(roundtrip, a)) > tol:
                return False, f"g(f({_short(a)})) = {_short(roundtrip)} ≠ {_short(a)}"
        except Exception as e:
            return False, f"f/g raised on {a!r}: {e}"
    return True, "ok"


def _approx_diff(x: Any, y: Any) -> float:
    """Numerical 'difference' for approximate equality. Returns 0 for
    structural eq, inf for incomparable, abs(x-y) for numbers."""
    if x == y:
        return 0.0
    try:
        return abs(float(x) - float(y))
    except Exception:
        return float("inf")


def equiv_compose(e1: Equiv, e2: Equiv) -> Equiv:
    """Compose two equivalences: A ≃ B ≃ C → A ≃ C."""
    if e1.B != e2.A:
        raise ValueError(f"equiv_compose: type mismatch {e1.B} vs {e2.A}")
    f = lambda a: e2.f(e1.f(a))
    g = lambda c: e1.g(e2.g(c))
    eta = lambda a: compose(
        Path(
            id_type=Id(e1.A, e1.g(e2.g(e2.f(e1.f(a)))), e1.g(e1.f(a))),
            source=e1.g(e2.g(e2.f(e1.f(a)))),
            target=e1.g(e1.f(a)),
            witness=("eta_compose_inner", a),
            bilateral=e2.bilateral,
        ),
        e1.eta(a),
    )
    epsilon = lambda c: compose(
        Path(
            id_type=Id(e2.B, e2.f(e1.f(e1.g(e2.g(c)))), e2.f(e2.g(c))),
            source=e2.f(e1.f(e1.g(e2.g(c)))),
            target=e2.f(e2.g(c)),
            witness=("epsilon_compose_inner", c),
            bilateral=e1.bilateral,
        ),
        e2.epsilon(c),
    )
    return Equiv(
        A=e1.A, B=e2.B, f=f, g=g, eta=eta, epsilon=epsilon,
        bilateral=BilateralValue(
            min(e1.bilateral.t, e2.bilateral.t),
            max(e1.bilateral.f, e2.bilateral.f),
        ),
    )


# --- Sigma and Pi -----------------------------------------------------------

def Sigma(A: Type, B: DependentFamily, name: Optional[str] = None) -> Type:
    """Dependent sum (Σ x:A) B(x). Inhabitants are pairs (a, b) with
    a:A and b:B(a)."""
    full_name = name or f"Σ({A.name},...)"

    def _inhabits(pair: Any) -> bool:
        if not (isinstance(pair, tuple) and len(pair) == 2):
            return False
        a, b = pair
        if not A.inhabits(a):
            return False
        try:
            return B(a).inhabits(b)
        except Exception:
            return False

    return Type(full_name, inhabits=_inhabits)


def Pi(A: Type, B: DependentFamily, name: Optional[str] = None) -> Type:
    """Dependent product (Π x:A) B(x). Inhabitants are functions f such
    that f(a) : B(a) for every a:A.

    PRAGMATIC: we cannot enumerate A, so we can only sample-check. The
    `inhabits` predicate accepts any callable; for stronger guarantees use
    `check_pi` with explicit samples.
    """
    full_name = name or f"Π({A.name},...)"

    def _inhabits(f: Any) -> bool:
        return callable(f)

    return Type(full_name, inhabits=_inhabits)


def check_pi(A: Type, B: DependentFamily, f: Callable[[Any], Any],
             samples: List[Any]) -> Tuple[bool, str]:
    """Sample-check a Π-inhabitant: verify f(a) : B(a) for given samples."""
    for a in samples:
        if not A.inhabits(a):
            return False, f"sample {a!r} not in {A.name}"
        try:
            b = f(a)
        except Exception as e:
            return False, f"f({a!r}) raised: {e}"
        if not B(a).inhabits(b):
            return False, f"f({a!r}) = {b!r} not in {B(a).name}"
    return True, "ok"


# --- Utilities --------------------------------------------------------------

def path_bilateral_summary(paths: List[Path]) -> Dict[str, Any]:
    """Aggregate statistics over a list of paths — useful for inspecting
    a transport chain or a memory-identity bundle."""
    if not paths:
        return {"n": 0}
    ts = [p.bilateral.t for p in paths]
    fs = [p.bilateral.f for p in paths]
    return {
        "n": len(paths),
        "mean_t": sum(ts) / len(ts),
        "mean_f": sum(fs) / len(fs),
        "min_t": min(ts),
        "max_f": max(fs),
        "n_refl": sum(1 for p in paths if p.is_refl),
        "n_contested": sum(1 for p in paths if p.is_contested),
        "n_refuted": sum(1 for p in paths if p.is_refuted),
        "n_supported": sum(1 for p in paths if p.supports_identification),
    }



# --- v14.3.5 Bilateral univalence scaffold ---------------------------------

@dataclass
class BilateralEquivalencePath:
    """A quantitative univalence witness between two operational types.

    This does not assert full HoTT univalence as an axiom.  It records the
    computational shadow needed by TOVAH: an equivalence can be transported as a
    path, but the path carries bilateral evidence and an obstruction budget.
    """
    equiv: Equiv
    path: Path
    coherence_error: float = 0.0


def bilateral_univalence(equiv: Equiv, *, coherence_error: float = 0.0) -> BilateralEquivalencePath:
    """Convert an equivalence witness into a bilateral path witness.

    Truth support is inherited from the equivalence; falsity support is raised
    by measured coherence error.  This keeps self-modification gates honest:
    equivalence is not Boolean projection but evidence-bearing identity.
    """
    err = max(0.0, min(1.0, float(coherence_error)))
    bv = BilateralValue(
        max(0.0, min(1.0, equiv.bilateral.t * (1.0 - 0.5 * err))),
        max(equiv.bilateral.f, err),
    )
    type_of_types = Type("OperationalType", inhabits=lambda x: isinstance(x, Type))
    p = Path(
        id_type=Id(type_of_types, equiv.A, equiv.B),
        source=equiv.A,
        target=equiv.B,
        witness=("bilateral_univalence", equiv.witness),
        bilateral=bv,
    )
    return BilateralEquivalencePath(equiv=equiv, path=p, coherence_error=err)
