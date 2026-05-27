"""
TOVAH v14.2.0 hott/obstruction.py — Local-to-global obstruction classifier.

Priority #5 from the architecture brief:

    Implement a small internal obstruction layer:
      local states
      overlaps
      transition symmetries
      cocycle
      globalization check
      obstruction class

    This would directly bridge TOVAH to the UAP papers.

This module is that bridge. The UAP papers work with connected 1-types,
BΣ classifiers, torsors, Čech H¹, and a higher obstruction ladder up to
H² lifting. We implement the operational pieces:

  - LocalFragment      — a piece of the system viewed locally
  - Overlap            — where two fragments meet
  - TransitionSymmetry — element of the symmetry group on an overlap
  - Cocycle            — collection of transitions satisfying compatibility
  - cocycle_check      — Čech 1-cocycle condition
  - obstruction_class  — element of H¹(X; Σ) for abelian Σ; pointed-set
                         class for nonabelian Σ
  - globalize          — attempt to glue locals into a global object;
                         returns either the global object or the
                         obstruction that prevented gluing
  - lifting_obstruction — second-order: given an extension
                         1 → A → G → Q → 1 and a Q-cocycle, the H² class

PARACONSISTENT GUARANTEE: each transition is *evidenced bilaterally*. A
'cocycle' may close with high T-evidence and high F-evidence on the same
overlap — that's a K-class obstruction. We report it explicitly instead
of pretending the cocycle is either valid or invalid.

This module is deliberately small and concrete. Full higher H^n lattice
construction is left to v15+ — the architecture brief flags that as
selective verifier work, not a runtime requirement.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, FrozenSet, Generic, Hashable,
                    Iterable, List, Optional, Set, Tuple, TypeVar)

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott.core import Type, Id, Path
from tovah_v14.hott.paraconsistent import (
    IdentityClass, PIdJudgment, classify_path,
)


# ---------------------------------------------------------------------------
# Symmetry-group abstraction
# ---------------------------------------------------------------------------

G = TypeVar("G")  # symmetry-group element type


class AbelianGroup(Generic[G]):
    """Tiny abelian-group interface. Just what cocycle/coboundary needs.

    Use IntGroup() for Z, ModGroup(n) for Z/n. For nonabelian groups,
    use NonAbelianGroup (no addition, just composition).
    """

    def identity(self) -> G:
        raise NotImplementedError

    def op(self, x: G, y: G) -> G:
        raise NotImplementedError

    def inv(self, x: G) -> G:
        raise NotImplementedError

    @property
    def abelian(self) -> bool:
        return True


class IntGroup(AbelianGroup[int]):
    def identity(self) -> int: return 0
    def op(self, x: int, y: int) -> int: return x + y
    def inv(self, x: int) -> int: return -x


class ModGroup(AbelianGroup[int]):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("ModGroup needs n > 0")
        self.n = n

    def identity(self) -> int: return 0
    def op(self, x: int, y: int) -> int: return (x + y) % self.n
    def inv(self, x: int) -> int: return (-x) % self.n


class NonAbelianGroup(Generic[G]):
    """Nonabelian-group interface (e.g. permutations, matrices).

    The op/inv must be supplied by the caller via the constructor.
    """

    def __init__(self, identity_elem: G,
                 op: Callable[[G, G], G],
                 inv: Callable[[G], G]):
        self._id = identity_elem
        self._op = op
        self._inv = inv

    def identity(self) -> G: return self._id
    def op(self, x: G, y: G) -> G: return self._op(x, y)
    def inv(self, x: G) -> G: return self._inv(x)

    @property
    def abelian(self) -> bool: return False


# ---------------------------------------------------------------------------
# Local fragments and overlaps
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalFragment:
    """A local view of a global object.

    Attributes:
      name:     identifier
      domain:   set of 'points' the fragment is defined on (we use
                hashable identifiers — the UAP papers' Uᵢ are formal)
      data:     the local data assigned to the fragment (arbitrary)
    """
    name: str
    domain: FrozenSet[Hashable]
    data: Any = None


@dataclass
class Overlap:
    """The intersection of two LocalFragments — Uᵢ ∩ Uⱼ.

    transitions: dict mapping (i, j) → g_ij ∈ Σ, where g_ij is the
    transition symmetry on the overlap.
    """
    fragment_i: str
    fragment_j: str
    domain: FrozenSet[Hashable]


@dataclass
class TransitionSymmetry(Generic[G]):
    """A single symmetry-group element g_ij decorating an overlap.

    bilateral: evidence FOR / AGAINST this being the correct transition.
    For example, if two memory bands' overlap is decorated by 'no
    swap (= identity)' with T=0.95 from concordant timestamps but F=0.5
    from one disagreement, we still get a K-class transition and
    cocycle_check flags it.
    """
    overlap: Overlap
    g: G
    bilateral: BilateralValue = field(default_factory=lambda: BilateralValue(1.0, 0.0))
    witness: Any = ""


# ---------------------------------------------------------------------------
# Cocycles
# ---------------------------------------------------------------------------

@dataclass
class Cocycle(Generic[G]):
    """A Čech 1-cocycle on a cover.

    transitions: keyed by ordered pair (i, j), value = TransitionSymmetry.
    The cocycle condition (g_ik = g_ij · g_jk on triple overlaps) is
    NOT auto-enforced — it's *checked* by `cocycle_check`. The whole
    point of the obstruction layer is that this check can fail.

    PARACONSISTENT: each transition carries bilateral evidence; the
    cocycle check returns aggregated evidence, not a Boolean.
    """
    group: Any  # AbelianGroup or NonAbelianGroup
    transitions: Dict[Tuple[str, str], TransitionSymmetry[G]] = field(default_factory=dict)


@dataclass
class CocycleCheck:
    """Result of cocycle_check on a Cocycle."""
    closes: bool  # True iff classical check passes for every triple
    bilateral: BilateralValue
    triples_checked: int
    triples_failed: List[Tuple[str, str, str]] = field(default_factory=list)
    class_: IdentityClass = IdentityClass.G
    reason: str = ""


def cocycle_check(c: Cocycle, fragments: List[LocalFragment]) -> CocycleCheck:
    """Test the Čech 1-cocycle condition on every triple.

    Condition (abelian, additive notation):
        g_ik = g_ij + g_jk    (on the triple overlap)
    Or (multiplicative / nonabelian):
        g_ik = g_ij ∘ g_jk

    PARACONSISTENT EVIDENCE: each triple contributes a bilateral path-
    of-evidence. T accumulates from triples that close; F accumulates
    from triples that fail. The aggregate is reported.
    """
    Σ = c.group
    triples = [(i.name, j.name, k.name)
               for i in fragments for j in fragments for k in fragments
               if i.name < j.name < k.name]
    failed = []
    t_sum = 0.0
    f_sum = 0.0
    n = 0
    for (i, j, k) in triples:
        try:
            gij = c.transitions[(i, j)]
            gjk = c.transitions[(j, k)]
            gik = c.transitions[(i, k)]
        except KeyError:
            # Cover gap: not a refutation but a gap.
            continue
        composed = Σ.op(gij.g, gjk.g)
        # Bilateral of the triple = min of the constituents' bilateral.
        triple_t = min(gij.bilateral.t, gjk.bilateral.t, gik.bilateral.t)
        triple_f = max(gij.bilateral.f, gjk.bilateral.f, gik.bilateral.f)
        if composed == gik.g:
            t_sum += triple_t
            n += 1
        else:
            failed.append((i, j, k))
            f_sum += max(triple_f, 0.7)  # baseline F for a classical failure
            n += 1
    n = max(1, n)
    bv = BilateralValue(t_sum / n, f_sum / n)
    closes = (len(failed) == 0)
    cls = IdentityClass.of(bv.t, bv.f)
    reason = (
        "cocycle closes classically" if closes
        else f"{len(failed)} triple(s) failed the cocycle condition"
    )
    return CocycleCheck(
        closes=closes, bilateral=bv, triples_checked=n,
        triples_failed=failed, class_=cls, reason=reason,
    )


# ---------------------------------------------------------------------------
# Obstruction class
# ---------------------------------------------------------------------------

@dataclass
class ObstructionClass(Generic[G]):
    """An element (or pointed set element) of the H¹ obstruction.

    For abelian Σ: an honest cohomology class — the (sums of) cocycle
    classes modulo coboundaries. Represented here by a representative
    cocycle and the equivalence-relation discriminator.

    For nonabelian Σ: pointed set H¹(X; Σ) — represented by the cocycle
    and a "trivializability" flag computed via `is_trivializable`.
    """
    cocycle: Cocycle[G]
    is_trivial: bool
    representative: Cocycle[G]
    bilateral: BilateralValue
    class_: IdentityClass
    reason: str


def coboundary(group: Any, fragments: List[LocalFragment],
               assignment: Dict[str, G]) -> Cocycle[G]:
    """Build the coboundary cocycle from a 0-cochain.

    Given a_i for each i in `assignment`, returns the cocycle
    g_ij = a_i · a_j^{-1} (multiplicative) or a_i - a_j (additive).

    The image of `coboundary` is the set of TRIVIAL cocycles. A cocycle
    is in the same H¹-class as zero iff it is a coboundary.
    """
    c = Cocycle(group=group)
    for i in fragments:
        for j in fragments:
            if i.name == j.name:
                continue
            ai = assignment.get(i.name, group.identity())
            aj = assignment.get(j.name, group.identity())
            g_ij = group.op(ai, group.inv(aj))
            overlap = Overlap(i.name, j.name, i.domain & j.domain)
            c.transitions[(i.name, j.name)] = TransitionSymmetry(
                overlap=overlap, g=g_ij,
                bilateral=BilateralValue(1.0, 0.0),
                witness=("coboundary", i.name, j.name),
            )
    return c


def is_trivializable(cocycle: Cocycle[G], fragments: List[LocalFragment]
                     ) -> Tuple[bool, Optional[Dict[str, G]]]:
    """Check whether `cocycle` is a coboundary.

    Convention: `coboundary` sets g_ij = a_i · a_j^{-1} (= a_i - a_j
    additively). Earlier versions recovered assignments only from direct
    pivot edges, which falsely rejected chain-shaped connected covers.

    This implementation propagates a 0-cochain by graph traversal over all
    available transition edges. Each connected component is pinned to the
    group identity, then every transition is verified against
    g_ij = a_i · a_j^{-1}. This handles star, chain, and sparse connected
    covers while preserving the same convention as `coboundary()`.
    """
    if not fragments:
        return True, {}
    Σ = cocycle.group
    names = [f.name for f in fragments]
    known: Dict[str, G] = {}
    neighbors: Dict[str, List[Tuple[str, str, TransitionSymmetry[G]]]] = defaultdict(list)
    for (i, j), trans in cocycle.transitions.items():
        neighbors[i].append(("forward", j, trans))   # known i -> solve j
        neighbors[j].append(("reverse", i, trans))   # known j -> solve i

    def _solve_forward(ai: G, gij: G) -> G:
        # gij = ai · inv(aj). Hence inv(aj) = inv(ai) · gij.
        return Σ.inv(Σ.op(Σ.inv(ai), gij))

    def _solve_reverse(aj: G, gij: G) -> G:
        # Here known endpoint is j for transition i->j. gij = ai · inv(aj),
        # so ai = gij · aj.
        return Σ.op(gij, aj)

    for root in names:
        if root in known:
            continue
        known[root] = Σ.identity()
        queue = [root]
        while queue:
            current = queue.pop(0)
            a_current = known[current]
            for direction, other, trans in neighbors.get(current, []):
                if direction == "forward":
                    candidate = _solve_forward(a_current, trans.g)
                else:
                    candidate = _solve_reverse(a_current, trans.g)
                if other in known:
                    if known[other] != candidate:
                        return False, None
                    continue
                known[other] = candidate
                queue.append(other)

    # Verify: every supplied g_ij should equal a_i · a_j^{-1}.
    for (i, j), trans in cocycle.transitions.items():
        ai = known.get(i, Σ.identity())
        aj = known.get(j, Σ.identity())
        expected = Σ.op(ai, Σ.inv(aj))
        if expected != trans.g:
            return False, None
    return True, known

def obstruction_class(cocycle: Cocycle[G], fragments: List[LocalFragment]
                      ) -> ObstructionClass[G]:
    """Compute the (operational) H¹ obstruction class.

    For abelian Σ: trivial iff the cocycle is a coboundary; otherwise
    the class is nontrivial. We don't represent the full quotient; we
    report only the trivial-or-not bit with the bilateral evidence.

    PARACONSISTENT: the bilateral evidence is aggregated from the
    cocycle's transitions and the cocycle-check result.
    """
    check = cocycle_check(cocycle, fragments)
    if not check.closes:
        # If the cocycle doesn't even close, the H¹ question doesn't
        # cleanly apply — we surface this as a refuted obstruction.
        return ObstructionClass(
            cocycle=cocycle, is_trivial=False, representative=cocycle,
            bilateral=BilateralValue(0.1, max(0.7, check.bilateral.f)),
            class_=IdentityClass.B,
            reason=f"cocycle does not close: {check.reason}",
        )
    is_triv, _assignment = is_trivializable(cocycle, fragments)
    bv = check.bilateral
    if is_triv:
        return ObstructionClass(
            cocycle=cocycle, is_trivial=True, representative=cocycle,
            bilateral=bv, class_=IdentityClass.of(bv.t, bv.f),
            reason="cocycle is a coboundary; obstruction is trivial",
        )
    else:
        return ObstructionClass(
            cocycle=cocycle, is_trivial=False, representative=cocycle,
            bilateral=BilateralValue(min(0.5, bv.t), max(0.6, 1 - bv.t)),
            class_=IdentityClass.B,
            reason="cocycle is not a coboundary; nontrivial obstruction to globalization",
        )


# ---------------------------------------------------------------------------
# Globalization
# ---------------------------------------------------------------------------

@dataclass
class GlobalizationResult:
    """Outcome of attempting to glue local fragments along a cocycle."""
    success: bool
    global_object: Optional[Any]  # only populated on success
    obstruction: Optional[ObstructionClass]  # only populated on failure
    bilateral: BilateralValue
    reason: str


def globalize(fragments: List[LocalFragment],
              cocycle: Cocycle[G],
              glue: Optional[Callable[[List[LocalFragment]], Any]] = None
              ) -> GlobalizationResult:
    """Attempt to glue local fragments into a global object.

    Args:
      fragments: the local pieces (their `data` is what gets glued)
      cocycle:   the transitions between fragments
      glue:      optional caller-supplied gluing function. By default we
                 form a 'tagged union' over fragments (each fragment's
                 data labeled by its name) — the trivial gluing that
                 works whenever the obstruction class is trivial.

    Returns a GlobalizationResult:
      success=True if the obstruction class is trivial (cocycle is a
      coboundary). The result.global_object is the glue function's output.

      success=False if the obstruction is nontrivial. The result.obstruction
      carries the H¹ representative for diagnostics.
    """
    obs = obstruction_class(cocycle, fragments)
    if obs.is_trivial and obs.class_ != IdentityClass.B:
        if glue is None:
            glob = {f.name: f.data for f in fragments}
        else:
            glob = glue(fragments)
        return GlobalizationResult(
            success=True, global_object=glob, obstruction=None,
            bilateral=obs.bilateral,
            reason="locals glue (cocycle is a coboundary)",
        )
    return GlobalizationResult(
        success=False, global_object=None, obstruction=obs,
        bilateral=obs.bilateral,
        reason=obs.reason,
    )


# ---------------------------------------------------------------------------
# Lifting obstruction (H², scaffold)
# ---------------------------------------------------------------------------

@dataclass
class LiftingObstruction(Generic[G]):
    """H² lifting obstruction, scaffold implementation.

    Given a central extension 1 → A → Ĝ → Q → 1 and a Q-valued cocycle
    {q_ij}, ask: can we lift to a Ĝ-valued cocycle {ĝ_ij}? The
    obstruction is a class in H²(X; A).

    This implementation:
      - Accepts a Q-cocycle and a section s : Q → Ĝ (lifts to a
        candidate ĝ_ij = s(q_ij))
      - Computes the 2-cochain δ(ĝ) (the failure of the lifts to satisfy
        the multiplicative cocycle condition on triples)
      - Returns the 2-cochain as the operational 'H² class
        representative'

    For abelian A this matches the standard derivation; for nonabelian
    contexts it's a sketch. v15 work.
    """
    A_group: AbelianGroup  # the central subgroup
    Q_cocycle: Cocycle  # the original Q-cocycle
    section: Callable[[Any], Any]  # s : Q → Ĝ
    failures: Dict[Tuple[str, str, str], Any]  # triple → δ(s)(i,j,k) in A
    bilateral: BilateralValue
    class_: IdentityClass
    reason: str


def lifting_obstruction(Q_cocycle: Cocycle, A_group: AbelianGroup,
                        section: Callable[[Any], Any],
                        Ghat_op: Callable[[Any, Any], Any],
                        Ghat_inv: Callable[[Any], Any],
                        project_to_A: Callable[[Any], Any],
                        fragments: List[LocalFragment]
                        ) -> LiftingObstruction:
    """Compute the operational lifting obstruction (scaffold).

    For every triple, set ĝ_ij = section(q_ij). The candidate cocycle
    condition is ĝ_ik = ĝ_ij · ĝ_jk. The DIFFERENCE
        δ(s)(i,j,k) = ĝ_ik · (ĝ_ij · ĝ_jk)^{-1}
    lives in the central subgroup A (by the assumption that the
    projection Ĝ → Q satisfies q(ĝ_ij · ĝ_jk) = q_ij q_jk = q_ik). We
    record this difference per triple as the operational 2-cochain.
    """
    failures: Dict[Tuple[str, str, str], Any] = {}
    triples = [(i.name, j.name, k.name)
               for i in fragments for j in fragments for k in fragments
               if i.name < j.name < k.name]
    nontriv = 0
    triples_seen = 0
    for (i, j, k) in triples:
        if not ((i, j) in Q_cocycle.transitions
                and (j, k) in Q_cocycle.transitions
                and (i, k) in Q_cocycle.transitions):
            continue
        triples_seen += 1
        q_ij = Q_cocycle.transitions[(i, j)].g
        q_jk = Q_cocycle.transitions[(j, k)].g
        q_ik = Q_cocycle.transitions[(i, k)].g
        g_ij = section(q_ij)
        g_jk = section(q_jk)
        g_ik = section(q_ik)
        # δ = g_ik · (g_ij · g_jk)^{-1}, projected to A.
        composed = Ghat_op(g_ij, g_jk)
        delta_hat = Ghat_op(g_ik, Ghat_inv(composed))
        delta_A = project_to_A(delta_hat)
        if delta_A != A_group.identity():
            failures[(i, j, k)] = delta_A
            nontriv += 1
    n = max(1, triples_seen)
    bv = BilateralValue(1.0 - nontriv / n, nontriv / n)
    return LiftingObstruction(
        A_group=A_group, Q_cocycle=Q_cocycle, section=section,
        failures=failures, bilateral=bv,
        class_=IdentityClass.of(bv.t, bv.f),
        reason=(
            "lift extends (section is a homomorphism on the cover)"
            if nontriv == 0
            else f"{nontriv} triple(s) have nontrivial δ; H² class is nontrivial"
        ),
    )



# --- v14.3.5 non-abelian Čech twist localization ---------------------------

@dataclass(frozen=True)
class NonAbelianTwist(Generic[G]):
    triple: Tuple[str, str, str]
    expected: G
    actual: G
    bilateral: BilateralValue
    reason: str


def nonabelian_twists(cocycle: Cocycle[G], fragments: List[LocalFragment]) -> List[NonAbelianTwist[G]]:
    """Return localized non-abelian triple twists g_ij·g_jk·g_ik^{-1}."""
    Σ = cocycle.group
    twists: List[NonAbelianTwist[G]] = []
    triples = [(i.name, j.name, k.name)
               for i in fragments for j in fragments for k in fragments
               if i.name < j.name < k.name]
    for (i, j, k) in triples:
        if not ((i, j) in cocycle.transitions and (j, k) in cocycle.transitions and (i, k) in cocycle.transitions):
            continue
        gij = cocycle.transitions[(i, j)]
        gjk = cocycle.transitions[(j, k)]
        gik = cocycle.transitions[(i, k)]
        actual = Σ.op(gij.g, gjk.g)
        expected = gik.g
        if actual != expected:
            bv = BilateralValue(
                min(gij.bilateral.t, gjk.bilateral.t, gik.bilateral.t),
                max(0.7, gij.bilateral.f, gjk.bilateral.f, gik.bilateral.f),
            )
            twists.append(NonAbelianTwist(
                triple=(i, j, k), expected=expected, actual=actual, bilateral=bv,
                reason="non-abelian cocycle composition failed on triple overlap",
            ))
    return twists
