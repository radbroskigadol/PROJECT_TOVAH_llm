"""
TOVAH v14.2.0 hott/module_equivalence.py — Typed module substitutability.

Priority #4 from the architecture brief:

    Full HoTT could upgrade [tool/module contracts] from ad hoc Python
    contracts into typed dependent contracts. Instead of
        tool returns Result
    you could have
        tool returns Result satisfying property P
    And if the tool is wrapped, patched, delegated, or moved to a
    subkernel, the system asks:
        Does property P transport across that wrapper/delegation?

This module is the structural answer. We model a module's contract as
a `ModuleContract`, the relationship between two modules as a
`ContractEquivalence` (a HoTT Equiv between their contract Types), and
expose `can_substitute` for the runtime question: 'given module A is
called for, can module B fulfill the call?'

PARACONSISTENT GUARANTEE: equivalence is bilateral. If module B provides
a property that module A's contract demands AND a property that
contradicts something A's contract forbids, that K-class is surfaced —
substitution is refused, the paradox is logged.

Public:
  ModuleProperty       — a named property the module must satisfy
  ModuleContract       — name + properties + capabilities + version
  contract_type        — lift a contract to a HoTT Type
  build_equiv          — construct an Equiv between two contract Types
  can_substitute       — True iff module B can replace module A in calls
  substitution_witness — full PIdJudgment for the substitution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott.core import (
    Type, Id, Path, Equiv, refl, is_equiv, equiv_compose,
)
from tovah_v14.hott.paraconsistent import (
    IdentityClass, PIdJudgment, judge_identity, classify_path,
)


# --- Properties and contracts ----------------------------------------------

@dataclass(frozen=True)
class ModuleProperty:
    """A named property a module promises (or forbids).

    Properties are checked by a `probe` function and weighted by
    `priority` (higher priority means refutation matters more).
    """
    name: str
    probe: Callable[[Any], bool]  # called on an output / on the module itself
    priority: int = 5  # 0..10
    required: bool = True
    describe: str = ""


@dataclass
class ModuleContract:
    """The contract a module satisfies.

    A module M satisfies contract C iff M.module_obj passes every probe
    in C.guarantees and fails every probe in C.forbids.
    """
    name: str
    version: str = ""
    capabilities: Set[str] = field(default_factory=set)
    guarantees: List[ModuleProperty] = field(default_factory=list)
    forbids: List[ModuleProperty] = field(default_factory=list)
    describe: str = ""


def contract_type(contract: ModuleContract) -> Type:
    """Lift a ModuleContract to a HoTT Type whose inhabitants are
    module-objects that satisfy it.

    'inhabits' checks all guarantees pass and all forbids fail.
    """

    def _inhabits(module_obj: Any) -> bool:
        try:
            for g in contract.guarantees:
                if not g.probe(module_obj) and g.required:
                    return False
            for f in contract.forbids:
                if f.probe(module_obj) and f.required:
                    return False
            return True
        except Exception:
            return False

    return Type(
        f"Module({contract.name}@{contract.version})",
        inhabits=_inhabits,
        carrier=contract,
    )


# --- Contract equivalence --------------------------------------------------

@dataclass
class ContractEquivalence:
    """A HoTT-style equivalence between two ModuleContracts.

    Attributes:
      equiv:        an `Equiv` between contract_type(A) and contract_type(B)
      capability_overlap:  capabilities present in both
      capability_only_a:   capabilities ONLY in A (missing in B → may
                           refute substitution)
      capability_only_b:   capabilities ONLY in B (extra capabilities are
                           fine for substitution-of-A-by-B)
      guarantees_satisfied:  guarantee names that both contracts provide
      guarantees_dropped:    A-guarantees that B does NOT provide (refute)
      forbids_satisfied:     forbids both contracts respect
      forbids_violated:      A-forbids that B violates (refute)
      bilateral:             aggregate evidence for/against equivalence
      judgment:              full PIdJudgment with structured paths
    """
    equiv: Optional[Equiv]
    capability_overlap: Set[str]
    capability_only_a: Set[str]
    capability_only_b: Set[str]
    guarantees_satisfied: List[str]
    guarantees_dropped: List[str]
    forbids_satisfied: List[str]
    forbids_violated: List[str]
    bilateral: BilateralValue
    judgment: PIdJudgment


def _build_substitution_paths(A: ModuleContract, B: ModuleContract,
                              A_type: Type, B_type: Type
                              ) -> Tuple[List[Path], List[Path],
                                          Set[str], Set[str], Set[str],
                                          List[str], List[str], List[str], List[str]]:
    """Walk capabilities + guarantees + forbids and produce evidence paths.

    Returns:
      supporting, refuting paths, plus the discriminated lists for the
      ContractEquivalence record.
    """
    supporting: List[Path] = []
    refuting: List[Path] = []
    cap_a = A.capabilities or set()
    cap_b = B.capabilities or set()
    overlap = cap_a & cap_b
    only_a = cap_a - cap_b
    only_b = cap_b - cap_a

    # Each shared capability is a T-evidence path (low weight individually,
    # but accumulates).
    for cap in overlap:
        supporting.append(Path(
            id_type=Id(A_type, A, B),
            source=A, target=B,
            witness=("capability_shared", cap),
            bilateral=BilateralValue(min(1.0, 0.15), 0.0),
        ))
    # Each A-only capability is HARD F-evidence: B cannot substitute for
    # A when A requires a capability B does not provide. Single missing
    # capability is sufficient to refute substitution.
    for cap in only_a:
        refuting.append(Path(
            id_type=Id(A_type, A, B),
            source=A, target=B,
            witness=("capability_missing", cap),
            bilateral=BilateralValue(0.0, 0.75),
        ))

    # Guarantees: by name match. A's guarantee is preserved if B has the
    # same name in its guarantees.
    b_guarantee_names = {g.name for g in B.guarantees}
    b_forbid_names = {f.name for f in B.forbids}
    sat: List[str] = []
    dropped: List[str] = []
    for g in A.guarantees:
        if g.name in b_guarantee_names:
            sat.append(g.name)
            supporting.append(Path(
                id_type=Id(A_type, A, B),
                source=A, target=B,
                witness=("guarantee_preserved", g.name),
                bilateral=BilateralValue(min(1.0, 0.15 + 0.05 * g.priority), 0.0),
            ))
        else:
            dropped.append(g.name)
            if g.required:
                refuting.append(Path(
                    id_type=Id(A_type, A, B),
                    source=A, target=B,
                    witness=("guarantee_dropped", g.name),
                    bilateral=BilateralValue(0.0, min(1.0, 0.20 + 0.05 * g.priority)),
                ))

    # Forbids: A forbids P. B must also forbid P (or at least not promise it).
    fsat: List[str] = []
    fviol: List[str] = []
    for f in A.forbids:
        if f.name in b_forbid_names:
            fsat.append(f.name)
            supporting.append(Path(
                id_type=Id(A_type, A, B),
                source=A, target=B,
                witness=("forbid_shared", f.name),
                bilateral=BilateralValue(min(1.0, 0.10 + 0.04 * f.priority), 0.0),
            ))
        elif f.name in b_guarantee_names:
            # B GUARANTEES what A FORBIDS — direct refutation.
            fviol.append(f.name)
            refuting.append(Path(
                id_type=Id(A_type, A, B),
                source=A, target=B,
                witness=("forbid_violated", f.name),
                bilateral=BilateralValue(0.0, min(1.0, 0.5 + 0.05 * f.priority)),
            ))
    return (supporting, refuting,
            overlap, only_a, only_b,
            sat, dropped, fsat, fviol)


def build_equiv(A: ModuleContract, B: ModuleContract,
                f: Optional[Callable[[Any], Any]] = None,
                g: Optional[Callable[[Any], Any]] = None
                ) -> ContractEquivalence:
    """Build a ContractEquivalence between A and B.

    Optional f, g are coercion functions (default: identity). When
    A and B share the same module-object schema, the identity suffices
    and we just check the contract overlap; when they differ, the caller
    supplies translation functions and we use them for the structural
    Equiv.
    """
    A_type = contract_type(A)
    B_type = contract_type(B)
    (supp, ref,
     overlap, only_a, only_b,
     sat, dropped, fsat, fviol) = _build_substitution_paths(A, B, A_type, B_type)

    # Rule-first substitutability summary. Individual shared capabilities are
    # intentionally low-weight evidence; a contract with only shared
    # capabilities used to remain G-class because PIdJudgment.best_t takes the
    # max path strength. Once all hard requirements are satisfied and no
    # refutations exist, add one aggregate path strong enough to witness
    # substitutability. Diagnostics still retain all atomic paths above.
    if not ref:
        total_required = max(1, len(A.capabilities) + len(A.guarantees) + len(A.forbids))
        preserved = len(overlap) + len(sat) + len(fsat)
        coverage = preserved / total_required
        summary_t = max(0.75, min(1.0, 0.60 + 0.40 * coverage))
        supp.append(Path(
            id_type=Id(A_type, A, B),
            source=A, target=B,
            witness=("contract_substitution_summary", {
                "required_capabilities": sorted(A.capabilities),
                "shared_capabilities": sorted(overlap),
                "guarantees_preserved": list(sat),
                "forbids_preserved": list(fsat),
                "coverage": coverage,
            }),
            bilateral=BilateralValue(summary_t, 0.0),
        ))

    judgment = judge_identity(Id(A_type, A, B), supp, ref)

    # Build a structural Equiv only if substitution is well-defined enough.
    if judgment.class_ == IdentityClass.A:
        f = f or (lambda m: m)
        g = g or (lambda m: m)
        # Trivial homotopies in the identity-coercion case.
        eta = lambda a: refl(A_type, a)
        epsilon = lambda b: refl(B_type, b)
        eq = Equiv(A=A_type, B=B_type, f=f, g=g, eta=eta, epsilon=epsilon,
                   bilateral=judgment.bilateral)
    else:
        eq = None

    return ContractEquivalence(
        equiv=eq,
        capability_overlap=overlap,
        capability_only_a=only_a,
        capability_only_b=only_b,
        guarantees_satisfied=sat,
        guarantees_dropped=dropped,
        forbids_satisfied=fsat,
        forbids_violated=fviol,
        bilateral=judgment.bilateral,
        judgment=judgment,
    )


# --- Substitutability API --------------------------------------------------

def can_substitute(A: ModuleContract, B: ModuleContract) -> bool:
    """Return True iff module B can replace module A in calls.

    GUARANTEES OF THIS DECISION:
      - All A-required guarantees must be present in B (else False)
      - No A-forbids may be violated by B (else False — and this raises
        a K-class on the substitution-judgment)
      - All A-capabilities must be present in B
    """
    ce = build_equiv(A, B)
    if ce.judgment.class_ == IdentityClass.A:
        return True
    return False


def substitution_witness(A: ModuleContract, B: ModuleContract
                         ) -> ContractEquivalence:
    """Return the full ContractEquivalence record.

    Use this when the caller wants the diagnostics (which guarantees
    dropped, which forbids violated, bilateral evidence). For a
    bool-only answer, use can_substitute.
    """
    return build_equiv(A, B)


# --- Convenience: probes from a probe registry -----------------------------

def make_probe(name: str, predicate: Callable[[Any], bool],
               priority: int = 5, required: bool = True) -> ModuleProperty:
    """Tiny constructor for ad-hoc ModuleProperty instances."""
    return ModuleProperty(
        name=name, probe=predicate, priority=priority, required=required,
    )
