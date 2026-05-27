"""
TOVAH v14.2.6 hott/ — Paraconsistent HoTT formal coherence layer.

This package is the architectural split flagged in the design brief:

    Use bilateral paraconsistency for runtime cognition.
    Use a specialized formal HoTT substrate for identity-preserving transformation.

The existing kernel/runtime is the FAST layer (BilateralValue propagation,
shadow optimizer, packet dispatch). This package is the SLOW layer:
identity types, paths, transport, J-induction, equivalence, and the
verifier surfaces built on them — patch certificates, memory identity,
module substitutability, obstruction classification.

Subpackages:
  core                — Type, Id, Path, refl, transport, J, Equiv, Sigma, Pi
  paraconsistent      — PIdJudgment, bilateral_J, bilateral_transport,
                        IdentityClass (A/B/K/G)
  patch_certificates  — Patch, TransportWitness, PatchCertificate,
                        certify_patch, verify_certificate, default_probes
  memory_identity     — MemoryReferent, classify_pair, find_genuine_conflicts
  module_equivalence  — ModuleContract, ContractEquivalence, can_substitute
  obstruction         — Cocycle, ObstructionClass, globalize,
                        lifting_obstruction (H²)
"""
from __future__ import annotations

from tovah_v14.hott.formal import (
    TypeErrorHoTT, DefinitionalEqualityError,
    Term as FormalTerm, Sort, Var, PiType, Lam, App, SigmaType, Pair, Fst, Snd,
    IdType, Refl, JElim, Ann,
    Definition as FormalDefinition, Context as FormalContext, Environment as FormalEnvironment,
    FormalHoTTChecker, free_vars, subst, rename_var, alpha_eq, pretty,
    identity_function, const_function,
)

from tovah_v14.hott.core import (
    Type, Id, Path, refl, compose, inverse,
    transport, TransportResult, J,
    Equiv, is_equiv, equiv_compose,
    Sigma, Pi, check_pi,
    TruncationLevel, DependentFamily,
    path_bilateral_summary,
)
from tovah_v14.hott.paraconsistent import (
    IdentityClass, PIdJudgment,
    classify_path, is_T_supported,
    judge_identity, combine_judgments,
    bilateral_J, bilateral_transport,
    check_refl_J_reduction, check_transport_along_refl,
    check_compose_associativity,
)
from tovah_v14.hott.patch_certificates import (
    KernelStateType, Patch, InvariantProbe, InvariantValue,
    TransportWitness, PatchCertificate,
    certify_patch, verify_certificate, default_probes,
    memory_coherence_probe, bilateral_state_coherence_probe,
    promotion_authority_probe, tool_permission_probe,
    sovereign_identity_probe, contradiction_hygiene_probe,
)
from tovah_v14.hott.memory_identity import (
    MemoryReferent, MemoryReferentType,
    build_referent, identity_path,
    PairDiagnosis, PairDiagnosisReport,
    is_genuine_conflict, classify_pair, find_genuine_conflicts,
)
from tovah_v14.hott.module_equivalence import (
    ModuleProperty, ModuleContract, ContractEquivalence,
    contract_type, build_equiv,
    can_substitute, substitution_witness, make_probe,
)
from tovah_v14.hott.obstruction import (
    AbelianGroup, IntGroup, ModGroup, NonAbelianGroup,
    LocalFragment, Overlap, TransitionSymmetry,
    Cocycle, CocycleCheck, cocycle_check,
    ObstructionClass, coboundary, is_trivializable, obstruction_class,
    GlobalizationResult, globalize,
    LiftingObstruction, lifting_obstruction,
)

__all__ = [
    # formal checker
    "TypeErrorHoTT", "DefinitionalEqualityError",
    "FormalTerm", "Sort", "Var", "PiType", "Lam", "App", "SigmaType", "Pair", "Fst", "Snd",
    "IdType", "Refl", "JElim", "Ann",
    "FormalDefinition", "FormalContext", "FormalEnvironment", "FormalHoTTChecker",
    "free_vars", "subst", "rename_var", "alpha_eq", "pretty",
    "identity_function", "const_function",
    # core
    "Type", "Id", "Path", "refl", "compose", "inverse",
    "transport", "TransportResult", "J",
    "Equiv", "is_equiv", "equiv_compose",
    "Sigma", "Pi", "check_pi",
    "TruncationLevel", "DependentFamily",
    "path_bilateral_summary",
    # paraconsistent
    "IdentityClass", "PIdJudgment",
    "classify_path", "is_T_supported",
    "judge_identity", "combine_judgments",
    "bilateral_J", "bilateral_transport",
    "check_refl_J_reduction", "check_transport_along_refl",
    "check_compose_associativity",
    # patch_certificates
    "KernelStateType", "Patch", "InvariantProbe", "InvariantValue",
    "TransportWitness", "PatchCertificate",
    "certify_patch", "verify_certificate", "default_probes",
    "memory_coherence_probe", "bilateral_state_coherence_probe",
    "promotion_authority_probe", "tool_permission_probe",
    "sovereign_identity_probe", "contradiction_hygiene_probe",
    # memory_identity
    "MemoryReferent", "MemoryReferentType",
    "build_referent", "identity_path",
    "PairDiagnosis", "PairDiagnosisReport",
    "is_genuine_conflict", "classify_pair", "find_genuine_conflicts",
    # module_equivalence
    "ModuleProperty", "ModuleContract", "ContractEquivalence",
    "contract_type", "build_equiv",
    "can_substitute", "substitution_witness", "make_probe",
    # obstruction
    "AbelianGroup", "IntGroup", "ModGroup", "NonAbelianGroup",
    "LocalFragment", "Overlap", "TransitionSymmetry",
    "Cocycle", "CocycleCheck", "cocycle_check",
    "ObstructionClass", "coboundary", "is_trivializable", "obstruction_class",
    "GlobalizationResult", "globalize",
    "LiftingObstruction", "lifting_obstruction",
]
