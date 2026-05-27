# TOVAH v14.2.5 — Formal HoTT Checker Pass

This release upgrades the buyer-facing HoTT claim from a merely "inspired"
coherence layer to a specialized formal HoTT substrate.

## Added

- `hott/formal.py`: bounded formal dependent-type-checking kernel.
- Immutable AST for HoTT terms:
  - universes `Type_i`
  - variables
  - Π-types, lambdas, application
  - Σ-types, pairs, projections
  - identity types
  - refl
  - J/path induction
  - annotations
- Transparent global definitions and axioms.
- Capture-avoiding substitution.
- Beta/J normalization.
- Alpha-equivalence and definitional equality.
- `FormalHoTTChecker` API with `infer`, `check`, `normalize`, `defeq`,
  `add_axiom`, and `add_definition`.
- `tests/test_formal_hott_checker_v14_2_5.py` with formal-kernel regression tests.
- `FORMAL_HOTT.md` explaining the implemented calculus and its limits.

## Buyer-facing positioning

Accurate claim:

> TOVAH contains a specialized formal HoTT substrate with a dependent type checker
> for the identity/path/transport/J fragment needed by autonomous-kernel
> coherence, patch certification, memory identity, module equivalence, and
> obstruction certificates.

Avoid claiming:

> TOVAH is a complete Lean/Coq/Agda-class proof assistant.

The implementation is real, but intentionally bounded to the kernel fragment
needed by the UAP/ShadowHoTT architecture.

## Verification

Targeted tests for the formal checker pass, and the broader import/compile suite
is expected to remain compatible with v14.2.4.
