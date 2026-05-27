# FORMAL_HOTT.md — v14.2.6 Formal HoTT Substrate

v14.2.6 adds a real executable dependent-type-checking kernel at:

```text
hott/formal.py
```

This is the buyer-safe distinction:

```text
TOVAH now contains a formal HoTT-style dependent type checker for the fragment
needed by the kernel: universes, Π, Σ, Id, refl, J/path induction,
normalization, substitution, alpha-equivalence, and definitional equality.

It is not advertised as a Lean/Coq/Agda replacement or a complete general-purpose
proof assistant with tactics, elaboration, inductive families, univalence, HITs,
or a large standard library.
```

## Implemented calculus

The formal checker implements:

```text
Type_i : Type_{i+1}
variables and transparent definitions
Π (x : A), B(x)
λ-abstraction and application
Σ (x : A), B(x)
dependent pairs and projections
Id_A(a,b)
refl_a : Id_A(a,a)
J/path induction for identity elimination
annotations
capture-avoiding substitution
beta/J normalization
alpha-equivalence
definitional equality
```

## Why this matters for TOVAH

The earlier `hott/core.py` layer provides operational HoTT witness objects:
`Type`, `Path`, `transport`, `J`, patch certificates, module equivalence, memory
identity, and obstruction classifiers.

The new `hott/formal.py` layer provides the formal checker underneath that
architecture. It lets TOVAH distinguish:

```text
an arbitrary Python witness
```

from:

```text
a term accepted by a dependent type checker for the relevant HoTT fragment
```

This is the correct place to grow UAP/ShadowHoTT proofs into machine-checkable
certificates.

## Example

```python
from tovah_v14.hott.formal import *

checker = FormalHoTTChecker()
checker.add_axiom("A", Sort(0))       # A : Type0
checker.add_axiom("a", Var("A"))      # a : A

A = Var("A")
a = Var("a")
idA = Lam("x", A, Var("x"))
id_type = PiType("x", A, A)

checker.check(idA, id_type)
assert checker.defeq(App(idA, a), a)
assert checker.defeq(checker.infer(Refl(a)), IdType(A, a, a))
```

## Current boundary

The checker intentionally does not yet implement:

```text
parser/elaborator/tactic language
inductive type declarations beyond the primitive Id/Π/Σ core
univalence as an axiom/schema
higher inductive types
quotient/completion machinery
large standard library
interactive proof state UI
kernel extraction to an external proof assistant
```

Those are viable future layers, but v14.2.6 is already a real formal core rather
than merely architecture prose.
