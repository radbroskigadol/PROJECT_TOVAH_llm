# ARCHITECTURE.md — TOVAH v14.2.6

## One-line summary

TOVAH is a **closed-loop paraconsistent AI kernel**: a bilateral truth/falsity neural and symbolic runtime with a specialized formal HoTT substrate for identity-preserving transformation.

## Design split

TOVAH is organized around a two-speed design:

```text
fast runtime layer:
    bilateral paraconsistent state, semantic lanes, memory, neural training, kernel loop

slow verifier layer:
    formal HoTT checker, paths, transport, J/path induction, patch certificates, memory identity, module equivalence, obstruction checks
```

The fast layer handles live adaptation. The slow layer checks whether transformations preserve identity and protected structure.

## Bilateral state

The primitive semantic carrier is a pair:

```text
BilateralValue(t, f)
```

where:

```text
t = truth/affirmation support
f = falsity/refutation support
```

Derived quantities:

```text
K / glut = min(t, f)
G / gap  = min(1 - t, 1 - f)
delta    = t - f
```

This makes contradiction and underdetermination first-class computational states.

## Four semantic lanes

The four ShadowHoTT lanes are interpretive projections over bilateral state:

```text
Lane A: classical-clean projection
Lane B: paraconsistent / glut-tolerant projection
Lane C: paracomplete / gap-tolerant projection
Lane D: forced totalization / classicalized readout
```

The training path routes metadata-heavy examples toward lanes:

```text
high classical weight -> Lane A
high K / glut         -> Lane B
high G / gap          -> Lane C
Lane D                -> reserved for forced totalization/readout, not ordinary training
```

v14.2.3 added explicit Lane B/C semantic matching:

```text
Lane B regularizer: match predicted contradiction mass K_pred to metadata K_meta
Lane C regularizer: match predicted gap mass G_pred to metadata G_meta
```

v14.2.6 adds compact hidden-state semantic heads for frontier mode so K/G auxiliary losses do not require full-vocab T/F tensors.

## Specialized formal HoTT substrate

The `hott/` package now has two layers:

```text
hott/formal.py
    bounded formal dependent-type-checking kernel

hott/core.py and verifier modules
    operational path/transport/certificate machinery used by the AI kernel
```

The formal checker implements:

```text
Type_i : Type_{i+1}
variables and transparent definitions
Π-types, lambdas, application
Σ-types, pairs, projections
Id_A(a,b)
refl_a
J/path induction
annotations
capture-avoiding substitution
beta/J normalization
alpha-equivalence
definitional equality
```

The operational verifier layer implements:

```text
Type / Id / Path witness objects
transport with bilateral evidence propagation
J-like elimination over runtime paths
Equiv
paraconsistent identity judgments
patch certificates
memory identity witnesses
module equivalence checks
local-to-global obstruction checks
```

Primary files:

```text
hott/formal.py
hott/core.py
hott/paraconsistent.py
hott/patch_certificates.py
hott/memory_identity.py
hott/module_equivalence.py
hott/obstruction.py
```

Scope limit: this is a specialized formal HoTT substrate for the UAP/ShadowHoTT kernel fragment. It is not advertised as a Lean/Coq/Agda-class general-purpose proof assistant with tactics, univalence, HITs, or a large standard library.

## Patch governance

Patch promotion is staged:

```text
proposed
static approved
sandbox passed
regression passed
shadow deployed
live promoted
revertable/quarantine paths
```

HoTT certification now fails closed. If the verifier errors or protected invariant transport fails, the patch should not move through the protected promotion boundary by default.

## Memory

Memory is conflict-preserving. Contradictory observations are not automatically overwritten. The system can retain both sides and mark conflict/glut structure.

The HoTT memory layer adds identity/provenance sensitivity: two claims only form a meaningful contradiction if the system can establish that they refer to the same relevant event/object/version/regime.

## Neural stack

There are two main neural cores:

```text
neural/shadow_model.py
    byte-level bilateral transformer research core

neural/scaling.py
    scalable bilateral transformer profiles with RoPE, GQA, RMSNorm, SwiGLU,
    tied embeddings, gradient checkpointing, hidden semantic heads
```

The scalable model supports practical frontier-style ingredients:

```text
RoPE positional encoding
GQA attention
shared or dual bilateral mode
hidden-state semantic heads
gradient checkpointing
AdamW wrapper
FSDP/DDP scaffolding
resumable checkpoint surfaces
```

## Training and corpus flow

Closed-loop corpus flow:

```text
kernel experience / patch event / module result / memory event
    -> corpus exporter
    -> quality filter and paraconsistent class metadata
    -> training shard
    -> pretraining / live training path
    -> updated model/runtime state
```

Metadata fields such as `bilateral_t` and `bilateral_f` affect semantic loss, high-glut/high-gap phase detection, lane routing, and B/C semantic matching.

## Frontier readiness

v14.2.6 adds 13B-adaptation scaffolding:

```text
frontier profile definitions
memory estimator
hidden semantic auxiliary mode
FSDP mixed precision helper
resumable checkpointing
sharded checkpoint save path
CLI launch flags
```

This is not a claim that 13B training is complete or validated. See `KNOWN_LIMITATIONS.md`.

## High-level data/control diagram

```text
          ┌────────────────────┐
          │  Live Kernel Loop   │
          └─────────┬──────────┘
                    │ events / traces / patch outcomes
                    ▼
          ┌────────────────────┐
          │ Closed-loop Corpus │
          └─────────┬──────────┘
                    │ metadata-bearing examples
                    ▼
          ┌────────────────────┐
          │  Bilateral Model   │
          │  T/F + lanes       │
          └─────────┬──────────┘
                    │ proposed changes / memories / modules
                    ▼
          ┌────────────────────┐
          │ HoTT Verification  │
          │ paths + transport  │
          └─────────┬──────────┘
                    │ accepted/rejected certificates
                    ▼
          ┌────────────────────┐
          │ Promotion / Memory │
          │ Governance         │
          └────────────────────┘
```
