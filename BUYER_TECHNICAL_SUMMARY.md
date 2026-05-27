# BUYER_TECHNICAL_SUMMARY.md — TOVAH v14.2.6

## What TOVAH is

TOVAH is a research-grade autonomous AI kernel that combines:

```text
bilateral paraconsistent semantics
four semantic lanes
contradiction-preserving memory
HoTT-inspired identity/transport verification
patch governance
closed-loop corpus export
bilateral neural training
frontier-model scaffolding
```

The project is best understood as **AI kernel / neuro-symbolic research IP**, not as a finished consumer product.

## Why it is technically distinct

### 1. Bilateral truth/falsity state

TOVAH does not collapse belief into one scalar confidence. It represents truth and falsity support independently.

This gives native representations for:

```text
affirmation
rejection
contradiction / glut
underdetermination / gap
```

### 2. Four semantic lanes

The architecture distinguishes:

```text
Lane A: classical-clean reasoning/readout
Lane B: contradiction-preserving paraconsistent routing
Lane C: gap-tolerant paracomplete routing
Lane D: forced totalization/classicalization
```

### 3. specialized formal HoTT substrate

The `hott/` package implements practical witness structures for identity, path, transport, patch certificates, memory identity, module equivalence, and local-to-global obstruction checks.

This is useful for self-modification because patches can be judged not only by tests, but by whether protected invariants transport across the patch boundary.

### 4. Closed-loop self-improvement scaffolding

The kernel records experiences, memory events, patch decisions, module results, and other traces into corpus material. That material can retain contradiction/gap metadata for future training.

### 5. Frontier-readiness path

v14.2.6 adds:

```text
ScalableBilateralCore
frontier profile definitions
hidden-state semantic heads
FSDP/DDP scaffolding
mixed-precision FSDP helper
resumable checkpoints
memory estimator
CLI flags for frontier planning
```

This makes 7B/13B adaptation plausible, but not yet proven.

## Current maturity

Strongest areas:

```text
conceptual architecture
bilateral semantic primitives
HoTT-inspired verifier package
patch promotion governance
metadata-aware high-glut/high-gap losses
targeted regression tests
```

Less mature areas:

```text
full production deployment
large-scale distributed training
independent benchmarks
full proof-assistant-grade HoTT
security hardening for arbitrary tool/patch execution
trained frontier weights
```

## Suggested commercial framing

Accurate description:

```text
A paraconsistent autonomous AI kernel with HoTT-inspired coherence verification, contradiction-preserving memory, semantic lane routing, live corpus generation, and frontier-model scaffolding.
```

Avoid:

```text
Finished AGI
production-safe autonomous agent
trained 13B model
complete HoTT implementation
fully benchmarked frontier model
```

## Acquisition / licensing value drivers

Potential value comes from:

```text
novel architecture
source code depth
research differentiation
self-modification governance design
semantic treatment of contradiction/gaps
possible patent/research-paper basis
buyer-specific strategic fit
```

Potential buyer categories:

```text
AI safety labs
autonomous-agent startups
neuro-symbolic AI groups
verification/formal-methods groups
AI coding-agent companies
research-oriented venture studios
```

## Recommended diligence package

For a buyer, provide privately:

```text
this tarball
README / architecture / handoff docs
changelog history
selected test output
demo transcript or screen recording
technical walkthrough call
license/evaluation agreement
```


## v14.2.6 formal checker addition

See `FORMAL_HOTT.md` for the new bounded dependent-type-checking kernel implementing universes, Π, Σ, Id, refl, J/path induction, normalization, alpha-equivalence, and definitional equality.

## v14.2.6 scale-handoff addition

The package now includes a scale-readiness layer: reference configs/scripts, `training/scale_ladder.py`, JSONL metric logging, checkpoint manifests, and lightweight evals for high-glut preservation, gap tolerance, patch certification, memory conflict identity, and semantic lane routing.
