# KNOWN_LIMITATIONS.md — TOVAH v14.2.6

This document states the current limits plainly.

## 1. Not a trained frontier model

The repository contains frontier-model scaffolding, not trained frontier weights.

The `frontier_13b` profile and memory estimator make a 13B adaptation plausible, but no 13B training run is included or certified.

## 2. Formal HoTT checker is real but bounded

The `hott/` package now includes `hott/formal.py`, a real dependent-type-checking kernel for the implemented HoTT fragment: universes, Π, Σ, Id, refl, J/path induction, normalization, substitution, alpha-equivalence, and definitional equality.

It does not yet implement:

```text
interactive parser/elaborator/tactic language
large standard library
univalence as an axiom/schema
higher inductive types
quotients/completion machinery
inductive family declarations beyond the primitive core
proof assistant UX comparable to Coq/Agda/Lean
external proof-certificate export
```

Use the phrase “specialized formal HoTT substrate” or “bounded formal HoTT checker,” not “complete Lean/Coq-class proof assistant.”

## 3. Full pytest has not been proven green in the lightweight sandbox

Targeted suites pass, and collection succeeds. Some long-running autonomy/research paths exceed lightweight sandbox time limits.

The test suite should be split into markers:

```text
fast
slow
autonomy
research
gpu
distributed
```

before serious CI.

## 4. Distributed training is scaffolded, not production-proven

The project contains FSDP/DDP scaffolding, mixed-precision helper surfaces, memory estimates, and checkpoint APIs.

Still needed for production-scale 7B/13B work:

```text
real multi-node tests
sharded checkpoint restore validation under FSDP
tensor parallelism or DeepSpeed/ZeRO strategy
throughput benchmarks
GPU memory benchmarks
failure/restart tests
profiling under actual target hardware
```

## 5. Security hardening required

The project contains autonomous tool and patch surfaces. Before any production deployment, perform a security review of:

```text
tool execution boundaries
patch staging and promotion
filesystem access
network/API access
advisor API usage
sandbox assumptions
quarantine/rollback paths
```

## 6. No independent external benchmark yet

Current validation is internal and test-based. A serious buyer or funder should expect external benchmarks for:

```text
language modeling loss / perplexity for small trained profiles
semantic K/G preservation
patch-certification reliability
memory conflict handling
agent-loop stability
frontier-profile throughput
```

## 7. Documentation is buyer-ready but not final product documentation

This documentation package is intended for technical evaluation and handoff. It is not yet polished commercial SDK documentation.

## 8. Some architecture names are aspirational

Terms like “ShadowHoTT,” “frontier,” and “closed-loop” describe the architecture and intended research direction. They should not be read as claims of completed frontier-scale training, formal proof-assistant equivalence, or production autonomy.

## 9. Legal/IP diligence still needed

Before sale or exclusive license, buyer and seller should review:

```text
authorship/provenance
third-party dependency licenses
prior distribution history
confidentiality terms
scope of transfer/license
patent/trade-secret strategy
```

## 10. Recommended honesty line

Accurate:

```text
TOVAH is a sophisticated research prototype with full source, tests, architecture notes, handoff docs, and frontier-readiness scaffolding.
```

Not yet accurate:

```text
TOVAH is a production-ready 13B frontier AI system.
```

## v14.2.6 scale limitation

The scale handoff improves buyer execution readiness, but 13B training is still unproven in this package. The recommended path is progressive validation from debug to frontier_dev to frontier_2b before any 7B/13B run.
