# TOVAH v14.2.0 — Paraconsistent HoTT formal layer + Frontier-scale infrastructure

**Release date:** 2026-05-11
**Theme:** Bring the *formal* J-preserving HoTT theory into TOVAH (not just its computational shadow), and provide the architectural runway to scale the LLM to frontier parameter counts (2B–13B).

## The diagnosis this release addresses

> TOVAH implements the computational shadow of a paraconsistent HoTT idea, but not the formal J-preserving theory itself.

That's accurate, and it has two consequences:

1. **No structural backbone.** The bilateral runtime (BilateralValue, T_sup/F_sup, lanes A/B/C/D) is the computational shadow *of* identity types, paths, refl, J-induction, transport, and equivalences — but these structural objects don't exist as first-class entities. We carry the *evidence* (the bilateral values) without carrying *what the evidence is about* (the identification claim).

2. **No selective verifier.** As a consequence there's no proof-carrying patch system, no transport-aware certificates, no module-equivalence judgments, no obstruction classifier. The promotion ladder runs on tests and policies; it doesn't ask the structural question "do all protected invariants transport across this patch?"

v14.2.0 supplies the missing structural backbone and the verifier surfaces built on it. The architectural principle, from the brief:

> **Use bilateral paraconsistency for runtime cognition; use full HoTT for identity-preserving transformation.**

The existing kernel is the fast layer. The new `hott/` package is the slow layer — invoked selectively at high-stakes decision points (patch promotion, module substitution, memory contradiction-judgment).

Separately, the LLM substrate is hardcoded at d=512/12 blocks (~52M params at the `large` profile) — far below frontier scale. v14.2.0 adds the architectural pieces needed to grow to 2B–13B if/when desired: RoPE, GQA, gradient checkpointing, tied embeddings, AdamW, DDP/FSDP scaffolding.

---

## Part 1 — Paraconsistent HoTT formal layer (`hott/` package)

### Why "paraconsistent" HoTT?

Ordinary HoTT treats `Id(A; a, b)` as a Type whose inhabitants are paths. Assertions about path existence are Boolean (the type is either inhabited or not).

In paraconsistent HoTT, every path carries **bilateral evidence**: a Path has `BilateralValue(t, f)` recording evidence *for* the identification and *against* it. `refl` has `(1, 0)`. A heuristic match might be `(0.7, 0.1)`. A contested identification — same module, same version, conflicting test outcomes — is `(0.9, 0.85)`, K-class, and we *do not collapse the contradiction*. We propagate it through transport and J, and we *gate* downstream consumers on the classification.

This is the structural answer to David's brief example: "module failed" vs "module succeeded" is only a real contradiction if the referents identify (A-class on the identity-path). Otherwise the apparent paradox is a sloppy-identity-matching artifact and the brief calls this out as a key correctness improvement.

### Package layout

```
hott/
  core.py                — Type, Id, Path, refl, compose, inverse,
                           transport, J, Equiv, is_equiv, equiv_compose,
                           Sigma, Pi, TruncationLevel
  paraconsistent.py      — IdentityClass (A/B/K/G), PIdJudgment,
                           bilateral_J, bilateral_transport,
                           judge_identity, combine_judgments
  patch_certificates.py  — Patch, InvariantProbe, TransportWitness,
                           PatchCertificate, certify_patch,
                           verify_certificate, default_probes
  memory_identity.py     — MemoryReferent, identity_path, classify_pair,
                           is_genuine_conflict, find_genuine_conflicts
  module_equivalence.py  — ModuleProperty, ModuleContract,
                           ContractEquivalence, can_substitute,
                           substitution_witness, build_equiv
  obstruction.py         — IntGroup/ModGroup/NonAbelianGroup, Cocycle,
                           cocycle_check, ObstructionClass, coboundary,
                           is_trivializable, globalize,
                           LiftingObstruction, lifting_obstruction (H²)
```

79 public exports. Pure Python — no new required dependencies.

### Structural laws preserved (verified by tests)

1. **refl-J reduction**: `J(C, d, refl_a) == d(a)`. The defining law of J-induction.
2. **transport along refl is identity**: `transport(P, refl_a, x).value == x` with bilateral `(1, 0)`.
3. **compose endpoint matching**: `compose(p, q)` only defined when `p.target == q.source`.
4. **compose semantics**: bilateral of composite = `(min(t), max(f))` — paths are only as strong as their weakest link, any refutation along the chain refutes the chain.
5. **inverse preserves bilateral strength**: `inverse(p).bilateral == p.bilateral`.
6. **RoPE-style equiv composition**: `equiv_compose(e1, e2)` produces a coherent equivalence A→C from A→B and B→C, with bilateral combined under min/max.

### Paraconsistent guarantees

- `bilateral_J(C, d, judgment)` *refuses to eliminate* when the identity-judgment is K-class (genuine paradox), B-class (refuted), or G-class (no info) when gating on A. Returns `(None, refreshed_judgment)` so the caller routes to paradox-handling, never silently picks a side.
- `bilateral_transport(P, judgment, x)` has the same gating discipline.
- `combine_judgments([j1, j2])` unions the supporting and refuting pools. A high-T judgment combined with a high-F judgment yields K-class, not "the louder one wins."

### Priority #1 from the brief: Patch certificates

`certify_patch(patch, probes)` walks each protected invariant, computes its value before and after the patch, builds a `TransportWitness`, and produces a `PatchCertificate` with a verdict:

- **pass** — every protected invariant transports (`fingerprint(pre) == fingerprint(post)`)
- **warn** — some non-protected invariant changed (gap)
- **block_refuted** — at least one protected invariant changed (B-class) → promotion must be blocked
- **block_paradox** — at least one invariant has K-class evidence (contested) → blocked with paradox-class diagnostics

The brief's six default invariants are implemented and shipped: `memory_coherence`, `bilateral_state_coherence`, `promotion_authority`, `tool_permission`, `sovereign_identity`, `contradiction_hygiene`.

### Priority #3 from the brief: Memory identity

`classify_pair(m1, m2)` returns one of five diagnoses:

- **SAME_OBJECT_AGREE** — confirmed same object, compatible assessments
- **SAME_OBJECT_CONFLICT** — confirmed same object, conflicting assessments (the real contradiction)
- **DIFFERENT_OBJECT** — referents refute identification → any apparent conflict is spurious
- **AMBIGUOUS_IDENTIFICATION** — referents partially match (K-class), can't decide
- **INSUFFICIENT_INFO** — no evidence either way

The five referent dimensions, weighted as primary/secondary:

| Dimension     | Tier      | T-weight | F-penalty on mismatch |
|---------------|-----------|---------:|----------------------:|
| subject       | primary   | 0.40     | 0.65                  |
| version       | primary   | 0.30     | 0.65                  |
| test          | primary   | 0.20     | 0.65                  |
| environment   | secondary | 0.05     | 0 (T-only)            |
| time_band     | secondary | 0.05     | 0 (T-only)            |

Any **primary** mismatch produces F ≥ 0.65 → pushes the identification out of A-class. So "GateModule v3 failed sandbox_run" and "GateModule v2 failed sandbox_run" are *not* treated as the same object even though the subject matches.

`is_genuine_conflict(m1, m2)` returns True only for SAME_OBJECT_CONFLICT.
`find_genuine_conflicts(memories)` walks a pool and filters out spurious gluts.

### Priority #4 from the brief: Module equivalence

`can_substitute(A, B)` decides whether module B can substitute for module A in calls, by walking:

- A-required capabilities ⊆ B's capabilities (else: refuted)
- A's guarantees ⊆ B's guarantees (else: refuted; priority-weighted)
- A's forbids ⊆ B's forbids (or B doesn't claim them) (else: K-class on forbid-violation)

`substitution_witness(A, B)` returns the full `ContractEquivalence` with `capability_overlap`, `capability_only_a`/`only_b`, `guarantees_satisfied`/`dropped`, `forbids_satisfied`/`violated`, bilateral evidence, and the full PIdJudgment.

### Priority #5 from the brief: Obstruction classifier

This bridges TOVAH to the UAP/ShadowHoTT papers. Implements the Čech-cohomology operational layer:

- `LocalFragment`, `Overlap`, `TransitionSymmetry`, `Cocycle` data types
- `cocycle_check(c, fragments)` — verifies the 1-cocycle condition on every triple; returns `CocycleCheck` with `closes`, `bilateral`, `triples_failed`
- `coboundary(group, fragments, assignment)` — builds the coboundary of a 0-cochain; convention `g_ij = a_i · a_j^{-1}`
- `is_trivializable(cocycle, fragments)` — returns `(bool, recovered_assignment_or_None)`. Implementation uses the inverse of `g_pivot,j` to recover `a_j` (sign-correct against the coboundary convention)
- `obstruction_class(cocycle, fragments)` — `H¹(X; Σ)` class: trivial (coboundary) or nontrivial
- `globalize(fragments, cocycle, glue)` — attempt to glue locals into a global object; returns `GlobalizationResult` with either `global_object` or `obstruction`
- `lifting_obstruction(...)` — H² scaffold for central extensions; returns 2-cochain failures over triples

Supports `IntGroup` (Z), `ModGroup(n)` (Z/n), `NonAbelianGroup(...)`.

The obstruction layer is the operational language for the UAP papers' "local fragments don't glue globally" phenomenon. The brief observes:

> The current TOVAH code operationalizes the bilateral/paraconsistent runtime shadow. But the UAP papers are working at a higher structural level: local fragments, overlap symmetries, globalization failure, obstruction classes, higher lifting problems.

This module is the bridge. The classes A/B/K/G of the bilateral runtime map naturally onto the cocycle-evidence aggregation.

### Promotion ladder integration

`PromotionLadder.advance()` takes a new optional `kernel_state_provider` arg. When supplied, the `regression_passed → shadow_deployed` transition runs `certify_patch()` against the six default invariants (or caller-supplied probes) and gates on the verdict:

- `block_refuted` / `block_paradox` → the patch is rejected with the certificate logged to `gate_log`
- `pass` / `warn` → the patch advances; certificate summary logged

**Non-breaking**: every existing caller that doesn't supply `kernel_state_provider` continues to work exactly as v14.1.x. The HoTT certification is opt-in and additive.

---

## Part 2 — Frontier-scale model infrastructure

### Why this exists

The original `ShadowTokenCore` preserves the v13 architecture verbatim: byte vocab, dense attention, hardcoded dimensions, ~52M params at the `large` profile. That's the right substrate for the bilateral *research* — but to compete with frontier models you need d=2048–4096, 22–32 blocks, modern attention tricks.

v14.2.0 adds `ScalableBilateralCore` alongside `ShadowTokenCore` (not replacing it). Use ShadowTokenCore for bilateral-runtime research; use ScalableBilateralCore when you want to actually train at scale.

### `neural/scaling.py` — frontier transformer

| Feature                      | Implementation                          |
|------------------------------|-----------------------------------------|
| RoPE positional encoding     | `_apply_rope` + per-head cache buffer   |
| Grouped-query attention      | `BilateralGQAttention(n_heads, n_kv_heads)` — reduces KV-cache by group factor |
| Gradient checkpointing       | `gradient_checkpointing=True` arg; uses `torch.utils.checkpoint` per block |
| Tied embeddings              | `tied_embeddings=True` arg; halves embed memory |
| RMSNorm                      | Replaces LayerNorm; matches LLaMA/Mistral |
| SwiGLU FFN                   | With bilateral cross-mixing             |
| SDPA fast path               | Calls `F.scaled_dot_product_attention` — picks Flash/Memory-Efficient on CUDA automatically |
| `bilateral_mode="dual"`      | v13-style separate T-attn and F-attn (~10% more params) |
| `bilateral_mode="shared"`    | Single attention path; T/F mixing only at FFN+head (compute-parity with classical) |

### Profiles (named by actual param count with bilateral overhead)

| Profile        | d_model | n_blocks | n_heads | n_kv_heads | max_len | Params (vocab=50257, shared) |
|----------------|--------:|---------:|--------:|-----------:|--------:|------------------------------:|
| frontier_dev   | 512     | 6        | 8       | 4          | 1024    | ~91M                          |
| frontier_2b    | 2048    | 22       | 16      | 8          | 2048    | ~2.5B                         |
| frontier_7b    | 3072    | 28       | 24      | 8          | 4096    | ~6.7B                         |
| frontier_13b   | 4096    | 32       | 32      | 8          | 4096    | ~13.2B                        |

Honest: the bilateral T/F dual embeddings add ~2× vocab × d_model on top of any same-shape classical transformer. The profile labels reflect *actual* bilateral param count, not the classical equivalent. Use `bilateral_mode='shared'` to keep attention compute parity with a classical model at the same param budget.

### `neural/adamw.py` — AdamW path

`AdamWWrapper` thinly wraps `torch.optim.AdamW` with the `ShadowOptimizer` interface (`zero_grad`, `step`, `.lr`, `set_schedule`, `last_stats`). Standard frontier defaults: lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=True, gradient norm clipping to 1.0.

`make_optimizer(params, kind="shadow"|"adamw")` — single factory selectable via env var or `pretrain()` arg.

**The bilateral ShadowOptimizer is kept** as research substrate. AdamW is the frontier-scaling option.

### `neural/distributed.py` — DDP/FSDP scaffolding

`is_distributed_available()`, `init_distributed(backend="nccl")`, `wrap_ddp(model)`, `wrap_fsdp(model, auto_wrap_min_params=1e8)`, `rank()`, `world_size()`, `is_main()`, `barrier()`, `cleanup()`, `distributed_sampler_for_dataset(...)`.

Launch single-host 4-GPU run: `torchrun --nproc_per_node=4 run_tovah.py --pretrain --profile frontier_2b --use-fsdp`. The training loop autodetects RANK/WORLD_SIZE and wires up DDP/FSDP.

**Honest scope**: this is scaffolding. Pipeline parallelism, tensor parallelism, ZeRO-3 are NOT implemented (require Megatron-style rewrites or deepspeed/accelerate integration). FSDP is integrated at the `auto_wrap_min_params` boundary; per-block fine-grained sharding policy is v15+ work.

---

## Test summary

- 382 baseline tests (v14.1.2): **all green**
- 78 new v14.2.0 tests:
  - `test_hott_core.py`: 32 tests — structural laws (refl, transport, J, compose, equiv, Sigma/Pi, paraconsistent layer)
  - `test_hott_verifiers.py`: 18 tests — patch certificates, memory identity, module equivalence, obstruction classifier (incl. cocycle / coboundary / globalize / lifting)
  - `test_scaling.py`: 23 tests — RoPE, GQA, RMSNorm, SwiGLU, param-count estimation, forward+backward, gradient checkpointing, AdamW wrapper, distributed scaffolding
  - `test_hott_promotion_wiring.py`: 5 tests — promotion-ladder HoTT integration including back-compat

**Total: 460 / 460 pass.** Verified from a clean extraction.

```
================== 460 passed in 103.27s ==================
```

---

## What's deliberately deferred

Honest accounting of what didn't land in v14.2.0 and why:

- **Univalence**: equivalence-is-identity is a real research problem in HoTT. We supply `Equiv` and `is_equiv` as sample-based checks; we don't supply the univalence axiom or its computational content. v15+ if/when needed.
- **HITs (Higher Inductive Types)**: the brief explicitly defers these. We agree.
- **Full dependent type checker**: `check_pi` is sample-based. A real proof-checker is a separate project.
- **Pipeline/tensor parallelism**: requires per-layer rewrites or third-party integration. FSDP at the module-boundary level is what we provide.
- **Flash-Attention 2 / 3 native binding**: `torch.scaled_dot_product_attention` chooses the kernel automatically on CUDA. If you want explicit flash-attn package usage, that's a swap inside `BilateralGQAttention._attn_head`.
- **Megatron-style tensor-parallel sharding of bilateral attention**: would require splitting T and F streams across devices. Conceptually sound; multi-week project.
- **Lifting obstruction integration with the promotion ladder**: H² classes are computed by `lifting_obstruction` but not yet wired into a specific gate. Use case unclear at this scope.

## Files modified

- `mutation/promotion_ladder.py` — added optional `kernel_state_provider` + `hott_probes` to `advance()`
- `neural/__init__.py` — exports new scaling + AdamW + distributed
- `pyproject.toml` — version bump to 14.2.0

## Files added

- `hott/__init__.py`, `hott/core.py`, `hott/paraconsistent.py`, `hott/patch_certificates.py`, `hott/memory_identity.py`, `hott/module_equivalence.py`, `hott/obstruction.py`
- `neural/scaling.py`, `neural/adamw.py`, `neural/distributed.py`
- `tests/test_hott_core.py`, `tests/test_hott_verifiers.py`, `tests/test_scaling.py`, `tests/test_hott_promotion_wiring.py`
- `CHANGELOG_v14.2.0.md`

## Strategic framing

The bilateral runtime is fast. The HoTT verifier is slow but only invoked at high-stakes decisions (patch promotion, module substitution, memory contradiction-judgment). This is **selective verification** — the right model for adding structural rigor without paying the cost on every kernel cycle.

For LLM scaling: TOVAH is now architecturally capable of growing to 13B params if/when you choose. The compute requirements remain real (multi-GPU is still multi-GPU), but the architecture no longer blocks it. The bilateral T/F dual-embedding overhead is honest — frontier-classical parity at the same compute budget uses `bilateral_mode='shared'`; the v13-style fully-dual architecture remains available for research.

The brief's final line is still the design principle:

> **Use bilateral paraconsistency for runtime cognition; use full HoTT for identity-preserving transformation.**

v14.2.0 supplies a first-class HoTT-inspired verifier side; it is not a complete dependent type checker.
