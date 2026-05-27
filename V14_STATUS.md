# TOVAH v14.2.0 — Subsystem Status (Paraconsistent HoTT formal layer + frontier-scale infrastructure)

> **Presale audit note:** this file is a v14.2.0 historical status snapshot. For the current buyer-facing status, use `README.md`, `KNOWN_LIMITATIONS.md`, and `TEST_REPORT.md`. The current package is v14.2.5; it should be described as a specialized formal HoTT substrate, not a complete HoTT proof assistant.

## All manifest methods callable: 21/21
## Historical v14.2.0 tests: 460/460 reported at that snapshot (superseded by TEST_REPORT.md for v14.2.5)

| Subsystem | Status | Notes |
|-----------|--------|-------|
| **HoTT formal layer (v14.2.0)** | **NEW** | Type, Id, Path, refl, J, transport, Equiv; bilateral paraconsistent semantics over identity-types |
| **Paraconsistent J/transport** | **NEW** | bilateral_J / bilateral_transport refuse to eliminate on K-class (paradox), B-class (refuted), G-class (gap) |
| **Patch certificates** | **NEW (priority #1)** | certify_patch / verify_certificate; 6 default protected invariants; wired into promotion ladder |
| **Memory identity** | **NEW (priority #3)** | classify_pair / is_genuine_conflict; primary dims (subject/version/test) gate same-object decisions |
| **Module equivalence** | **NEW (priority #4)** | ModuleContract, ContractEquivalence, can_substitute; capability + guarantee + forbid transport |
| **Obstruction classifier** | **NEW (priority #5)** | Cocycle, cocycle_check, coboundary, is_trivializable, globalize; H² lifting scaffold; bridge to UAP papers |
| **Promotion-ladder HoTT gate** | **NEW** | Optional kernel_state_provider in advance(); block_refuted / block_paradox verdicts at regression→shadow |
| **Frontier-scale transformer** | **NEW** | ScalableBilateralCore: RoPE, GQA, RMSNorm, SwiGLU, gradient checkpointing, tied embeddings |
| **Frontier profiles** | **NEW** | frontier_dev (~91M), frontier_2b (~2.5B), frontier_7b (~6.7B), frontier_13b (~13.2B) |
| **AdamW optimizer path** | **NEW** | AdamWWrapper for scaling regime (lr=3e-4, betas=(0.9,0.95), wd=0.1); ShadowOptimizer preserved as research substrate |
| **Distributed scaffolding** | **NEW** | init_distributed, wrap_ddp, wrap_fsdp; auto-detection from RANK/WORLD_SIZE env |
| ShadowHoTT runtime | active | bilateral/cache/refresh/determinization intact |
| Neural (v13 ShadowTokenCore) | active | Preserved for bilateral-runtime research; ScalableBilateralCore for scaling |
| Tool layer | active | Budget-aware dispatch |
| Invariant/report | active | Certification, traces, diagnostics |
| Contradiction governance | active | Glut hygiene, conflict preservation; now augmented by memory_identity |
| Persistence | active | Atomic shadow saves, no init snapshot |
| Mutation/promotion | active (bounded) + HoTT certification | Optional kernel_state_provider gate |
| Memory | active | Consolidation+forgetting; conflict preservation |
| Tasks/Plans | active | Advancement+cleanup |
| Self-model | active | Integrates competence/budgets/health/blocked-growth |
| Module health | active | Bilateral health per role; influences goals |
| Autonomous cycle | active (bounded) | Records blocked growth; uses self-model for goals |
| Research | active | Structured synthesis with contradiction detection |
| Patch preflight | active | Single authoritative path; explicit create-new |
| Service discovery | active | Deterministic ranked candidates + advisor |
| Regression | active (bounded) | Lightweight tier; no unbounded neural forward |
| Continuous corpus export | active (v14.1.1) | Streams to JSONL shards |
| Phase-aware corpus sampling | active (v14.1.1) | A/K class mix |
| Batched pretraining | active (v14.1.1+v14.1.2) | pretrain() + TRAIN_FROM_CORPUS |
| Bilateral loss objective | fixed (v14.1.2 P0-1) | Mean-reduced |
| K-class reachability | fixed (v14.1.2 P0-2) | independent truth/falsity evidence args |
| Corpus deduplication | fixed (v14.1.2 P0-3) | Content-fingerprinted lineage |
| Training context window | fixed (v14.1.2 P0-4) | max_len 320→1024; envelope-stripping |
| Model evaluation harness | active (v14.1.2 P0-5) | held_out_perplexity, top1_accuracy, gen_sample, divergence, calibration |
| Tokenizer abstraction | active (v14.1.2 P1-1) | ByteTokenizer + BPETokenizer |
| DataLoader pipeline | active (v14.1.2 P1-2) | CorpusShardDataset(IterableDataset) |
| Mixed precision (AMP) | active (v14.1.2 P1-3) | pretrain(dtype='bf16') |
| LR schedule | active (v14.1.2 P1-4) | warmup → cosine decay |
| Divergence guards | active (v14.1.2 P3) | NaN/Inf rollback + snapshots |

## Architectural principle (the design split)

> **Use bilateral paraconsistency for runtime cognition; use full HoTT for identity-preserving transformation.**

The existing kernel/runtime is the FAST layer (BilateralValue propagation, ShadowOptimizer, packet dispatch). The new `hott/` package is the SLOW layer — invoked selectively at high-stakes decisions: patch promotion, module substitution, memory contradiction-judgment, local-to-global obstruction. Pure Python, no new required dependencies, fully non-breaking on the API surface.

## Not implemented / deferred (honest accounting)

| Item | Reason |
|------|--------|
| Univalence axiom + HITs | Research-grade depth; deferred per brief |
| Full dependent type checker | check_pi is sample-based; a real proof-checker is a separate project |
| Pipeline / tensor parallelism | Requires per-layer rewrites or deepspeed/accelerate; FSDP at module-boundary is what we provide |
| Megatron-style tensor-parallel bilateral attention | Splitting T/F across devices is multi-week work |
| Lifting obstruction wired into a specific gate | H² class computed but not yet bound to a promotion checkpoint; use case still being scoped |
| Provenance chain in live records (audit P1-6) | Larger refactor across kernel call sites; punted from v14.1.x |
| Multi-GPU turnkey training driver | Scaffolding provided (DDP/FSDP wrappers); external launcher (torchrun) required |
| INGEST_LEVBEL | PDF migration not done (deliberate) |
| `kernel.py` refactor | Still single 4.7k-line module |

## v14.2.0 quick verification

```bash
pip install -r tovah_v14/requirements.txt
python -m pytest tovah_v14/tests/ -q   # 460/460
```

End-to-end demo (post-build, freshly-init kernel):
- refl-J reduction holds: `J(C, d, refl_a) == d(a)` ✓
- transport along refl is identity with bilateral (1, 0) ✓
- Rogue patch (`sovereign_id` changed) → `block_refuted` with protected_failed=['sovereign_id'] ✓
- Same-version conflict detected as `SAME_OBJECT_CONFLICT`; different-version filtered as `AMBIGUOUS_IDENTIFICATION` ✓
- Module substitution refused when required capability missing (capability_only_a={'write'}) ✓
- Coboundary cocycle globalizes; non-closing cocycle refused with diagnostics ✓
- `frontier_2b` profile estimable at ~2.47B params (vocab=50,257, bilateral_mode='shared')
- `make_optimizer(kind='adamw')` builds AdamW with frontier defaults; schedule warmup→cosine works

## Required to run

Install:
```bash
pip install -r tovah_v14/requirements.txt
# Optional GPU build of torch: see https://pytorch.org/get-started/locally/
# Optional BPE backend: pip install tokenizers
```

Run:
```bash
python tovah_v14/run_tovah.py
# or, after pip install -e .:
tovah
```

Pretrain from accumulated corpus:
```bash
# Once shards exist under tovah_corpus/stream/:
echo "TRAIN_FROM_CORPUS:|3|16" > david_says.txt
# Or via Python:
from tovah_v14.training import pretrain
pretrain("tovah_corpus/stream", epochs=3, batch_size=16,
         save_path="tovah_pretrained.pt")
```
