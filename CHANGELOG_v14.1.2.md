# TOVAH v14.1.2 — Pretraining-Readiness Audit Fixes

**Release date:** 2026-05-11
**Theme:** Fix every P0/P1 issue identified in the v14.1.1 pretraining-readiness audit.

## Summary

v14.1.1 closed the operational loop (kernel → corpus → training step → kernel state). v14.1.2 makes that loop produce *meaningful training signal* by fixing five P0 showstoppers and four P1 issues that would have caused real pretraining runs to fail silently or produce garbage. All fixes are verified empirically end-to-end.

**Tests:** 364 → 382 (18 new audit-fix regression tests added). All green on three consecutive runs.

---

## P0 (Showstoppers — would have made any real training run fail)

### P0-1. Mean-reduce the bilateral / semantic loss

**Was:** `semantic_rank_nullity_loss` returned `α·sum(min(T,F)) + β·sum(1-max(T,F))` — a raw sum over `B×L×V ≈ 327K` elements yielding magnitudes around 250,000. After the 0.3 weighting in the total loss, semantic regularization outweighed cross-entropy by ~12,000x. Training was effectively just bilateral-mass minimization with next-token prediction as a rounding error.

**Now:** Returns `α·mean(min(T,F)) + β·mean(1-max(T,F))`, plus the same softplus budget terms on the same averages. Loss is in O(1) range and behaves correctly with cross-entropy in the same total loss.

**Empirical verification (post-fix, freshly-init kernel):**
| Batch size | v14.1.1 loss | v14.1.2 loss |
|-----------:|-------------:|-------------:|
| 1          | 7,246        | 8.50         |
| 4          | 27,546       | 7.44         |
| 8          | 51,954       | 6.60         |
| 16         | 98,624       | 5.97         |
| 32         | 188,478      | (similar)    |

Loss now decreases with batch size (more averaging, less noise) instead of growing monotonically.

**Warning:** This change invalidates every checkpoint trained under the old objective. The semantics are not back-compatible.

**File:** `neural/training.py`

### P0-2. Independent T/F evidence so K-class is mathematically reachable

**Was:** `ExperienceStore.record` derived `T = 0.5 + 0.5·reward, F = 0.5 - 0.5·reward`, so `T + F = 1` always, so `min(T,F) ≤ 0.5`, so the K-class threshold `T ≥ 0.55 AND F ≥ 0.55` was unreachable from any single reward signal. Empirically: 0 K-class examples in 170 real records. The entire claim of "phase-aware sampling with 25% K-class" was operationally vacuous.

**Now:** `record()` accepts optional independent `truth_evidence` and `falsity_evidence` arguments (each ∈ [0,1]). When both are supplied, the bilateral assessment is set directly from them — so a research finding with both confirming and refuting evidence sets `T=0.8, F=0.75` and lands in K-class. When neither is supplied, the legacy reward-based derivation runs unchanged.

**Empirical verification (post-fix):** with 60 A-records (`reward_signal=0.8`), 20 K-records (`truth_evidence=0.8, falsity_evidence=0.75`), and 10 B-records (`reward_signal=-0.7`):
- Class distribution: **A=60, B=10, K=20, G=0** ← K is no longer zero
- 22% K-class in the corpus

**File:** `selfmodel/experience.py`

### P0-3. Content-fingerprint lineage IDs so dedup actually fires

**Was:** `_experience_to_example`, `_packet_to_example`, `_mutation_to_example`, `_memory_to_example`, `_wave_outcome_to_example` all included timestamps in the lineage hash. The same logical record recorded at two times got two different lineage IDs, so `deduplicate()` collapsed 0% of operationally identical records. The 47% dedup ratio reported in v14.1.0 was an artifact of state-file replay (no timestamp variance), not live behaviour.

**Now:** All six extractor functions use content-only fingerprints (record_id when present, otherwise hash of content fields like kind + text head + payload + patch name). Timestamps are stored on the resulting `TrainingExample` but are not in the lineage.

**Empirical verification:** Recording the same `record_id` five times across distinct timestamps:
- v14.1.1: 5 distinct lineage IDs, 0 collapsed
- v14.1.2: 1 lineage ID, 4 collapsed (`MERGE_WITH_PROVENANCE` strategy preserved)

**File:** `training/corpus_builder.py` (six functions touched)

### P0-4. Raise `max_len` and strip envelope so the model sees real content

**Was:** 47% of typical experience records were silently truncated at byte 320; 61% of total text bytes in the corpus never reached the model. The model's training text was 41.5% structural JSON envelope ("`[experience kind=research]\n...`"), so the model wasted capacity learning to predict our square-bracket bookkeeping.

**Now (three sub-fixes):**

1. `max_len` raised in every profile in `config/constants.py`:
   | Profile  | v14.1.1 | v14.1.2 |
   |---------:|--------:|--------:|
   | debug    | 256     | 512     |
   | standard | 320     | 1024    |
   | heavy    | 384     | 1024    |
   | large    | 512     | 1024    |
2. New `strip_envelope()` helper in `corpus_builder.py` removes the structural `[kind ...]` prefix from the training text body. Envelope information moves to `metadata["envelope"]` where it belongs.
3. New `_chunk_text()` helper splits long texts (e.g., multi-finding research outputs, mutation diffs) into UTF-8-boundary-safe chunks with configurable overlap. Used by the new `CorpusShardDataset` when `chunk_long_text=True` (default).

Also wired into live operation: `_sample_live_corpus` in `kernel/kernel.py` now strips envelope before yielding samples to `train_shadow_step`.

**File:** `config/constants.py`, `training/corpus_builder.py`, `kernel/kernel.py`

### P0-5. Build a model evaluation harness

**Was:** Zero hits for `perplexity`, `val_loss`, `eval_loss`, `held_out`, `validation` (in the model-quality sense), or `tokenizer` in the entire codebase. The "13/13 capability tests pass" reported every cycle were structural sanity checks (does `shadow_model` exist? is `task_queue` a list?), not model quality. There was no way to tell whether training was working.

**Now:** New `training/eval.py` module exposes:
- `split_train_val(shard_dir, val_fraction)` — shard-granularity split (validation shards never seen during training).
- `held_out_perplexity()` — returns `{perplexity, bits_per_byte, cross_entropy_nats, n_tokens, n_examples}`.
- `token_top1_accuracy()` — fraction of next-byte predictions where `argmax == truth`. Random baseline reported alongside (1/256).
- `gen_sample(model, prompt, max_tokens, temperature)` — qualitative generation probe with optional greedy decoding.
- `detect_divergence(loss_history, window, blowup_ratio)` — flags NaN/Inf, monotone-increasing tail, and blow-ups (current > 10× recent median).
- `bilateral_calibration()` — Pearson correlation between predicted entropy and labeled (T, F) mass.
- `run_full_eval()` — runs all of the above and returns one composite dict.

The new `pretrain()` calls `run_full_eval()` at end of each epoch and (optionally) every `eval_every_steps`. A non-finite loss or divergence flag triggers automatic rollback to the most recent snapshot.

**Empirical verification (post-fix):** `held_out_perplexity` returns `ppl=90.09, bits_per_byte=6.49, n_tokens=7280` on a freshly-init kernel — sensible numbers (random init on byte vocab → log₂(256) = 8 bits/byte; 6.49 reflects model already having some structural bias from initialization).

**File:** `training/eval.py` (new)

---

## P1 (Will work badly without these — training runs but results disappoint)

### P1-1. Tokenizer abstraction (byte / BPE)

**Was:** Byte-level only, vocab=256, hardcoded `encode_bytes()`. Cannot leverage pre-trained tokenizers; ~3.5× sequence-length penalty on natural language vs BPE.

**Now:** `training/tokenizer.py` with:
- `ByteTokenizer` (always available, no deps)
- `BPETokenizer` wrapping HuggingFace `tokenizers` (optional dep)
- `train_bpe(shard_dir, vocab_size, save_path)` to train a BPE on accumulated corpus
- `load_tokenizer(spec)` factory: `"byte"` or path to `.json`
- Selectable via `TOVAH_TOKENIZER` env var or `pretrain(tokenizer=...)` argument

`pretrain()` validates that `tokenizer.vocab_size == model.vocab_size`; mismatch is a loud error, not a silent truncation.

**File:** `training/tokenizer.py` (new)

### P1-2. DataLoader, Dataset, num_workers, pin_memory

**Was:** `pretrain()` loaded the whole corpus into Python memory, parsed JSON on the main thread, offered no shuffling or prefetching. Zero hits for `DataLoader` or `Dataset` in the entire codebase.

**Now:** `training/dataset.py` provides:
- `CorpusShardDataset(IterableDataset)` — worker-aware streaming over JSONL shards. Supports `class_filter`, `kind_filter`, envelope stripping, long-text chunking, length-stratified sampling.
- `build_collate_fn(tokenizer, max_len, pad_id)` — returns a collate function that tokenizes batches and produces `{input_ids, target_ids, attention_mask, bilateral_t, bilateral_f, kinds, paraconsistent_classes}`.

`pretrain()` uses these via `torch.utils.data.DataLoader` with configurable `num_workers`, `pin_memory`, `drop_last`. Multi-worker JSON parsing unlocks 2-4× GPU utilization at non-trivial model sizes.

**File:** `training/dataset.py` (new)

### P1-3. Mixed precision (bf16 / fp16)

**Was:** Full fp32 always. No autocast. Modern GPUs idle on tensor cores.

**Now:** `pretrain(dtype="bf16"|"fp16"|"fp32")` argument. When CUDA + non-fp32, wraps forward in `torch.autocast(device_type="cuda", dtype=...)`. CPU falls back to fp32 with a warning.

**File:** `training/pretrain.py`

### P1-4. LR warmup + cosine decay

**Was:** `ShadowOptimizer` had phase-aware scaling (Classical/Active/Collapse) but no time-based schedule. Sufficient for < 10K steps; would plateau or diverge for any real pretraining run.

**Now:** `ShadowOptimizer.set_schedule(warmup_steps, total_steps, min_lr_ratio)` enables linear warmup → cosine decay. The phase-aware multiplier still applies on top, preserving the bilateral semantics. `pretrain(warmup_steps=N)` wires it up automatically; `last_stats` exposes `scheduled_lr` and `step` for visibility.

**File:** `neural/optimizer.py`

### P3. NaN/Inf rollback on divergence + periodic checkpointing

**Was:** A bf16 + un-warmed-up LR run that produced NaN gradients would silently poison the optimizer state. No rollback. The `_save_model_snapshot` / `_rollback_model` machinery existed but `pretrain()` never invoked either.

**Now:** `pretrain()` takes an in-memory snapshot before training and every `snapshot_every_steps`. On non-finite loss or divergence flag (with `abort_on_divergence=True`, default), restores the last snapshot and re-initializes optimizer state. Snapshots are bounded to the last 3 to keep memory tight.

**File:** `training/pretrain.py`

---

## P2 (Quality issues, not blockers but worth fixing)

- **Envelope text moved to metadata** (covered under P0-4). Every extractor in `corpus_builder.py` now stores envelope info in `metadata["envelope"]` instead of `text`.
- **Length-stratified sampling** available in `CorpusShardDataset(length_stratified=True)`. Long texts get emitted multiple times (√(len/max_len)) so they get proportionally more gradient updates than short ones.
- **Empty-text fallback placeholders** in every extractor — no example is ever emitted with empty `text`, which would have been a silent skip in earlier versions.

---

## What did NOT change (deliberate scope limits)

- **Provenance chain population in live records (P1-6 in the audit)** — still length 0 for most live records. Threading `parent_id` arguments through every kernel call site is a larger refactor than P0/P1 surgery. Punted to v14.2.
- **Foreach / fused optimizer ops (P1-5)** — savings are real but smaller than the upstream wins (P0-1 loss scale, P1-2 DataLoader, P1-3 AMP). Punted.
- **Gradient checkpointing for `large` profile** — only matters if `max_len ≥ 2048` and the activation memory is the binding constraint. Not yet needed at `max_len=1024`.
- **Multi-GPU / DDP / FSDP** — out of scope for v14.1.x. The corpus scale (~50 days continuous to Chinchilla-optimal at the `standard` profile, ~400 days at `large`) means single-GPU pretraining is the realistic target anyway.

## Strategic framing reminder

The pretraining-readiness audit recommended reframing TOVAH's training story as **continual fine-tuning of a pretrained backbone**, not from-scratch pretraining. v14.1.2 makes the pieces useful for both paths:
- The fixed loss + eval harness + DataLoader + AMP path is what fine-tuning needs.
- The bilateral / paraconsistent objective (now mean-reduced and competing fairly with cross-entropy) is the genuinely novel research signal — best studied as a fine-tuning regularizer on a backbone that already understands language modeling.

The from-scratch story is still possible (the `debug` profile with the eval harness can run overnight on CPU and produce real perplexity traces), but the audit's scale math (3 MB/day effective trainable bytes) is unchanged: 50+ days of continuous operation per Chinchilla-optimal training cycle. Fine-tuning is what gives the bilateral signal a fair scientific test against a real baseline.

---

## Files modified

`neural/training.py`, `neural/optimizer.py`, `selfmodel/experience.py`, `config/constants.py`, `training/corpus_builder.py`, `training/__init__.py`, `training/pretrain.py`, `kernel/kernel.py`, `tests/test_closed_loop_wiring.py`.

## Files added

`training/eval.py`, `training/tokenizer.py`, `training/dataset.py`, `tests/test_audit_fixes_v14_1_2.py`, `CHANGELOG_v14.1.2.md`.

## Test summary

```
================== 382 passed in 33.06s ==================
```

364 existing tests + 18 new audit-fix regression tests. Stable across runs.
