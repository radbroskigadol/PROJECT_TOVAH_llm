# TOVAH v14.1.1 — Closed-loop training wiring

This release does not change any of the v14.1.0 fixes — all 357 prior tests
still pass — and closes the gap identified by audit: the pretraining
corpus produced by v14.1.0 was never consumed by the live training loop.
This release wires the loop end to end.

## Test results

| Run | Result |
|---|---|
| Baseline (v14.1.0) | 357 pass / 0 fail |
| After v14.1.1 (with 7 new smoke tests) | **364 pass / 0 fail** |

## What was broken (audit recap)

- `_train_shadow_step` (called every 18 s by `run_loop`) used a hardcoded
  2-element synthetic corpus: a JSON self-summary plus the literal string
  `"shadowhott bilateral evidence four lanes constraints"`.
- `tovah_v14/training/` contained a full corpus pipeline, but nothing
  outside `EXPORT_CORPUS:` ever called `build_corpus` or `read_jsonl_shards`.
- `ContinuousExporter`'s docstring asked for boot-time instantiation that
  did not exist anywhere in the kernel.
- Five "append to corpus" surfaces were documented but never wired:
  experience records, packet log, gate decisions, module proposals, wave
  outcomes.
- `_experience_to_example` only understood the on-disk state-file schema,
  not live `asdict(ExperienceRecord)` — even if streaming had been wired,
  live records would have produced empty text bodies.

## What this release does

### 1. ContinuousExporter instantiated at boot

`ProtozoanKernel.__init__` now opens a `ContinuousExporter` against
`tovah_corpus/stream/` before `_configure_kernel_ecology` (so initial
ecology packets are captured). Best-effort: any failure logs a warning
and disables streaming without blocking boot.

### 2. Five emission surfaces wired

| Surface | Mechanism | Captures |
|---|---|---|
| `ExperienceStore.record()` | new `on_record` callback | every experience |
| `PromotionLadder.gate_log.append` (×2 sites) | new `on_gate_decision` callback | every gate decision |
| `_dispatch_kernel_packet` | `_emit_packet_to_corpus(event)` | every non-heartbeat packet |
| `_handle_module_proposal_decision` | `_emit_module_proposal_to_corpus(payload)` | every module proposal |
| `_record_wave_resolution_history` / `_record_wave_escalation_history` | `_emit_wave_outcome_to_corpus(payload, kind)` | every wave outcome |

All five helpers are defensive: missing `continuous_exporter` attribute,
disabled exporter, or any exception logs at DEBUG and continues. The
kernel's primary behaviour is never perturbed by corpus side effects.

### 3. `_train_shadow_step` now reads the corpus

When called with `corpus=None` (the live-loop case), the method now
calls `_sample_live_corpus(batch_size=8, k_class_ratio=0.25)` which:
- Globs `tovah_corpus/stream/tovah_stream_*.jsonl` (most recent 3 shards).
- Bounds work to 2000 examples per call.
- Classifies each example into A (high-T, low-F) or K (high-T, high-F).
- Draws 75% A-class and 25% K-class examples (configurable ratio).
- Falls back to the v14.0 self-summary corpus only if no shards exist.

### 4. New batched pretraining entry point

`tovah_v14/training/pretrain.py` — `pretrain(shard_dir, *, model, optimizer,
epochs, batch_size, max_examples, k_class_ratio, save_path, log_every,
device, profile_name, seed) -> dict`.

- Walks shards once, builds A/B/K/G pools.
- Draws phase-aware batches with `_draw_batch`.
- Runs N epochs of `train_shadow_step`.
- Optionally writes a `.pt` checkpoint.
- Returns a structured summary (pool sizes, epoch-avg/first/last loss,
  total steps, walltime, final phase, save path).

### 5. New `TRAIN_FROM_CORPUS` David command

Syntax:

    TRAIN_FROM_CORPUS[:<shard_dir>[|<epochs>[|<batch_size>[|<save_path>]]]]

Defaults: shard_dir=`tovah_corpus/stream`, epochs=1, batch_size=8.
- Calls `pretrain` with the live `shadow_model` and `shadow_optimizer`.
- Reflects pretraining loss into `loss_history` (so reports/traces see it).
- Updates `_training_phase` to the final phase observed.
- Records `module_health.record_success("trainer")`.
- Registered in `COMMAND_REGISTRY` with `approval_required=True`.

### 6. Bilingual experience schema

`_experience_to_example` now reads both the state-file schema
(`rec_id`, `kind`, `description`, `topic`, `findings`) and the live
`asdict(ExperienceRecord)` schema (`record_id`, `action_type`, `context`,
`outcome`, `reward_signal`, `bilateral_assessment`, `tags`). Lineage IDs
remain deterministic over equivalent records.

### 7. `large` model profile

`config/constants.py::MODEL_PROFILES["large"]` —
`d_model=512, d_hidden=2048, n_heads=8, n_blocks=12, max_len=512`
(~52M params). Selectable via `TOVAH_PROFILE=large`. Use for actual
pretraining runs; the `standard` profile remains the live-operation default.

### 8. Corpus directory paths

`config/paths.py` exposes `CORPUS_DIR` (`tovah_corpus/`) and
`CORPUS_STREAM_DIR` (`tovah_corpus/stream/`), both registered in
`ALL_DIRS` so `ensure_directories()` creates them.

### 9. Quieter silent drops in `corpus_builder.py`

The seven `except Exception: pass` blocks in `build_corpus()` have been
converted to per-record try/except with `logging.debug(...)`. A single
bad record in any source no longer drops the whole corpus, and drops are
now visible at DEBUG log level.

### 10. Packaging

- `requirements.txt` — torch, requests, plus optional pypdf, python-dotenv,
  openai, pytest.
- `pyproject.toml` — PEP 517 build config, entry point `tovah =
  tovah_v14.run_tovah:main`, pytest config, optional extras (`pdf`,
  `env`, `advisor`, `test`, `all`).

### 11. New test suite: `tests/test_closed_loop_wiring.py` (7 tests)

Covers:
- `ContinuousExporter` is wired at boot.
- `experience.record()` writes a parseable JSONL line to the stream.
- `_sample_live_corpus` reads back from on-disk shards.
- `_train_shadow_step()` with no corpus arg uses sampled data and
  produces a valid loss/phase.
- `pretrain()` runs over streamed shards and returns a structured summary.
- `TRAIN_FROM_CORPUS:|1|4` David command produces a structured response.
- `ContinuousExporter` resumes to the next shard across boots without
  truncating earlier shards.

## File-by-file change list

### Modified

- `kernel/kernel.py` — ContinuousExporter wire-up, five corpus-emit helpers,
  packet/module-proposal/wave-history hooks, `_sample_live_corpus`, rewritten
  `_train_shadow_step`, `TRAIN_FROM_CORPUS` command.
- `kernel/preflight.py` — `TRAIN_FROM_CORPUS` registered in
  `COMMAND_REGISTRY`.
- `mutation/promotion_ladder.py` — `on_gate_decision` callback;
  `_emit_gate_decision` helper; two gate_log.append sites now fire it.
- `selfmodel/experience.py` — `on_record` callback; record() invokes it.
- `training/corpus_builder.py` — bilingual `_experience_to_example`;
  logged per-record try/except in `build_corpus`.
- `training/__init__.py` — exports `pretrain`.
- `config/constants.py` — `large` model profile.
- `config/paths.py` — `CORPUS_DIR`, `CORPUS_STREAM_DIR`, `ALL_DIRS`.

### Added

- `training/pretrain.py` — batched pretraining entry point.
- `tests/test_closed_loop_wiring.py` — 7 closed-loop smoke tests.
- `requirements.txt`
- `pyproject.toml`
- `CHANGELOG_v14.1.1.md` (this file)

## Operating notes

- Stream shards land in `tovah_corpus/stream/`. They roll over at 1000
  examples per shard by default; for long-running deployments expect
  many shards. Disk-pressure cleanup is the operator's responsibility.
- `_sample_live_corpus` reads only the most recent 3 shards by design.
  For broader sampling, call `pretrain()` (or `TRAIN_FROM_CORPUS`) with
  the full `tovah_corpus/stream/` directory.
- The live-loop training step is still single-batch; `TRAIN_FROM_CORPUS`
  is the way to run real multi-epoch pretraining. The two share the same
  `train_shadow_step` core, so loss numbers are directly comparable.
- The `large` profile (~52M params) is not intended for live operation
  on CPU — boot will take noticeably longer and each training step
  will be slow. Use it for offline pretraining and either save the
  checkpoint or keep the kernel running.
