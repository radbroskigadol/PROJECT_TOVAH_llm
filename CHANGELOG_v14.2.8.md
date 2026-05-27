# TOVAH v14.2.8 — Learning-Speed and Hybrid-Optimizer Patch

## Implemented

- **Strict profile validation.** Unknown profiles now raise a `ValueError` instead of silently falling back to `standard`; accidental `--profile tiny` is no longer accepted unless a real `tiny` profile is defined.
- **Portable ShadowOptimizer checkpoints.** Shadow optimizer tensor state is now serialized by parameter order via `state_by_index`, not Python object id. Legacy same-process id state is retained for backward compatibility.
- **Real gradient accumulation.** The pretraining loop now performs `loss.backward()` over microbatches and applies `optimizer.step_grads(...)` after `--grad-accum-steps` microbatches.
- **Tokenizer-aware evaluation.** Perplexity, top-1 accuracy, generation, and bilateral calibration now use the active tokenizer instead of hardcoded byte encoding.
- **BPE-main training path.** `--tokenizer auto` / `--tokenizer auto-bpe` prefer a `tokenizer.json` when present and can train one with `--train-bpe-if-missing`.
- **New CLI controls.** Added `--grad-accum-steps`, `--warmup-steps`, `--min-lr-ratio`, `--val-fraction`, `--eval-every-steps`, `--snapshot-every-steps`, `--tokenizer`, `--train-bpe-if-missing`, `--bpe-vocab-size`, `--bpe-save-path`, `--pin-memory`, `--length-stratified`, `--class-filter`, and `--kind-filter`.
- **Hybrid AdamW/Shadow optimizer.** Added `--optimizer hybrid`, which proposes AdamW and ShadowOptimizer updates and mixes them through a checkpointable 256-bit online gate.
- **Paradox corpus generator.** Added `tools/generate_paradox_corpus.py` for large offline synthetic paraconsistent corpora with explicit train/validation shards.
- **128×100 hybrid run script.** Added `scripts/run_hybrid_128.ps1` for the requested repeated short-run hybrid-gate training protocol.

## Practical intent

Use AdamW or hybrid for base language-learning speed, then use Shadow/hybrid on large paradox/glut/gap corpora to specialize TOVAH’s paraconsistent layers. The hybrid gate is experimental; metrics log `adamw_weight` and `shadow_weight` so it can be audited over 128 short runs.
