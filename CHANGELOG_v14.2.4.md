# TOVAH v14.2.4 — Frontier Readiness Pass

This pass hardens the 7B/13B scaling path without claiming that a full 13B run
has been empirically proven.

## Added

- Compact hidden-state semantic-support heads in `ScalableBilateralCore`.
- Frontier training mode `frontier_semantic_mode={auto,hidden,logits}`.
- Hidden semantic auxiliary mode avoids full-vocab `F_logits` and avoids
  computing K/G losses over `B×L×V` T/F tensors in frontier training.
- `estimate_frontier_memory(...)` launch guard for 2B/7B/13B profiles.
- FSDP mixed-precision policy helper and `wrap_fsdp(..., mixed_precision=...)`.
- Resumable checkpoint module with model, optimizer, RNG, metadata, and sharded
  FSDP-aware save surfaces.
- CLI flags:
  - `--frontier-semantic-mode auto|hidden|logits`
  - `--fsdp-mixed-precision fp32|bf16|fp16`
  - `--resume-from PATH`
  - `--save-sharded`
  - `--estimate-frontier-memory`

## Changed

- `pretrain()` now returns frontier memory estimates for frontier profiles.
- `save_path` now writes a resumable training checkpoint instead of only a bare
  model state dict.
- `AdamWWrapper` now supports `state_dict()` / `load_state_dict()`.
- `ShadowOptimizer` now has a best-effort state serialization surface.

## Honest scope

This makes TOVAH more 13B-adaptable, not 13B-proven. Tensor/pipeline
parallelism, throughput benchmarking, and real multi-node convergence testing
remain future systems work.
