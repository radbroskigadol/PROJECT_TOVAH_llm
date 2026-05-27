# SCALE_READINESS.md — TOVAH v14.2.6

This release adds a buyer-facing scale handoff layer. It does **not** claim that
a 13B TOVAH model has already been trained. It provides the artifacts a buyer's
ML engineering team needs to evaluate and extend the path from debug validation
to multi-GPU frontier-scale training.

## What is included

- Reference scale ladder from `debug_5m` to `frontier_13b_reference`.
- FSDP runbooks and shell launchers.
- Compact hidden-state semantic heads for frontier K/G losses.
- Checkpoint save/load with manifest files.
- JSONL metric logger for scale runs.
- Lightweight eval harness for ShadowHoTT-specific behavior.
- Tokenizer, data pipeline, and security runbooks.

## Recommended buyer validation sequence

1. Run `scripts/eval_smoke.sh`.
2. Run `scripts/train_debug_cpu.sh`.
3. Run `python run_tovah.py --pretrain --profile frontier_13b --estimate-frontier-memory --batch-size 1 --dtype bf16 --use-fsdp --gradient-checkpointing`.
4. On GPU hardware, run `frontier_dev` with hidden semantic mode.
5. Move to `frontier_2b` FSDP after checkpoint/resume and metrics are clean.
6. Attempt 7B/13B only after data streaming and checkpoint recovery have been rehearsed.

## Claim boundary

Accurate claim: TOVAH has a specialized formal HoTT substrate, scale-ready
reference configs, and 13B adaptation scaffolding.

Do not claim: completed 13B training, production-grade distributed trainer,
third-party benchmark superiority, or Lean/Coq-class proof assistant completeness.
