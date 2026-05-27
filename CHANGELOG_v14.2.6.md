# TOVAH v14.2.6 — Scale Handoff Pass

## Added

- `SCALE_READINESS.md`
- `SCALING_LADDER.md`
- `FSDP_RUNBOOK.md`
- `TOKENIZER_STRATEGY.md`
- `DATA_PIPELINE.md`
- `EVALS.md`
- `SECURITY.md`
- `SAFE_MODE.md`
- `THREAT_MODEL.md`
- `SCALE_HANDOFF.md`
- `configs/*.yaml` reference scale configs
- `scripts/train_debug_cpu.sh`
- `scripts/train_fsdp_single_node.sh`
- `scripts/train_fsdp_multi_node.sh`
- `scripts/eval_smoke.sh`
- `scripts/create_tiny_corpus.py`
- `tovah_v14.evals` lightweight buyer-facing eval harness
- `training/scale_ladder.py`
- `training/metrics.py`

## Changed

- Version bumped to 14.2.6.
- `run_tovah.py` accepts `--metrics-path`.
- `pretrain()` can emit JSONL scale metrics.
- Checkpoint saves now write a manifest file.

## Calibration

This release improves buyer scale handoff. It does not claim completed 13B
training or production-grade distributed training.
