# SCALE_HANDOFF.md — TOVAH v14.2.6

This is the operational handoff index for buyers.

## Start here

1. `README.md`
2. `BUYER_TECHNICAL_SUMMARY.md`
3. `ARCHITECTURE.md`
4. `FORMAL_HOTT.md`
5. `SCALE_READINESS.md`
6. `SCALING_LADDER.md`
7. `FSDP_RUNBOOK.md`
8. `EVALS.md`
9. `SECURITY.md`

## Commands

```bash
pip install -e . --no-deps
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python -m tovah_v14.evals.run_all
scripts/train_debug_cpu.sh
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python -m tovah_v14.training.scale_ladder
python run_tovah.py --pretrain --profile frontier_13b --estimate-frontier-memory --dtype bf16 --use-fsdp --gradient-checkpointing
```

## What the buyer receives

- source tree
- tests
- specialized formal HoTT substrate
- frontier adaptation scaffolding
- scale docs/configs/scripts
- eval harness
- presale audit report
- known limitations
- license-options memo

## What the buyer should build next

- hardware-specific FSDP/TP/ZeRO-3 recipe
- production streaming dataset backend
- real tokenizer artifact and data manifest
- tiny pretrained proof-of-life checkpoint
- third-party security review
