# SAFE_MODE.md — TOVAH v14.2.6

Safe mode is an operating posture for evaluation. It is not a formal sandbox.

## Recommended safe-mode posture

```bash
export TOVAH_SAFE_MODE=1
unset GROK_API_KEY
unset OPENAI_API_KEY
```

Run only:

```bash
python -m tovah_v14.evals.run_all
scripts/train_debug_cpu.sh
python run_tovah.py --pretrain --estimate-frontier-memory --profile frontier_13b
```

Avoid the long-running live kernel loop until the buyer has reviewed tool
permissions, patch gates, persistence directories, and environment variables.
