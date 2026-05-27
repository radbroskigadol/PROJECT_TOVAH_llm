# SCALING_LADDER.md — TOVAH v14.2.6

The scaling ladder is codified in `training/scale_ladder.py` and summarized by:

```bash
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python -m tovah_v14.training.scale_ladder
```

## Stages

| Stage | Profile | Purpose |
|---|---|---|
| `debug_5m` | `debug` | CPU install/pretrain/checkpoint smoke. |
| `research_50m` | `large` | Classic bilateral logits and ShadowOptimizer validation. |
| `frontier_dev` | `frontier_dev` | ScalableBilateralCore/RoPE/GQA/hidden semantic heads. |
| `frontier_2b` | `frontier_2b` | First serious FSDP training target. |
| `frontier_7b` | `frontier_7b` | Multi-GPU sustained training target. |
| `frontier_13b_reference` | `frontier_13b` | Buyer cluster reference plan. |

## Why hidden semantic mode matters

Frontier training still needs `T_logits` for next-token cross entropy, but K/G
semantic auxiliaries can use compact hidden-state support heads instead of full
vocabulary `T/F` sigmoid tensors. This keeps the ShadowHoTT gradient signal
without doubling the largest logit tensor.

## Required buyer measurements

For each stage, record:

- loss and validation loss
- lane entropy
- K/G predicted and metadata means
- phase distribution
- grad norm and LR
- tokens/sec and samples/sec
- GPU memory allocated/reserved
- checkpoint time and resume success
- data loading time
