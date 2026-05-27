#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Required env from the cluster scheduler:
#   NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE SHARD_DIR
: "${NNODES:?set NNODES}"
: "${NODE_RANK:?set NODE_RANK}"
: "${MASTER_ADDR:?set MASTER_ADDR}"
: "${MASTER_PORT:=29500}"
: "${NPROC_PER_NODE:=8}"
: "${SHARD_DIR:=/data/tovah/shards}"
: "${PROFILE:=frontier_13b}"
torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  run_tovah.py \
    --pretrain \
    --shard-dir "$SHARD_DIR" \
    --profile "$PROFILE" \
    --device cuda \
    --dtype bf16 \
    --optimizer adamw \
    --bilateral-mode shared \
    --gradient-checkpointing \
    --use-fsdp \
    --fsdp-mixed-precision bf16 \
    --frontier-semantic-mode hidden \
    --batch-size "${BATCH_SIZE:-1}" \
    --max-steps "${MAX_STEPS:-10000}" \
    --log-every "${LOG_EVERY:-50}" \
    --save-sharded \
    --save-path "${SAVE_PATH:-checkpoints/${PROFILE}_fsdp}" \
    --metrics-path "${METRICS_PATH:-tovah_metrics/${PROFILE}_fsdp.jsonl}"
