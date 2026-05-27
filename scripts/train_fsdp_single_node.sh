#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NPROC_PER_NODE:=4}"
: "${SHARD_DIR:=tovah_corpus/stream}"
: "${PROFILE:=frontier_2b}"
python scripts/create_tiny_corpus.py --out "$SHARD_DIR"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" run_tovah.py \
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
  --batch-size 1 \
  --max-steps "${MAX_STEPS:-100}" \
  --log-every 10 \
  --save-sharded \
  --save-path "checkpoints/${PROFILE}_fsdp" \
  --metrics-path "tovah_metrics/${PROFILE}_fsdp.jsonl"
