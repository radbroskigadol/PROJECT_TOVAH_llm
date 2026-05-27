#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/create_tiny_corpus.py --out tovah_corpus/stream
python run_tovah.py \
  --pretrain \
  --shard-dir tovah_corpus/stream \
  --profile debug \
  --device cpu \
  --dtype fp32 \
  --batch-size 2 \
  --max-steps 3 \
  --log-every 1 \
  --save-path checkpoints/debug_5m.pt \
  --metrics-path tovah_metrics/debug_5m.jsonl
