#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python -m tovah_v14.evals.run_all
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python -m tovah_v14.training.scale_ladder >/tmp/tovah_scale_ladder.json
