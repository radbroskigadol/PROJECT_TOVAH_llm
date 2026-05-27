# Hybrid Training Runbook

## 1. Generate a large paradox corpus

```powershell
python .\tools\generate_paradox_corpus.py --out .\tovah_corpus\paradox_big --n 200000 --shard-size 10000
```

The generator creates explicit `train_*.jsonl` and `val_*.jsonl` shards. `split_train_val()` now respects these explicit validation shards.

## 2. Train or load a BPE tokenizer

```powershell
python .\run_tovah.py --pretrain `
  --shard-dir .\tovah_corpus\paradox_big `
  --profile debug `
  --optimizer adamw `
  --tokenizer auto-bpe `
  --train-bpe-if-missing `
  --bpe-vocab-size 8192 `
  --max-steps 1 `
  --batch-size 1 `
  --device cpu `
  --save-path .\checkpoints\bpe_smoke.pt
```

After this, `tovah_corpus\paradox_big\tokenizer.json` should exist and later runs will use it automatically with `--tokenizer auto` or `--tokenizer auto-bpe`.

## 3. Run the 128 × 100-step hybrid protocol

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_hybrid_128.ps1 `
  -ShardDir .\tovah_corpus\paradox_big `
  -Profile heavy `
  -Runs 128 `
  -StepsPerRun 100 `
  -BatchSize 1 `
  -GradAccumSteps 4 `
  -Device cpu
```

Each chunk writes a checkpoint and metrics JSONL. Metrics rows include:

- `optimizer_mode`
- `adamw_weight`
- `shadow_weight`
- `grad_accum_steps`
- loss, phase, learning rate, token count

## 4. Summarize hybrid weights

```powershell
@'
import json
from pathlib import Path
rows=[]
for p in sorted(Path("runs").glob("tovah_heavy_hybrid_*_metrics.jsonl")):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d=json.loads(line)
            if d.get("adamw_weight") is not None:
                rows.append(d)
print("rows", len(rows))
if rows:
    print("last loss", rows[-1].get("loss"))
    print("last adamw_weight", rows[-1].get("adamw_weight"))
    print("last shadow_weight", rows[-1].get("shadow_weight"))
    print("avg last 50 loss", sum(r["loss"] for r in rows[-50:]) / min(50, len(rows)))
'@ | Set-Content .\summarize_hybrid.py
python .\summarize_hybrid.py
```
