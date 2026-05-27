#!/usr/bin/env python3
"""Sequential, resumable generated Shadow-depth length runner.

The original PowerShell foreach can make 96/160-token probes *look* stalled on CPU.
This wrapper runs one length at a time, writes per-length logs, and can skip
already-complete JSON outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def is_complete_probe(path: Path, requested_tok: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    msd = data.get("model_shadow_depth", {}) if isinstance(data, dict) else {}
    return bool(msd.get("enabled") and msd.get("overall") and int(msd.get("max_gen_tokens", -1)) == int(requested_tok))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run v14.3.3 generated Shadow-depth probes by length")
    ap.add_argument("--eval-script", default="training/eval.py")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-template", required=True, help="Example: runs/probe_tok{tok}.json")
    ap.add_argument("--tokens", nargs="+", type=int, default=[48, 96, 160])
    ap.add_argument("--max-examples-shadow-model", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-seconds", type=int, default=0, help="0 means no timeout")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-dir", default="runs/probe_logs_v14_3_3")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for tok in args.tokens:
        out_path = Path(args.out_template.format(tok=tok))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and is_complete_probe(out_path, tok):
            print(f"[skip] tok={tok}: complete output already exists at {out_path}")
            continue

        cmd: List[str] = [
            sys.executable,
            args.eval_script,
            "--checkpoint", args.checkpoint,
            "--shard-dir", args.shard_dir,
            "--tokenizer", args.tokenizer,
            "--out", str(out_path),
            "--n-gen-samples", "0",
            "--max-examples-shadow-model", str(args.max_examples_shadow_model),
            "--shadow-model-max-gen-tokens", str(tok),
            "--shadow-model-temperature", str(args.temperature),
        ]
        log_path = log_dir / f"probe_tok{tok}.log"
        print(f"[run] tok={tok}: {' '.join(cmd)}")
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("+ " + " ".join(cmd) + "\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=(args.timeout_seconds or None),
                    check=False,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                print(f"[timeout] tok={tok} exceeded {args.timeout_seconds}s; log={log_path}")
                return 124
        elapsed = time.time() - start
        if proc.returncode != 0:
            print(f"[fail] tok={tok}: exit={proc.returncode}, elapsed={elapsed:.1f}s, log={log_path}")
            return proc.returncode
        if not is_complete_probe(out_path, tok):
            print(f"[warn] tok={tok}: process exited but output is not a complete generated-probe JSON: {out_path}")
            print(f"       log={log_path}")
            return 2
        print(f"[ok] tok={tok}: complete in {elapsed:.1f}s -> {out_path}")

    print("[done] all requested token lengths finished or were skipped as complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
