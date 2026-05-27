#!/usr/bin/env python3
"""Check generated Shadow-depth probe outputs for v14.3.3.

This answers the practical question: did 96/160 stall, or are they merely slow / incomplete?
It validates JSON, extracts model_shadow_depth.overall, and prints a compact table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

KEYS = [
    "shadow_depth_mean",
    "collapse_pressure",
    "support_profile_consistency",
    "truth_falsity_calibration",
    "loop_drift_behavior",
    "contradiction_retention",
    "gap_recognition",
    "residue_preservation",
    "noncollapse_under_gluts",
]


def inspect_file(path: Path) -> Dict[str, object]:
    status: Dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "valid_json": False,
        "complete_model_shadow_depth": False,
    }
    if not path.exists():
        status["status"] = "missing"
        return status
    status["size_bytes"] = path.stat().st_size
    if path.stat().st_size == 0:
        status["status"] = "empty"
        return status
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        status["valid_json"] = True
    except Exception as exc:
        status["status"] = "invalid_json_or_partial_write"
        status["error"] = str(exc)
        return status

    msd = data.get("model_shadow_depth") if isinstance(data, dict) else None
    if not isinstance(msd, dict) or not msd.get("enabled", False):
        status["status"] = "json_without_generated_model_shadow_depth"
        return status
    overall = msd.get("overall")
    if not isinstance(overall, dict):
        status["status"] = "json_missing_model_shadow_depth_overall"
        return status

    status["complete_model_shadow_depth"] = True
    status["status"] = "complete"
    status["max_gen_tokens"] = msd.get("max_gen_tokens")
    status["metric_provenance"] = msd.get("metric_provenance")
    for k in KEYS:
        status[k] = overall.get(k)
    return status


def fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    if x is None:
        return "-"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check v14.3.3 Shadow-depth probe outputs")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--pattern", default="tovah_v14_3_2a_real_mixed_generated_probe_128_tok{tok}.json")
    ap.add_argument("--tokens", nargs="+", type=int, default=[48, 96, 160])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    rows = []
    for tok in args.tokens:
        path = runs / args.pattern.format(tok=tok)
        row = inspect_file(path)
        row["requested_tok"] = tok
        rows.append(row)

    print("requested_tok | status | file_tok | bytes | shadow_depth | collapse | support | loop | noncollapse_glut")
    print("-" * 112)
    for r in rows:
        print(
            " | ".join(
                [
                    fmt(r.get("requested_tok")),
                    fmt(r.get("status")),
                    fmt(r.get("max_gen_tokens")),
                    fmt(r.get("size_bytes")),
                    fmt(r.get("shadow_depth_mean")),
                    fmt(r.get("collapse_pressure")),
                    fmt(r.get("support_profile_consistency")),
                    fmt(r.get("loop_drift_behavior")),
                    fmt(r.get("noncollapse_under_gluts")),
                ]
            )
        )

    incomplete = [r for r in rows if r.get("status") != "complete"]
    if incomplete:
        print("\nIncomplete/missing lengths:", ", ".join(str(r["requested_tok"]) for r in incomplete))
        print("If a file is present but invalid JSON, it was probably interrupted during write; delete it and rerun that length.")
    else:
        print("\nAll requested generated-probe files are complete JSON outputs.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
