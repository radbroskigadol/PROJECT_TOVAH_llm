#!/usr/bin/env python3
"""Create a tiny metadata-bearing TOVAH corpus shard for smoke tests/demos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RECORDS = [
    {"text": "classical true sample for lane A", "class": "A", "bilateral_t": 0.90, "bilateral_f": 0.10, "kind": "demo"},
    {"text": "contradictory evidence sample for lane B", "class": "K", "bilateral_t": 0.90, "bilateral_f": 0.88, "kind": "demo"},
    {"text": "underdetermined sparse evidence sample for lane C", "class": "G", "bilateral_t": 0.12, "bilateral_f": 0.10, "kind": "demo"},
    {"text": "false leaning classical sample for lane A", "class": "A", "bilateral_t": 0.08, "bilateral_f": 0.92, "kind": "demo"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tovah_corpus/stream", help="output shard directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "demo_shard_000.jsonl"
    with shard.open("w", encoding="utf-8") as fh:
        for rec in RECORDS:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(shard)


if __name__ == "__main__":
    main()
