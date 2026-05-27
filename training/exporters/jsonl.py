"""
TOVAH v14 training/exporters/jsonl.py — JSONL shard writer + reader.

Writes the corpus into N shards of `shard_size` examples each. The shard
filename embeds an index so a downstream loader can read deterministically.
Each line is one TrainingExample serialized as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from tovah_v14.training.corpus_builder import TrainingExample


def write_jsonl_shards(examples: List[TrainingExample],
                       out_dir: str | Path,
                       *, shard_size: int = 1000,
                       prefix: str = "tovah_corpus") -> List[Path]:
    """Write `examples` to JSONL shards. Returns list of shard file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files: List[Path] = []
    n_shards = (len(examples) + shard_size - 1) // max(1, shard_size)
    for i in range(n_shards):
        chunk = examples[i * shard_size: (i + 1) * shard_size]
        shard_path = out_dir / f"{prefix}_{i:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for ex in chunk:
                f.write(json.dumps(ex.to_dict(), default=str, ensure_ascii=False))
                f.write("\n")
        shard_files.append(shard_path)
    return shard_files


def read_jsonl_shards(shard_dir: str | Path,
                      *, prefix: str = "tovah_corpus"
                      ) -> Iterator[TrainingExample]:
    """Iterate over examples from JSONL shards in `shard_dir`."""
    shard_dir = Path(shard_dir)
    for shard_path in sorted(shard_dir.glob(f"{prefix}_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield TrainingExample(**d)
