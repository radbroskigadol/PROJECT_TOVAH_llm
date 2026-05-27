"""
TOVAH v14 training/manifest.py — Per-export run manifest.

Without a manifest, no one can reproduce a pretraining run from the
exported corpus. The manifest records:
  - source counts (per `kind`)
  - dedup ratio
  - paraconsistent class distribution (A/B/K/G)
  - lineage chain depth distribution
  - shard files
  - export timestamp + corpus version
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tovah_v14.config.constants import VERSION
from tovah_v14.training.corpus_builder import TrainingExample
from tovah_v14.training.quality_filter import class_counts


CORPUS_FORMAT_VERSION = "1.0"


def build_manifest(examples: List[TrainingExample],
                   *, dedup_stats: Dict[str, Any],
                   lineage_stats: Dict[str, Any],
                   shard_files: List[str]) -> Dict[str, Any]:
    """Assemble the manifest dict."""
    kind_counter: Counter = Counter(e.kind for e in examples)
    outcome_counter: Counter = Counter(e.outcome_label for e in examples)
    cls_counts = class_counts(examples)

    quality_scores = [e.quality_score for e in examples]
    qmin = min(quality_scores) if quality_scores else 0.0
    qmax = max(quality_scores) if quality_scores else 0.0
    qmean = (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0

    return {
        "tovah_version": VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "exported_at": time.time(),
        "exported_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totals": {
            "examples": len(examples),
            "kinds": dict(kind_counter),
            "outcomes": dict(outcome_counter),
        },
        "dedup": dedup_stats,
        "paraconsistent": {
            "class_counts": cls_counts,
            "fraction_A": cls_counts.get("A", 0) / max(1, len(examples)),
            "fraction_B": cls_counts.get("B", 0) / max(1, len(examples)),
            "fraction_K": cls_counts.get("K", 0) / max(1, len(examples)),
            "fraction_G": cls_counts.get("G", 0) / max(1, len(examples)),
        },
        "quality": {"min": qmin, "max": qmax, "mean": qmean},
        "lineage": lineage_stats,
        "shards": list(shard_files),
    }


def write_manifest(manifest: Dict[str, Any], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return p
