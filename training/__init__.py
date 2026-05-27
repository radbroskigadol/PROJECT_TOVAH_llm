"""
TOVAH v14 training/ — Pretraining-corpus assembly pipeline.

This package converts TOVAH's accumulated operational telemetry into a
deduplicated, lineage-aware, paraconsistently-classified JSONL corpus
suitable for LLM pretraining.

The architecture mirrors the rest of the codebase:
- corpus_builder.py    — source-by-source extraction into TrainingExample
- dedup.py             — collapse near-duplicates across waves and kernels
- lineage_graph.py     — DAG of provenance (upstream/downstream chains)
- quality_filter.py    — A/B/K/G routing via paraconsistent invariants
- exporters/           — JSONL and Parquet writers with manifest
- continuous_export.py — kernel hook for streaming export
- manifest.py          — per-export run manifest with statistics
- dataset.py           — streaming IterableDataset for DataLoader (v14.1.2)
- tokenizer.py         — byte / BPE tokenizer abstraction (v14.1.2)
- eval.py              — held-out perplexity + accuracy + calibration (v14.1.2)
- pretrain.py          — production training loop with AMP/sched/eval (v14.1.2)

Public entry points:
    build_corpus(kernel)         -> List[TrainingExample]
    export_corpus(out_dir, kernel, since_cycle=0)
    pretrain(shard_dir, ...)     -> Dict (training summary)
"""
from __future__ import annotations

from tovah_v14.training.corpus_builder import (
    TrainingExample,
    build_corpus,
    build_corpus_from_state_files,
    strip_envelope,
)
from tovah_v14.training.dedup import deduplicate, DedupStrategy
from tovah_v14.training.quality_filter import classify_examples, ParaconsistentClass
from tovah_v14.training.lineage_graph import LineageGraph, build_lineage_graph
from tovah_v14.training.manifest import build_manifest, write_manifest
from tovah_v14.training.exporters.jsonl import write_jsonl_shards, read_jsonl_shards
from tovah_v14.training.continuous_export import ContinuousExporter
from tovah_v14.training.pretrain import pretrain
from tovah_v14.training.tokenizer import (
    ByteTokenizer, BPETokenizer, load_tokenizer, train_bpe,
)
from tovah_v14.training.dataset import CorpusShardDataset, build_collate_fn
# Do not import ``training.eval`` eagerly here.  Running
# ``python -m tovah_v14.training.eval`` first imports this package, and an
# eager import of the same module triggers runpy's "found in sys.modules"
# warning.  v14.3.2a keeps the public names via lazy __getattr__ below.
_EVAL_EXPORTS = {
    "run_full_eval",
    "held_out_perplexity",
    "token_top1_accuracy",
    "gen_sample",
    "detect_divergence",
    "bilateral_calibration",
    "split_train_val",
}

__all__ = [
    # v14.1.0 / v14.1.1
    "TrainingExample",
    "build_corpus",
    "build_corpus_from_state_files",
    "deduplicate",
    "DedupStrategy",
    "classify_examples",
    "ParaconsistentClass",
    "LineageGraph",
    "build_lineage_graph",
    "build_manifest",
    "write_manifest",
    "write_jsonl_shards",
    "read_jsonl_shards",
    "ContinuousExporter",
    "export_corpus",
    "pretrain",
    # v14.1.2 audit fixes
    "strip_envelope",
    "ByteTokenizer", "BPETokenizer", "load_tokenizer", "train_bpe",
    "CorpusShardDataset", "build_collate_fn",
    "run_full_eval", "held_out_perplexity", "token_top1_accuracy",
    "gen_sample", "detect_divergence", "bilateral_calibration",
    "split_train_val",
]


def export_corpus(out_dir, kernel=None, *, since_cycle=0,
                  state_dir=None, dedup_strategy="merge_with_provenance",
                  shard_size=1000):
    """One-shot end-to-end export.

    If `kernel` is given, builds from the live kernel's in-memory state.
    If `state_dir` is given (and kernel is None), builds from on-disk
    state JSON files in that directory.
    """
    from pathlib import Path
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if kernel is not None:
        examples = build_corpus(kernel, since_cycle=since_cycle)
    elif state_dir is not None:
        examples = build_corpus_from_state_files(state_dir, since_cycle=since_cycle)
    else:
        raise ValueError("export_corpus needs either `kernel` or `state_dir`")

    # Deduplicate.
    examples_unique, dedup_stats = deduplicate(
        examples, strategy=DedupStrategy(dedup_strategy))

    # Classify (A/B/K/G via bilateral assessments).
    classified = classify_examples(examples_unique)

    # Lineage graph (saved alongside as JSON).
    lineage = build_lineage_graph(examples_unique)

    # Write shards.
    shard_files = write_jsonl_shards(classified, out_path / "shards",
                                     shard_size=shard_size)

    # Manifest.
    mani = build_manifest(
        classified,
        dedup_stats=dedup_stats,
        lineage_stats=lineage.stats(),
        shard_files=[str(f) for f in shard_files],
    )
    write_manifest(mani, out_path / "manifest.json")

    # Lineage graph dump.
    import json
    with open(out_path / "lineage_graph.json", "w", encoding="utf-8") as f:
        json.dump(lineage.to_dict(), f, indent=2, default=str)

    return {
        "out_dir": str(out_path),
        "total_examples": len(examples),
        "unique_examples": len(examples_unique),
        "shards": [str(f) for f in shard_files],
        "manifest": mani,
    }


def __getattr__(name):
    if name in _EVAL_EXPORTS:
        from tovah_v14.training import eval as _eval_mod
        return getattr(_eval_mod, name)
    raise AttributeError(name)
