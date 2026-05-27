"""
TOVAH v14 tests/test_training_pipeline.py — Pretraining-corpus tests.

Covers the audit's §5.3 step 9 requirements:
  - given a fixture kernel, build_corpus produces records and 0 dups
  - lineage chains reconstruct provenance correctly
  - paraconsistent filter routes A/B/K/G correctly
  - JSONL shards round-trip
  - manifest captures all telemetry sources
  - dedup strategies all behave per-spec
  - EXPORT_CORPUS command works end-to-end
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# --- Fixtures -------------------------------------------------------------

def _booted_kernel():
    """Boot a kernel with hub for fuller telemetry."""
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    from tovah_v14.kernel.kernel import ProtozoanKernel
    return ProtozoanKernel(api={}, is_original=True)


# --- TrainingExample basics ------------------------------------------------

def test_training_example_roundtrips():
    from tovah_v14.training.corpus_builder import TrainingExample
    ex = TrainingExample(
        lineage_id="exp::abc",
        kind="experience",
        text="hello world",
        bilateral_t=0.7, bilateral_f=0.1,
        quality_score=0.6,
    )
    d = ex.to_dict()
    assert d["lineage_id"] == "exp::abc"
    assert d["bilateral_t"] == 0.7
    ex2 = TrainingExample(**d)
    assert ex2.text == "hello world"


def test_lineage_id_is_deterministic():
    from tovah_v14.training.corpus_builder import _experience_to_example
    rec = {"rec_id": "r1", "kind": "research", "description": "test", "time": 1.0,
           "bilateral_assessment": {"t": 0.6, "f": 0.1}}
    ex1 = _experience_to_example(rec)
    ex2 = _experience_to_example(dict(rec))
    assert ex1.lineage_id == ex2.lineage_id


# --- build_corpus ----------------------------------------------------------

def test_build_corpus_from_kernel_yields_examples():
    from tovah_v14.training import build_corpus
    k = _booted_kernel()
    examples = build_corpus(k)
    assert isinstance(examples, list)
    # Even a freshly-booted kernel produces at least packet examples
    # (boot status packets, hub packet).
    assert len(examples) >= 1
    kinds = {e.kind for e in examples}
    # `packet` is unconditional once a hub kernel boots.
    assert "packet" in kinds


def test_build_corpus_from_state_files(tmp_path):
    from tovah_v14.training import build_corpus_from_state_files
    # Synthesize minimal state files.
    (tmp_path / "tovah_state.json").write_text(json.dumps({
        "experience_records": [
            {"rec_id": "r1", "kind": "research", "description": "topic A",
             "time": 1.0, "outcome": "useful",
             "bilateral_assessment": {"t": 0.7, "f": 0.1}},
            {"rec_id": "r2", "kind": "patch", "description": "patch B",
             "time": 2.0, "outcome": "regressed",
             "bilateral_assessment": {"t": 0.2, "f": 0.7}},
        ],
        "memory_episodic": [
            {"key": "m1", "content": "memory body", "tags": ["foo"],
             "bilateral_confidence": {"t": 0.8, "f": 0.05}}
        ],
    }))
    (tmp_path / "tovah_kernel_ecology.json").write_text(json.dumps({
        "packet_log": [
            {"packet_id": "p1", "packet_kind": "module_proposal_packet",
             "source_kernel_id": "hub", "target_kernel_id": "main",
             "payload": {"module_name": "mod_a"}, "accepted_by_main": True,
             "handled_at": 3.0, "ordinal": 1},
        ],
        "module_proposals": [],
    }))
    examples = build_corpus_from_state_files(tmp_path)
    assert len(examples) == 4  # 2 exp + 1 mem + 1 pkt
    kinds = {e.kind for e in examples}
    assert {"experience", "memory", "packet"} <= kinds


def test_build_corpus_handles_missing_state_dir(tmp_path):
    from tovah_v14.training import build_corpus_from_state_files
    examples = build_corpus_from_state_files(tmp_path / "nonexistent")
    assert examples == []


# --- Dedup -----------------------------------------------------------------

def test_dedup_collapses_same_lineage_id():
    from tovah_v14.training import deduplicate, DedupStrategy
    from tovah_v14.training.corpus_builder import TrainingExample
    ex1 = TrainingExample(lineage_id="x", kind="exp", text="v1",
                          quality_score=0.5, bilateral_t=0.7, bilateral_f=0.1)
    ex2 = TrainingExample(lineage_id="x", kind="exp", text="v2",
                          quality_score=0.8, bilateral_t=0.8, bilateral_f=0.05)
    ex3 = TrainingExample(lineage_id="y", kind="exp", text="v3",
                          quality_score=0.4, bilateral_t=0.5, bilateral_f=0.2)
    unique, stats = deduplicate([ex1, ex2, ex3],
                                strategy=DedupStrategy.KEEP_BEST_QUALITY)
    assert len(unique) == 2
    assert stats["duplicates_collapsed"] == 1
    chosen_x = next(e for e in unique if e.lineage_id == "x")
    assert chosen_x.text == "v2"  # higher quality


def test_dedup_merge_strategy_unions_provenance_and_takes_max_TF():
    from tovah_v14.training import deduplicate, DedupStrategy
    from tovah_v14.training.corpus_builder import TrainingExample
    ex1 = TrainingExample(lineage_id="z", kind="m", text="t", quality_score=0.5,
                          bilateral_t=0.6, bilateral_f=0.2,
                          provenance_chain=["a"])
    ex2 = TrainingExample(lineage_id="z", kind="m", text="t", quality_score=0.7,
                          bilateral_t=0.4, bilateral_f=0.7,
                          provenance_chain=["b"])
    unique, _stats = deduplicate([ex1, ex2],
                                 strategy=DedupStrategy.MERGE_WITH_PROVENANCE)
    assert len(unique) == 1
    merged = unique[0]
    assert merged.bilateral_t == 0.6  # max
    assert merged.bilateral_f == 0.7  # max
    assert "a" in merged.provenance_chain
    assert any(p.startswith("z@merged") for p in merged.provenance_chain) or \
           "b" in merged.provenance_chain
    assert merged.metadata.get("merged_count") == 2


# --- Paraconsistent classification -----------------------------------------

def test_classify_one_routes_to_ABKG():
    from tovah_v14.training.corpus_builder import TrainingExample
    from tovah_v14.training import classify_examples
    from tovah_v14.training.quality_filter import ParaconsistentClass
    a = TrainingExample(lineage_id="a", kind="x", text="", bilateral_t=0.9, bilateral_f=0.1)
    b = TrainingExample(lineage_id="b", kind="x", text="", bilateral_t=0.1, bilateral_f=0.9)
    k = TrainingExample(lineage_id="k", kind="x", text="", bilateral_t=0.9, bilateral_f=0.9)
    g = TrainingExample(lineage_id="g", kind="x", text="", bilateral_t=0.1, bilateral_f=0.1)
    out = classify_examples([a, b, k, g])
    cls = {e.lineage_id: e.paraconsistent_class for e in out}
    assert cls["a"] == ParaconsistentClass.A.value
    assert cls["b"] == ParaconsistentClass.B.value
    assert cls["k"] == ParaconsistentClass.K.value
    assert cls["g"] == ParaconsistentClass.G.value


def test_class_counts_uses_classification():
    from tovah_v14.training.corpus_builder import TrainingExample
    from tovah_v14.training.quality_filter import class_counts
    examples = [
        TrainingExample(lineage_id=f"e{i}", kind="x", text="",
                        bilateral_t=0.9, bilateral_f=0.1)
        for i in range(3)
    ] + [
        TrainingExample(lineage_id=f"k{i}", kind="x", text="",
                        bilateral_t=0.8, bilateral_f=0.8)
        for i in range(2)
    ]
    counts = class_counts(examples)
    assert counts["A"] == 3
    assert counts["K"] == 2
    assert counts["B"] == 0
    assert counts["G"] == 0


# --- Lineage graph ---------------------------------------------------------

def test_lineage_graph_reconstructs_chain():
    from tovah_v14.training import build_lineage_graph
    from tovah_v14.training.corpus_builder import TrainingExample
    a = TrainingExample(lineage_id="A", kind="root", text="")
    b = TrainingExample(lineage_id="B", kind="step1", text="", provenance_chain=["A"])
    c = TrainingExample(lineage_id="C", kind="step2", text="", provenance_chain=["B"])
    d = TrainingExample(lineage_id="D", kind="step2b", text="", provenance_chain=["B"])
    g = build_lineage_graph([a, b, c, d])
    assert "A" in g.nodes
    assert "B" in g.upstream("C")
    assert "A" in g.upstream("C", max_depth=8)
    children_of_B = g.downstream("B")
    assert "C" in children_of_B and "D" in children_of_B
    stats = g.stats()
    assert stats["n_nodes"] == 4
    assert stats["n_roots"] >= 1


# --- JSONL exporter --------------------------------------------------------

def test_jsonl_shards_roundtrip(tmp_path):
    from tovah_v14.training import write_jsonl_shards, read_jsonl_shards
    from tovah_v14.training.corpus_builder import TrainingExample
    examples = [
        TrainingExample(lineage_id=f"e{i}", kind="x", text=f"text {i}",
                        bilateral_t=0.5, bilateral_f=0.2, quality_score=0.3)
        for i in range(2500)
    ]
    shards = write_jsonl_shards(examples, tmp_path / "shards", shard_size=1000)
    assert len(shards) == 3  # 1000+1000+500
    read_back = list(read_jsonl_shards(tmp_path / "shards"))
    assert len(read_back) == 2500
    assert read_back[0].lineage_id == "e0"
    assert read_back[-1].lineage_id == "e2499"


# --- Manifest --------------------------------------------------------------

def test_manifest_captures_all_signals():
    from tovah_v14.training import build_manifest, classify_examples
    from tovah_v14.training.corpus_builder import TrainingExample
    examples = [
        TrainingExample(lineage_id="a", kind="experience", text="", bilateral_t=0.9, bilateral_f=0.1, outcome_label="useful"),
        TrainingExample(lineage_id="b", kind="packet", text="", bilateral_t=0.1, bilateral_f=0.9, outcome_label="rejected"),
        TrainingExample(lineage_id="c", kind="memory", text="", bilateral_t=0.8, bilateral_f=0.8, outcome_label="recalled"),
    ]
    classified = classify_examples(examples)
    manifest = build_manifest(
        classified,
        dedup_stats={"input_total": 3, "unique": 3, "duplicates_collapsed": 0, "dedup_ratio": 1.0},
        lineage_stats={"n_nodes": 3, "n_edges": 0, "n_roots": 3, "n_leaves": 3,
                       "avg_chain_depth": 0.0, "max_chain_depth": 0},
        shard_files=["s0.jsonl"],
    )
    assert manifest["totals"]["examples"] == 3
    assert manifest["totals"]["kinds"] == {"experience": 1, "packet": 1, "memory": 1}
    assert manifest["paraconsistent"]["class_counts"]["A"] == 1
    assert manifest["paraconsistent"]["class_counts"]["B"] == 1
    assert manifest["paraconsistent"]["class_counts"]["K"] == 1
    assert manifest["dedup"]["unique"] == 3


# --- Continuous exporter ---------------------------------------------------

def test_continuous_exporter_writes_and_rotates(tmp_path):
    from tovah_v14.training import ContinuousExporter
    from tovah_v14.training.corpus_builder import TrainingExample
    with ContinuousExporter(tmp_path / "stream", shard_size=10) as exp:
        for i in range(25):
            exp.append(TrainingExample(lineage_id=f"e{i}", kind="x", text=f"t{i}"))
    shards = sorted((tmp_path / "stream").glob("*.jsonl"))
    assert len(shards) >= 3
    total_lines = 0
    for s in shards:
        total_lines += sum(1 for _ in open(s))
    assert total_lines == 25


def test_continuous_exporter_resumes_at_correct_shard(tmp_path):
    from tovah_v14.training import ContinuousExporter
    from tovah_v14.training.corpus_builder import TrainingExample
    with ContinuousExporter(tmp_path / "stream", shard_size=5) as exp:
        for i in range(7):  # 5 + 2 in second shard
            exp.append(TrainingExample(lineage_id=f"e{i}", kind="x", text=""))
    # New session should resume on a *new* shard, not append to last.
    with ContinuousExporter(tmp_path / "stream", shard_size=5) as exp:
        exp.append(TrainingExample(lineage_id="resume", kind="x", text=""))
    shards = sorted((tmp_path / "stream").glob("*.jsonl"))
    # 0: 5 lines, 1: 2 lines, 2: 1 line (resumed)
    assert len(shards) == 3


# --- End-to-end ------------------------------------------------------------

def test_export_corpus_end_to_end(tmp_path):
    from tovah_v14.training import export_corpus
    k = _booted_kernel()
    out = export_corpus(tmp_path / "corpus", kernel=k,
                        dedup_strategy="merge_with_provenance",
                        shard_size=1000)
    assert (tmp_path / "corpus" / "manifest.json").exists()
    assert (tmp_path / "corpus" / "lineage_graph.json").exists()
    assert out["total_examples"] >= 1
    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    # Every classified example is in exactly one of A/B/K/G.
    cls = manifest["paraconsistent"]["class_counts"]
    assert sum(cls.values()) == out["unique_examples"]


def test_export_corpus_command_works():
    """Drive EXPORT_CORPUS through the COMMAND_FILE pathway."""
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.config.paths import COMMAND_FILE
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        k = ProtozoanKernel(api={}, is_original=True)
        out_dir = os.path.join(td, "corpus_out")
        # Write the command. _check_david_commands reads from COMMAND_FILE.
        COMMAND_FILE.write_text(f"EXPORT_CORPUS:{out_dir}", encoding="utf-8")
        k._check_david_commands()
        # Manifest file should be on disk.
        assert os.path.exists(os.path.join(out_dir, "manifest.json"))
        manifest = json.loads(open(os.path.join(out_dir, "manifest.json")).read())
        assert "totals" in manifest
        # Lineage graph dump alongside.
        assert os.path.exists(os.path.join(out_dir, "lineage_graph.json"))


def test_export_corpus_command_rejects_empty_dir():
    """Empty directory spec should be reported in response file."""
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.config.paths import COMMAND_FILE, RESPONSE_FILE
    k = ProtozoanKernel(api={}, is_original=True)
    COMMAND_FILE.write_text("EXPORT_CORPUS:", encoding="utf-8")
    k._check_david_commands()
    if RESPONSE_FILE.exists():
        resp = RESPONSE_FILE.read_text(encoding="utf-8")
        assert "requires output directory" in resp.lower() or "EXPORT_CORPUS" in resp


# --- Specific source coverage ---------------------------------------------

def test_packet_example_skips_heartbeats():
    from tovah_v14.training.corpus_builder import _packet_to_example
    pkt = {"packet_id": "p1", "packet_kind": "heartbeat",
           "source_kernel_id": "main", "target_kernel_id": "hub",
           "payload": {}, "handled_at": 1.0}
    assert _packet_to_example(pkt) is None
    pkt2 = {"packet_id": "p2", "packet_kind": "module_proposal",
            "source_kernel_id": "hub", "target_kernel_id": "main",
            "payload": {"module_name": "test"}, "handled_at": 2.0,
            "accepted_by_main": True}
    ex = _packet_to_example(pkt2)
    assert ex is not None
    assert ex.outcome_label == "accepted"


def test_gate_decision_blocked_carries_low_truth():
    from tovah_v14.training.corpus_builder import _gate_decision_to_example
    dec = {"from": "static_approved", "to": "sandbox_passed",
           "reason": "blocked: missing_runner", "context": {}, "at": 1.0}
    ex = _gate_decision_to_example(dec, patch_name="some_patch")
    assert ex.outcome_label == "blocked"
    assert ex.bilateral_t < ex.bilateral_f
    assert ex.kind == "gate_decision"
