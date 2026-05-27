"""Regression tests for v14.2.6 scale-handoff package."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tovah_v14.evals import run_all
from tovah_v14.neural.adamw import AdamWWrapper
from tovah_v14.neural.checkpointing import checkpoint_manifest_path, save_training_checkpoint
from tovah_v14.neural.scaling import ScalableBilateralCore
from tovah_v14.training.metrics import ScaleMetricLogger
from tovah_v14.training.scale_ladder import SCALE_LADDER, memory_plan


REQUESTED_DOCS = [
    "SCALE_READINESS.md",
    "SCALING_LADDER.md",
    "FSDP_RUNBOOK.md",
    "TOKENIZER_STRATEGY.md",
    "DATA_PIPELINE.md",
    "EVALS.md",
    "SECURITY.md",
    "SAFE_MODE.md",
    "THREAT_MODEL.md",
    "SCALE_HANDOFF.md",
]

REQUESTED_SCRIPTS = [
    "scripts/create_tiny_corpus.py",
    "scripts/train_debug_cpu.sh",
    "scripts/train_fsdp_single_node.sh",
    "scripts/train_fsdp_multi_node.sh",
    "scripts/eval_smoke.sh",
]

REQUESTED_CONFIGS = [
    "configs/debug_5m.yaml",
    "configs/100m.yaml",
    "configs/1b_fsdp.yaml",
    "configs/13b_fsdp_reference.yaml",
]


def test_scale_handoff_docs_scripts_configs_present():
    root = Path(__file__).resolve().parents[1]
    for rel in REQUESTED_DOCS + REQUESTED_SCRIPTS + REQUESTED_CONFIGS:
        path = root / rel
        assert path.exists(), rel
        assert path.stat().st_size > 100, rel


def test_scale_ladder_has_13b_reference_memory_plan():
    assert any(stage.name == "frontier_13b_reference" for stage in SCALE_LADDER)
    plan = memory_plan(vocab_size=1024, dtype="bf16")
    rows = {row["name"]: row for row in plan["stages"]}
    assert "frontier_13b_reference" in rows
    est = rows["frontier_13b_reference"]["memory_estimate"]
    assert est["profile"] == "frontier_13b"
    assert est["fsdp_sharded"] is True
    assert est["total_gb_per_rank_full_vocab_aux_est"] > est["total_gb_per_rank_hidden_aux_est"]


def test_buyer_evals_run_all_passes():
    payload = run_all.run()
    assert payload["passed"] is True
    assert len(payload["results"]) >= 6
    assert {r["eval"] for r in payload["results"]} >= {
        "smoke_language_modeling",
        "semantic_consistency",
        "high_glut_preservation",
        "gap_tolerance",
        "patch_certification_eval",
        "memory_conflict_eval",
    }


def test_scale_metric_logger_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = ScaleMetricLogger(path)
    rec = logger.log(step=1, loss=2.5, phase="Active Learning", mean_K_pred=0.1)
    assert rec["step"] == 1
    line = path.read_text().strip()
    loaded = json.loads(line)
    assert loaded["loss"] == 2.5
    assert loaded["phase"] == "Active Learning"


def test_checkpoint_manifest_written(tmp_path):
    model = ScalableBilateralCore(vocab_size=16, d_model=16, n_heads=2, n_kv_heads=1, n_blocks=1, max_len=8, bilateral_mode="shared", gradient_checkpointing=False)
    opt = AdamWWrapper(model.parameters(), base_lr=1e-4)
    ckpt = tmp_path / "ckpt.pt"
    written = save_training_checkpoint(ckpt, model, opt, step=3, epoch=1, metadata={"scale_handoff": True})
    assert written == ckpt
    manifest = checkpoint_manifest_path(ckpt)
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["version"] == "14.2.6"
    assert data["step"] == 3
    assert data["metadata"]["scale_handoff"] is True
