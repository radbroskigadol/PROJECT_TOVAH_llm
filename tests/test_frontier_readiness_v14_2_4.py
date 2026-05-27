"""Regression tests for v14.2.6 frontier-readiness pass."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tovah_v14.neural.scaling import (
    make_scalable_model,
    ScalableBilateralCore,
    estimate_frontier_memory,
)
from tovah_v14.training.pretrain import _forward_for_pretrain, _train_one_batch
from tovah_v14.neural.adamw import AdamWWrapper
from tovah_v14.neural.checkpointing import save_training_checkpoint, load_training_checkpoint
from tovah_v14.neural import distributed


def test_hidden_semantic_forward_skips_full_f_logits(monkeypatch):
    model = ScalableBilateralCore(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, n_blocks=1, max_len=32, gradient_checkpointing=False, bilateral_mode="shared")

    def boom(_x):
        raise AssertionError("head_F should not be called in hidden semantic mode")

    monkeypatch.setattr(model.head_F, "forward", boom)
    x = torch.randint(0, 64, (2, 8))
    tl, fl, gl, tp, fp, mode = _forward_for_pretrain(
        model, x, frontier_semantic_mode="hidden"
    )
    assert mode == "hidden"
    assert fl is None
    assert tl.shape == (2, 8, 64)
    assert gl.shape == (2, 8, 4)
    assert tp.shape == (2, 8, 1)
    assert fp.shape == (2, 8, 1)
    assert torch.all((tp >= 0) & (tp <= 1))
    assert torch.all((fp >= 0) & (fp <= 1))


def test_train_one_batch_uses_hidden_semantic_mode_for_frontier():
    model = ScalableBilateralCore(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, n_blocks=1, max_len=32, gradient_checkpointing=False, bilateral_mode="shared")
    opt = AdamWWrapper(model.parameters(), base_lr=1e-4)
    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "target_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8),
        "bilateral_t": torch.tensor([0.90, 0.86]),
        "bilateral_f": torch.tensor([0.88, 0.89]),
    }
    loss, phase = _train_one_batch(
        model,
        opt,
        batch,
        con_budget=0.12,
        gap_budget=0.20,
        lambda_budget=0.05,
        device="cpu",
        dtype=torch.float32,
        frontier_semantic_mode="hidden",
    )
    assert isinstance(loss, float)
    assert phase == "Collapse-Resistant Paradox"
    assert opt.last_stats["phase"] == phase


def test_frontier_memory_estimator_reports_hidden_aux_savings():
    est = estimate_frontier_memory(
        "frontier_13b",
        vocab_size=50_257,
        batch_size=1,
        seq_len=1024,
        world_size=8,
        use_fsdp=True,
        dtype="bf16",
    )
    assert est["parameters"] > 1.0e10
    assert est["total_gb_per_rank_full_vocab_aux_est"] > est["total_gb_per_rank_hidden_aux_est"]
    assert est["avoidable_f_logits_gb_per_rank"] > 0
    assert est["fsdp_sharded"] is True


def test_fsdp_mixed_precision_policy_builds_or_raises_cleanly():
    # fp32/default returns None without requiring a distributed process group.
    assert distributed.fsdp_mixed_precision_policy("fp32") is None
    try:
        policy = distributed.fsdp_mixed_precision_policy("bf16")
    except RuntimeError:
        pytest.skip("torch build lacks FSDP MixedPrecision")
    assert policy is not None


def test_training_checkpoint_round_trip(tmp_path):
    model = ScalableBilateralCore(vocab_size=32, d_model=32, n_heads=2, n_kv_heads=1, n_blocks=1, max_len=16, gradient_checkpointing=False, bilateral_mode="shared")
    opt = AdamWWrapper(model.parameters(), base_lr=1e-4)
    x = torch.randint(0, 32, (1, 4))
    tl, _fl, _gl = model(x)
    opt.step(tl.mean())
    ckpt = tmp_path / "ckpt.pt"
    written = save_training_checkpoint(
        ckpt, model, opt, step=opt.t, epoch=1, metadata={"test": True}
    )
    assert written == ckpt
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["format"] == "tovah_training_checkpoint_v1"
    assert payload["version"] in {"14.2.6", "14.2.9", "14.3.4"}
    assert payload["optimizer_state"]["t"] == opt.t

    model2 = ScalableBilateralCore(vocab_size=32, d_model=32, n_heads=2, n_kv_heads=1, n_blocks=1, max_len=16, gradient_checkpointing=False, bilateral_mode="shared")
    opt2 = AdamWWrapper(model2.parameters(), base_lr=1e-4)
    loaded = load_training_checkpoint(ckpt, model2, opt2, strict=False, restore_rng=False)
    assert loaded["step"] == opt.t
    assert opt2.t == opt.t
