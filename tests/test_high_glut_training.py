"""Regression tests for v14.2.3 high-glut gradient-flow control."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from tovah_v14.neural.adamw import AdamWWrapper
from tovah_v14.neural.optimizer import ShadowOptimizer
from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.neural.training import (
    lane_routing_loss,
    lane_semantic_matching_loss,
    metadata_weighted_semantic_loss,
    _compute_lane_b_regularizer,
    _compute_lane_c_regularizer,
    phase_from_semantic_state,
    train_shadow_step,
)


def _tiny_model() -> ShadowTokenCore:
    return ShadowTokenCore(
        vocab_size=256,
        d_model=32,
        d_hidden=64,
        n_heads=2,
        n_blocks=1,
        max_len=32,
    )


def test_phase_uses_mean_glut_and_metadata_not_shape_dependent_sum():
    # Low predicted K alone is not enough for collapse.
    T = torch.full((2, 3, 5), 0.80)
    Fv = torch.full((2, 3, 5), 0.10)
    assert phase_from_semantic_state(T, Fv) != "Collapse-Resistant Paradox"

    # A true K-heavy corpus batch should enter the collapse-resistant phase even
    # if the tensor is small and regardless of h_lambda.
    bt = torch.tensor([0.90, 0.85])
    bf = torch.tensor([0.88, 0.92])
    assert phase_from_semantic_state(T, Fv, bilateral_t=bt, bilateral_f=bf) == "Collapse-Resistant Paradox"


def test_phase_uses_mean_predicted_glut():
    T = torch.full((1, 2, 8), 0.92)
    Fv = torch.full((1, 2, 8), 0.90)
    assert phase_from_semantic_state(T, Fv) == "Collapse-Resistant Paradox"


def test_lane_routing_sends_k_to_lane_b_and_gap_to_lane_c():
    gate = torch.zeros(2, 4, requires_grad=True)
    bt = torch.tensor([0.90, 0.10])
    bf = torch.tensor([0.88, 0.12])
    loss = lane_routing_loss(gate, bt, bf)
    loss.backward()
    # Cross-entropy gradient is p-target. Negative gradient means increasing
    # that logit lowers the loss.
    assert gate.grad[0, 1] < 0  # K-heavy sample -> lane B
    assert gate.grad[1, 2] < 0  # G-heavy sample -> lane C


def test_lane_routing_sends_classical_to_lane_a():
    gate = torch.zeros(2, 4, requires_grad=True)
    bt = torch.tensor([0.90, 0.10])
    bf = torch.tensor([0.10, 0.90])
    loss = lane_routing_loss(gate, bt, bf)
    loss.backward()
    assert gate.grad[0, 0] < 0
    assert gate.grad[1, 0] < 0


def test_metadata_semantic_loss_relaxes_true_glut_budget():
    # Same high-overlap predicted state; K metadata should be penalized less
    # than classical metadata because the contradiction is expected evidence.
    T = torch.full((1, 4, 16), 0.80, requires_grad=True)
    Fv = torch.full((1, 4, 16), 0.78, requires_grad=True)
    k_loss, k_stats = metadata_weighted_semantic_loss(
        T, Fv, torch.tensor([0.90]), torch.tensor([0.88])
    )
    c_loss, c_stats = metadata_weighted_semantic_loss(
        T, Fv, torch.tensor([0.90]), torch.tensor([0.10])
    )
    assert k_loss.item() < c_loss.item()
    assert k_stats["metadata_k_mean"] > c_stats["metadata_k_mean"]


def test_train_shadow_step_accepts_metadata_records_and_enters_collapse_phase():
    model = _tiny_model()
    opt = ShadowOptimizer(model.parameters(), base_lr=1e-3)
    corpus = [
        {"text": "contradictory evidence one", "bilateral_t": 0.92, "bilateral_f": 0.90},
        {"text": "contradictory evidence two", "bilateral_t": 0.88, "bilateral_f": 0.91},
    ]
    loss, phase = train_shadow_step(model, opt, corpus)
    assert isinstance(loss, float)
    assert phase == "Collapse-Resistant Paradox"
    assert opt.last_stats["lr"] == pytest.approx(4e-4)


def test_adamw_honors_collapse_phase_with_lower_lr_and_tighter_clip():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = AdamWWrapper([p], base_lr=1e-3)
    loss = (p ** 2).sum()
    stats = opt.step(loss, phase="Collapse-Resistant Paradox")
    assert stats["lr"] == pytest.approx(4e-4)
    assert stats["phase_multiplier"] == pytest.approx(0.4)
    assert stats["grad_clip_norm"] == pytest.approx(0.5)


def test_pretrain_one_batch_consumes_metadata_and_sets_collapse_phase():
    from tovah_v14.training.pretrain import _train_one_batch

    model = _tiny_model()
    opt = ShadowOptimizer(model.parameters(), base_lr=1e-3)
    batch = {
        "input_ids": torch.tensor([[65, 66, 67, 68], [69, 70, 71, 72]], dtype=torch.long),
        "target_ids": torch.tensor([[66, 67, 68, 69], [70, 71, 72, 73]], dtype=torch.long),
        "attention_mask": torch.ones(2, 4),
        "bilateral_t": torch.tensor([0.91, 0.89]),
        "bilateral_f": torch.tensor([0.90, 0.92]),
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
    )
    assert isinstance(loss, float)
    assert phase == "Collapse-Resistant Paradox"



def test_lane_b_regularizer_matches_predicted_k_to_metadata_k():
    gate = torch.tensor([[0.0, 5.0, 0.0, -5.0]])
    matched_T = torch.full((1, 3, 8), 0.82, requires_grad=True)
    matched_F = torch.full((1, 3, 8), 0.80, requires_grad=True)
    collapsed_T = torch.full((1, 3, 8), 0.82, requires_grad=True)
    collapsed_F = torch.full((1, 3, 8), 0.10, requires_grad=True)
    bt = torch.tensor([0.82])
    bf = torch.tensor([0.80])

    matched, matched_stats = _compute_lane_b_regularizer(matched_T, matched_F, gate, bt, bf)
    collapsed, collapsed_stats = _compute_lane_b_regularizer(collapsed_T, collapsed_F, gate, bt, bf)

    assert matched.mean().item() < collapsed.mean().item()
    assert matched_stats["lane_b_meta_k_mean"] == pytest.approx(0.80, abs=1e-6)
    assert collapsed_stats["lane_b_pred_k_mean"] < matched_stats["lane_b_pred_k_mean"]


def test_lane_b_regularizer_gradients_preserve_underpredicted_contradiction():
    gate = torch.tensor([[0.0, 5.0, 0.0, -5.0]])
    T = torch.full((1, 2, 8), 0.20, requires_grad=True)
    Fv = torch.full((1, 2, 8), 0.20, requires_grad=True)
    bt = torch.tensor([0.90])
    bf = torch.tensor([0.88])

    raw, _ = _compute_lane_b_regularizer(T, Fv, gate, bt, bf)
    loss = raw.mean()
    loss.backward()

    # K_pred=min(T,F) is too low, so gradient descent should increase both
    # channels. Equivalently, the direct gradient is negative.
    assert T.grad.mean().item() < 0
    assert Fv.grad.mean().item() < 0


def test_lane_c_regularizer_matches_predicted_g_to_metadata_g():
    gate = torch.tensor([[0.0, 0.0, 5.0, -5.0]])
    matched_T = torch.full((1, 3, 8), 0.12, requires_grad=True)
    matched_F = torch.full((1, 3, 8), 0.10, requires_grad=True)
    collapsed_T = torch.full((1, 3, 8), 0.90, requires_grad=True)
    collapsed_F = torch.full((1, 3, 8), 0.10, requires_grad=True)
    bt = torch.tensor([0.12])
    bf = torch.tensor([0.10])

    matched, matched_stats = _compute_lane_c_regularizer(matched_T, matched_F, gate, bt, bf)
    collapsed, collapsed_stats = _compute_lane_c_regularizer(collapsed_T, collapsed_F, gate, bt, bf)

    assert matched.mean().item() < collapsed.mean().item()
    assert matched_stats["lane_c_meta_g_mean"] == pytest.approx(0.88, abs=1e-6)
    assert collapsed_stats["lane_c_pred_g_mean"] < matched_stats["lane_c_pred_g_mean"]


def test_lane_c_regularizer_gradients_tolerate_underpredicted_gap():
    gate = torch.tensor([[0.0, 0.0, 5.0, -5.0]])
    T = torch.full((1, 2, 8), 0.90, requires_grad=True)
    Fv = torch.full((1, 2, 8), 0.10, requires_grad=True)
    bt = torch.tensor([0.10])
    bf = torch.tensor([0.08])

    raw, _ = _compute_lane_c_regularizer(T, Fv, gate, bt, bf)
    loss = raw.mean()
    loss.backward()

    # G_pred=1-max(T,F) is too low because T is too high. Gradient descent
    # should lower T, so the direct gradient on T is positive.
    assert T.grad.mean().item() > 0


def test_lane_semantic_matching_loss_applies_k_and_g_weights():
    gate = torch.tensor([
        [0.0, 5.0, 0.0, -5.0],
        [0.0, 0.0, 5.0, -5.0],
        [5.0, 0.0, 0.0, -5.0],
    ])
    T = torch.stack([
        torch.full((2, 8), 0.20),  # K underpredicted for row 0
        torch.full((2, 8), 0.90),  # G underpredicted for row 1
        torch.full((2, 8), 0.20),  # mismatch but classical metadata below should zero K/G weights
    ], dim=0).requires_grad_()
    Fv = torch.stack([
        torch.full((2, 8), 0.20),
        torch.full((2, 8), 0.10),
        torch.full((2, 8), 0.20),
    ], dim=0).requires_grad_()
    bt = torch.tensor([0.90, 0.10, 1.00])
    bf = torch.tensor([0.88, 0.08, 0.00])

    loss, stats = lane_semantic_matching_loss(T, Fv, gate, bt, bf)
    assert loss.item() > 0
    assert stats["lane_semantic_k_weight_mean"] > 0
    assert stats["lane_semantic_g_weight_mean"] > 0

    # With purely classical metadata, K/G multipliers eliminate the matching loss.
    classical_loss, _ = lane_semantic_matching_loss(
        T[:1], Fv[:1], gate[:1], torch.tensor([1.0]), torch.tensor([0.0])
    )
    assert classical_loss.item() == pytest.approx(0.0, abs=1e-7)
