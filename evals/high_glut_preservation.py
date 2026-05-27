"""Eval that Lane B semantic matching rewards preserved contradiction."""
from __future__ import annotations

import torch

from tovah_v14.evals.common import emit, result
from tovah_v14.neural.training import _compute_lane_b_regularizer


def run() -> dict:
    gate = torch.tensor([[0.0, 5.0, 0.0, -5.0]])
    bt = torch.tensor([0.86])
    bf = torch.tensor([0.84])
    preserved_T = torch.full((1, 4, 8), 0.86)
    preserved_F = torch.full((1, 4, 8), 0.84)
    collapsed_T = torch.full((1, 4, 8), 0.86)
    collapsed_F = torch.full((1, 4, 8), 0.08)
    p_loss, p_stats = _compute_lane_b_regularizer(preserved_T, preserved_F, gate, bt, bf)
    c_loss, c_stats = _compute_lane_b_regularizer(collapsed_T, collapsed_F, gate, bt, bf)
    p = float(p_loss.mean().item())
    c = float(c_loss.mean().item())
    return result(
        "high_glut_preservation",
        p < c,
        preserved_loss=p,
        collapsed_loss=c,
        preserved_stats=p_stats,
        collapsed_stats=c_stats,
    )


if __name__ == "__main__":
    emit(run())
