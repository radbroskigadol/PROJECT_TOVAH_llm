"""Eval that Lane C semantic matching rewards preserved underdetermination."""
from __future__ import annotations

import torch

from tovah_v14.evals.common import emit, result
from tovah_v14.neural.training import _compute_lane_c_regularizer


def run() -> dict:
    gate = torch.tensor([[0.0, 0.0, 5.0, -5.0]])
    bt = torch.tensor([0.12])
    bf = torch.tensor([0.10])
    gap_T = torch.full((1, 4, 8), 0.12)
    gap_F = torch.full((1, 4, 8), 0.10)
    collapsed_T = torch.full((1, 4, 8), 0.90)
    collapsed_F = torch.full((1, 4, 8), 0.10)
    g_loss, g_stats = _compute_lane_c_regularizer(gap_T, gap_F, gate, bt, bf)
    c_loss, c_stats = _compute_lane_c_regularizer(collapsed_T, collapsed_F, gate, bt, bf)
    g = float(g_loss.mean().item())
    c = float(c_loss.mean().item())
    return result(
        "gap_tolerance",
        g < c,
        gap_loss=g,
        collapsed_loss=c,
        gap_stats=g_stats,
        collapsed_stats=c_stats,
    )


if __name__ == "__main__":
    emit(run())
