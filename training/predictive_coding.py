"""v14.3.5 bilateral predictive-coding auxiliaries.

This is a lightweight active-inference layer over TOVAH's bilateral semantics.
It does not replace backpropagation; it supplies a local free-energy objective
that can be attached to hidden/support states, or used in toy networks before a
larger asynchronous implementation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def bilateral_free_energy(
    lower_T: torch.Tensor,
    lower_F: torch.Tensor,
    pred_T: torch.Tensor,
    pred_F: torch.Tensor,
    *,
    precision_T: float = 1.0,
    precision_F: float = 1.0,
    preserve_glut: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Local bilateral free-energy loss.

    Truth support is the feed-forward expectation; falsity support is the
    refutation/error lane.  If ``preserve_glut`` is true, high shared K mass is
    not treated as a scalar failure; it gates the error so contradictions remain
    informative instead of being collapsed.
    """
    if lower_T.shape != pred_T.shape or lower_F.shape != pred_F.shape:
        raise ValueError("lower and predicted supports must have matching shapes")
    err_T = lower_T - pred_T
    err_F = lower_F - pred_F
    if preserve_glut:
        k = torch.minimum(lower_T, lower_F).detach()
        gate = 1.0 - 0.5 * k.clamp(0, 1)
    else:
        gate = 1.0
    loss_T = float(precision_T) * (gate * err_T.pow(2)).mean()
    loss_F = float(precision_F) * (gate * err_F.pow(2)).mean()
    loss = loss_T + loss_F
    return loss, {
        "free_energy": float(loss.detach().item()),
        "prediction_error_T": float(err_T.abs().detach().mean().item()),
        "prediction_error_F": float(err_F.abs().detach().mean().item()),
    }


def top_down_support_prediction_loss(
    supports: list[tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply local predictive coding over adjacent support layers.

    ``supports`` is ordered shallow→deep.  Each deeper support pair predicts the
    immediately lower pair.  This can be used once a model exposes per-layer
    semantic heads.
    """
    if len(supports) < 2:
        z = supports[0][0].sum() * 0.0 if supports else torch.tensor(0.0)
        return z, {"pc_edges": 0.0, "free_energy": 0.0}
    losses = []
    for i in range(len(supports) - 1):
        lower_T, lower_F = supports[i]
        pred_T, pred_F = supports[i + 1]
        losses.append(bilateral_free_energy(lower_T, lower_F, pred_T, pred_F)[0])
    total = torch.stack(losses).mean()
    return total, {"pc_edges": float(len(losses)), "free_energy": float(total.detach().item())}


__all__ = ["bilateral_free_energy", "top_down_support_prediction_loss"]
