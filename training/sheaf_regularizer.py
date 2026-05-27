"""v14.3.5 differentiable sheaf-obstruction regularizers.

The runtime SheafObserver remains a module-level diagnostic.  These losses make
local-to-global obstruction an optional training signal over token trajectories:
neighboring hidden/support states should glue unless the metadata says the
sequence is genuinely K/G-heavy.  The loss is deliberately local, cheap, and
fully differentiable.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def sequence_sheaf_obstruction_loss(
    T: torch.Tensor,
    Fv: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    bilateral_t: Optional[torch.Tensor] = None,
    bilateral_f: Optional[torch.Tensor] = None,
    allow_glut_slack: float = 0.50,
    allow_gap_slack: float = 0.50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a differentiable local gluing obstruction over a sequence.

    ``T`` and ``Fv`` may be compact B×L×1 supports or wider hidden/profile
    tensors.  The loss measures squared changes in K/G/A/B profile along token
    edges.  Metadata K/G increases the tolerated local obstruction, so real
    paradoxes are preserved rather than classicalized.
    """
    if T.shape[:2] != Fv.shape[:2]:
        raise ValueError("T and Fv must share batch/sequence dimensions")
    if T.shape[1] < 2:
        z = (T.sum() + Fv.sum()) * 0.0
        return z, {"sheaf_edges": 0.0, "sheaf_obstruction": 0.0, "sheaf_slack": 0.0}
    k = torch.minimum(T, Fv).mean(dim=-1)
    g = (1.0 - torch.maximum(T, Fv)).mean(dim=-1)
    a = F.relu(T - Fv).mean(dim=-1)
    b = F.relu(Fv - T).mean(dim=-1)
    prof = torch.stack([a, b, k, g], dim=-1)
    edge = (prof[:, 1:, :] - prof[:, :-1, :]).pow(2).sum(dim=-1)
    if attention_mask is not None:
        m = attention_mask.to(device=T.device, dtype=T.dtype)
        if m.shape[1] != T.shape[1]:
            L = min(m.shape[1], T.shape[1])
            m = m[:, :L]
            edge = edge[:, : max(0, L - 1)]
        edge_m = (m[:, 1:] * m[:, :-1]).to(edge.dtype)
    else:
        edge_m = torch.ones_like(edge)
    slack = torch.zeros(T.shape[0], device=T.device, dtype=T.dtype)
    if bilateral_t is not None and bilateral_f is not None:
        bt = bilateral_t.to(device=T.device, dtype=T.dtype).clamp(0, 1)
        bf = bilateral_f.to(device=T.device, dtype=T.dtype).clamp(0, 1)
        meta_k = torch.minimum(bt, bf)
        meta_g = torch.minimum(1.0 - bt, 1.0 - bf)
        slack = allow_glut_slack * meta_k + allow_gap_slack * meta_g
    slack_edges = slack.unsqueeze(-1).expand_as(edge)
    penalized = F.relu(edge - slack_edges).pow(2)
    denom = edge_m.sum().clamp_min(1.0)
    loss = (penalized * edge_m).sum() / denom
    return loss, {
        "sheaf_edges": float(denom.detach().item()),
        "sheaf_obstruction": float((edge * edge_m).sum().detach().item() / max(1.0, float(denom.detach().item()))),
        "sheaf_slack": float(slack.detach().mean().item()) if slack.numel() else 0.0,
    }


__all__ = ["sequence_sheaf_obstruction_loss"]
