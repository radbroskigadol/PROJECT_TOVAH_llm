"""v14.3.5 bilateral state-space prototype.

A lightweight recurrent/SSM block for long-context experiments.  It is not the
default trunk, but it provides the mathematically aligned T/F state propagation
path recommended for limited-compute scaling.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BilateralSSMBlock(nn.Module):
    """Linear-time T/F state propagation with controlled cross-mixing."""
    def __init__(self, d_model: int, state_dim: int | None = None, cross_mix: float = 0.05):
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim or d_model)
        self.cross_mix = float(cross_mix)
        self.in_T = nn.Linear(d_model, self.state_dim, bias=False)
        self.in_F = nn.Linear(d_model, self.state_dim, bias=False)
        self.out_T = nn.Linear(self.state_dim, d_model, bias=False)
        self.out_F = nn.Linear(self.state_dim, d_model, bias=False)
        self.log_decay_T = nn.Parameter(torch.zeros(self.state_dim))
        self.log_decay_F = nn.Parameter(torch.zeros(self.state_dim))

    def forward(self, T: torch.Tensor, Fv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = T.shape
        decay_T = torch.sigmoid(self.log_decay_T).view(1, -1)
        decay_F = torch.sigmoid(self.log_decay_F).view(1, -1)
        sT = torch.zeros(B, self.state_dim, device=T.device, dtype=T.dtype)
        sF = torch.zeros(B, self.state_dim, device=Fv.device, dtype=Fv.dtype)
        outs_T, outs_F = [], []
        uT = self.in_T(T)
        uF = self.in_F(Fv)
        for i in range(L):
            sT = decay_T * sT + uT[:, i, :] + self.cross_mix * sF
            sF = decay_F * sF + uF[:, i, :] + self.cross_mix * sT
            outs_T.append(self.out_T(sT))
            outs_F.append(self.out_F(sF))
        return torch.stack(outs_T, dim=1), torch.stack(outs_F, dim=1)


__all__ = ["BilateralSSMBlock"]
