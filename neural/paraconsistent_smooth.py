"""v14.3.5 smooth paraconsistent t-norm/t-conorm helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_min(x: torch.Tensor, y: torch.Tensor, temperature: float = 32.0) -> torch.Tensor:
    t = max(1e-6, float(temperature))
    return -torch.logsumexp(torch.stack([-t * x, -t * y], dim=0), dim=0) / t


def smooth_max(x: torch.Tensor, y: torch.Tensor, temperature: float = 32.0) -> torch.Tensor:
    t = max(1e-6, float(temperature))
    return torch.logsumexp(torch.stack([t * x, t * y], dim=0), dim=0) / t


def lukasiewicz_tnorm(x: torch.Tensor, y: torch.Tensor, smooth: float = 0.02) -> torch.Tensor:
    raw = x + y - 1.0
    if smooth <= 0:
        return raw.clamp_min(0.0)
    return smooth * F.softplus(raw / smooth)


def probabilistic_tconorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - x) * (1.0 - y)


__all__ = ["smooth_min", "smooth_max", "lukasiewicz_tnorm", "probabilistic_tconorm"]
