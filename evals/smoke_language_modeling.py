"""Tiny language-modeling smoke eval for the scalable bilateral core."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tovah_v14.evals.common import emit, result
from tovah_v14.neural.scaling import ScalableBilateralCore


def run() -> dict:
    torch.manual_seed(7)
    model = ScalableBilateralCore(
        vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2,
        n_blocks=1, max_len=32, bilateral_mode="shared", gradient_checkpointing=False,
    )
    x = torch.randint(0, 64, (2, 12))
    y = torch.roll(x, shifts=-1, dims=1)
    tl, fl, gate = model(x)
    loss = F.cross_entropy(tl.reshape(-1, 64), y.reshape(-1)).item()
    ok = tl.shape == (2, 12, 64) and fl.shape == (2, 12, 64) and gate.shape == (2, 12, 4)
    return result("smoke_language_modeling", ok and loss > 0, loss=loss, shape=list(tl.shape))


if __name__ == "__main__":
    emit(run())
