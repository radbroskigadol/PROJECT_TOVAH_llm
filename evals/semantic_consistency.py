"""Semantic consistency smoke eval for K/G lane-routing behavior."""
from __future__ import annotations

import torch

from tovah_v14.evals.common import emit, result
from tovah_v14.neural.training import lane_routing_loss


def run() -> dict:
    gate = torch.zeros(3, 4, requires_grad=True)
    bt = torch.tensor([0.90, 0.10, 0.90])
    bf = torch.tensor([0.88, 0.12, 0.10])
    loss = lane_routing_loss(gate, bt, bf)
    loss.backward()
    grads = gate.grad.detach()
    ok = bool(grads[0, 1] < 0 and grads[1, 2] < 0 and grads[2, 0] < 0)
    return result(
        "semantic_consistency",
        ok,
        loss=float(loss.item()),
        lane_b_grad=float(grads[0, 1].item()),
        lane_c_grad=float(grads[1, 2].item()),
        lane_a_grad=float(grads[2, 0].item()),
    )


if __name__ == "__main__":
    emit(run())
