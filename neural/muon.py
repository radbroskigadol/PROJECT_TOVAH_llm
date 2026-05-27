"""v14.3.5 Muon-style classical optimizer floor.

Muon is used here as the *classical projection floor* underneath TOVAH's
bilateral/UAP objectives.  This implementation includes the two details that
matter for practical scaling:

* Nesterov look-ahead momentum before Newton-Schulz orthogonalization.
* Shape-aware matrix learning-rate adjustment
  ``0.2 * lr * sqrt(max(out_dim, in_dim))`` for 2-D parameters.

Non-matrix tensors use AdamW-style RMS scaling.  The wrapper exposes the same
``zero_grad`` / ``step`` / ``step_grads`` interface as the rest of TOVAH.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


def zeropower_via_newtonschulz(g: torch.Tensor, steps: int = 3, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of a 2-D update matrix.

    Three quintic Newton-Schulz iterations are the default because the update is
    only used as an optimizer direction; exact polar decomposition is wasteful.
    """
    if g.ndim != 2:
        return g
    x = g.float()
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, int(steps))):
        xx_t = x @ x.T
        x = a * x + b * (xx_t @ x) + c * (xx_t @ xx_t @ x)
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


class MuonWrapper:
    """Muon-style optimizer with TOVAH's schedule/step_grads interface."""

    def __init__(self, params, *, base_lr: float = 3e-4, momentum: float = 0.95,
                 betas: tuple = (0.9, 0.95), weight_decay: float = 0.1,
                 eps: float = 1e-8, ns_steps: int = 3,
                 matrix_lr_scale: float = 0.2):
        self.params = [p for p in params if p.requires_grad]
        self.base_lr = float(base_lr)
        self._lr = float(base_lr)
        self.momentum = float(momentum)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.ns_steps = int(ns_steps)
        self.matrix_lr_scale = float(matrix_lr_scale)
        self.t = 0
        self._warmup_steps: Optional[int] = None
        self._total_steps: Optional[int] = None
        self._min_lr_ratio = 0.1
        self.state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.last_stats: Dict[str, Any] = {"mode": "muon", "lr": base_lr}

    def _state_for(self, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        st = self.state.get(id(p))
        if st is None:
            st = {"m": torch.zeros_like(p), "rms2": torch.zeros_like(p)}
            self.state[id(p)] = st
        return st

    def set_schedule(self, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> None:
        self._warmup_steps = int(warmup_steps)
        self._total_steps = int(total_steps)
        self._min_lr_ratio = float(min_lr_ratio)

    def _scheduled_lr(self) -> float:
        if self._warmup_steps is None or self._total_steps is None:
            return self.base_lr
        step = self.t
        if step < self._warmup_steps:
            return self.base_lr * (step / max(1, self._warmup_steps))
        progress = (step - self._warmup_steps) / max(1, self._total_steps - self._warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self._min_lr_ratio + (1.0 - self._min_lr_ratio) * cosine)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @staticmethod
    def _matrix_lr_adjustment(lr: float, p: torch.Tensor, scale: float = 0.2) -> float:
        if p.ndim != 2:
            return lr
        return float(scale) * float(lr) * math.sqrt(float(max(p.shape[0], p.shape[1])))

    @torch.no_grad()
    def step_grads(self, phase: str = "Active", loss_value: Optional[float] = None) -> Dict[str, Any]:
        self.t += 1
        scheduled = self._scheduled_lr()
        phase_mult = {"Classical": 0.7, "Active Learning": 1.0, "Active": 1.0, "Collapse-Resistant Paradox": 0.4}.get(phase, 1.0)
        lr = scheduled * phase_mult
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
        matrix_updates = 0
        fallback_updates = 0
        lr_adj_sum = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            st = self._state_for(p)
            prev_m = st["m"]
            # Nesterov look-ahead: use the gradient plus the momentum direction
            # the optimizer is about to take, then commit the EMA update.
            lookahead = g + self.momentum * (prev_m * self.momentum + g * (1.0 - self.momentum))
            prev_m.mul_(self.momentum).add_(g, alpha=1.0 - self.momentum)
            if p.ndim == 2 and p.numel() >= 16:
                update = zeropower_via_newtonschulz(lookahead, steps=self.ns_steps, eps=self.eps)
                step_lr = self._matrix_lr_adjustment(lr, p, self.matrix_lr_scale)
                matrix_updates += 1
            else:
                st["rms2"].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                v_hat = st["rms2"] / max(self.eps, 1.0 - self.beta2 ** self.t)
                m_hat = prev_m / max(self.eps, 1.0 - self.momentum ** self.t)
                update = m_hat / (v_hat.sqrt() + self.eps)
                step_lr = lr
                fallback_updates += 1
            if self.weight_decay > 0:
                p.mul_(1.0 - step_lr * self.weight_decay)
            p.add_(update, alpha=-step_lr)
            lr_adj_sum += step_lr
        self.zero_grad()
        self._lr = lr
        denom = max(1, matrix_updates + fallback_updates)
        self.last_stats = {
            "mode": "muon",
            "optimizer_family": "muon_v14_3_5",
            "phase": phase,
            "lr": lr,
            "scheduled_lr": scheduled,
            "phase_multiplier": phase_mult,
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
            "matrix_updates": matrix_updates,
            "fallback_updates": fallback_updates,
            "mean_effective_step_lr": lr_adj_sum / denom,
            "ns_steps": self.ns_steps,
            "nesterov": True,
            "matrix_lr_scale": self.matrix_lr_scale,
            "step": self.t,
        }
        if loss_value is not None:
            self.last_stats["loss_value"] = float(loss_value)
        return self.last_stats

    def step(self, loss: torch.Tensor, phase: str = "Active") -> Dict[str, Any]:
        loss.backward()
        return self.step_grads(phase=phase, loss_value=float(loss.detach().item()))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_family": "muon_v14_3_5",
            "t": self.t,
            "base_lr": self.base_lr,
            "lr": self._lr,
            "momentum": self.momentum,
            "betas": (self.beta1, self.beta2),
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "ns_steps": self.ns_steps,
            "matrix_lr_scale": self.matrix_lr_scale,
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "min_lr_ratio": self._min_lr_ratio,
            "state_by_index": [
                {k: v.detach().cpu() for k, v in self.state[id(p)].items()} if id(p) in self.state else {}
                for p in self.params
            ],
            "last_stats": dict(self.last_stats),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.t = int(state.get("t", self.t))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self._lr = float(state.get("lr", self._lr))
        self.momentum = float(state.get("momentum", self.momentum))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        self.eps = float(state.get("eps", self.eps))
        self.ns_steps = int(state.get("ns_steps", self.ns_steps))
        self.matrix_lr_scale = float(state.get("matrix_lr_scale", self.matrix_lr_scale))
        self._warmup_steps = state.get("warmup_steps", self._warmup_steps)
        self._total_steps = state.get("total_steps", self._total_steps)
        self._min_lr_ratio = float(state.get("min_lr_ratio", self._min_lr_ratio))
        by_index = state.get("state_by_index") or []
        for i, p in enumerate(self.params):
            if i >= len(by_index):
                continue
            src = by_index[i]
            dst = self._state_for(p)
            for k in ("m", "rms2"):
                if k in src and tuple(src[k].shape) == tuple(dst[k].shape):
                    dst[k].copy_(src[k].to(dst[k].device, dtype=dst[k].dtype))
        self.last_stats = dict(state.get("last_stats", self.last_stats))

    @property
    def lr(self) -> float:
        return self._lr


__all__ = ["MuonWrapper", "zeropower_via_newtonschulz"]
