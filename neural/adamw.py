"""
TOVAH v14.3.1 neural/adamw.py — AdamW path for the scaling regime.

The bilateral ShadowOptimizer is the research substrate: per-parameter
T_sup/F_sup channels, paradox damping, phase-aware LR. That's the right
optimizer for studying the bilateral dynamics themselves.

For *scaling to frontier params*, v14.3.4 also exposes Muon as a stronger
classical floor for matrix parameters. AdamW remains available, but frontier
defaults should prefer Muon when testing random-init scalable profiles.

This module keeps the AdamW path. We thin-wrap `torch.optim.AdamW` to:
  - Match ShadowOptimizer's interface (zero_grad, step, .lr, .last_stats)
    so `pretrain()` can use either without branching
  - Provide warmup + cosine decay schedule (the standard frontier recipe)
  - Optional foreach=True for ~2x optimizer step speed (PyTorch ≥1.12)

This is NOT a replacement for the bilateral optimizer — it's a
complement, selected via env var or `pretrain(optimizer=...)`:

    TOVAH_OPTIMIZER=shadow  → ShadowOptimizer (bilateral, research)
    TOVAH_OPTIMIZER=adamw   → AdamWWrapper (frontier scaling)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch


class AdamWWrapper:
    """torch.optim.AdamW with the ShadowOptimizer interface.

    Exposes the same surface (`zero_grad`, `step`, `.lr`, `.last_stats`,
    `set_schedule`) so the training loop can swap optimizers via a
    config flag without code branches.

    Key parameters (frontier defaults):
      lr=3e-4         (peak LR after warmup; standard for 1-7B scale)
      betas=(0.9, 0.95)  (β2 lower than torch default; matches Llama)
      weight_decay=0.1   (high WD; matches Llama / Mistral)
      eps=1e-8
    """

    def __init__(self, params, *,
                 base_lr: float = 3e-4,
                 betas: tuple = (0.9, 0.95),
                 weight_decay: float = 0.1,
                 eps: float = 1e-8,
                 foreach: bool = True):
        self.params = [p for p in params if p.requires_grad]
        self.base_lr = float(base_lr)
        self._lr = float(base_lr)
        # foreach=True dispatches the optimizer math as a single fused
        # operation per parameter group. Up to ~2x faster on large models.
        try:
            self._opt = torch.optim.AdamW(
                self.params,
                lr=base_lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps,
                foreach=foreach,
            )
        except TypeError:
            # Older torch without foreach kw.
            self._opt = torch.optim.AdamW(
                self.params,
                lr=base_lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps,
            )
        self.t = 0
        self._warmup_steps: Optional[int] = None
        self._total_steps: Optional[int] = None
        self._min_lr_ratio = 0.1
        self.last_stats: Dict[str, Any] = {
            "mode": "adamw",
            "lr": base_lr,
        }

    def set_schedule(self, warmup_steps: int, total_steps: int,
                     min_lr_ratio: float = 0.1) -> None:
        """Linear warmup → cosine decay (matches ShadowOptimizer)."""
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
        self._opt.zero_grad(set_to_none=True)

    def step_grads(self, phase: str = "Active", loss_value: Optional[float] = None) -> Dict[str, Any]:
        """Apply one AdamW update from already-accumulated ``p.grad`` tensors.

        This is the production path used by ``training.pretrain`` so gradient
        accumulation works correctly. ``step(loss)`` below remains a compatibility
        wrapper for older tests/callers.
        """
        self.t += 1
        scheduled = self._scheduled_lr()
        phase_mult = {
            "Classical": 0.7,
            "Active Learning": 1.0,
            "Active": 1.0,
            "Collapse-Resistant Paradox": 0.4,
        }.get(phase, 1.0)
        lr = scheduled * phase_mult
        for pg in self._opt.param_groups:
            pg["lr"] = lr
        clip_norm = 0.5 if phase == "Collapse-Resistant Paradox" else 1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, max_norm=clip_norm)
        self._opt.step()
        self._lr = lr
        self.last_stats = {
            "mode": "adamw",
            "phase": phase,
            "lr": lr,
            "scheduled_lr": scheduled,
            "phase_multiplier": phase_mult,
            "grad_clip_norm": clip_norm,
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
            "step": self.t,
            "paradox_mass": 0.0,
            "gap_mass": 0.0,
        }
        if loss_value is not None:
            self.last_stats["loss_value"] = float(loss_value)
        return self.last_stats

    def step(self, loss: torch.Tensor, phase: str = "Active") -> Dict[str, Any]:
        """Compatibility wrapper: backpropagate ``loss`` then call ``step_grads``."""
        loss.backward()
        return self.step_grads(phase=phase, loss_value=float(loss.detach().item()))


    def state_dict(self) -> Dict[str, Any]:
        """Serializable optimizer/schedule state for resumable training."""
        return {
            "optimizer": self._opt.state_dict(),
            "t": self.t,
            "base_lr": self.base_lr,
            "lr": self._lr,
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "min_lr_ratio": self._min_lr_ratio,
            "last_stats": dict(self.last_stats),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore optimizer/schedule state saved by state_dict()."""
        if not state:
            return
        opt_state = state.get("optimizer")
        if opt_state is not None:
            self._opt.load_state_dict(opt_state)
        self.t = int(state.get("t", self.t))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self._lr = float(state.get("lr", self._lr))
        self._warmup_steps = state.get("warmup_steps", self._warmup_steps)
        self._total_steps = state.get("total_steps", self._total_steps)
        self._min_lr_ratio = float(state.get("min_lr_ratio", self._min_lr_ratio))
        self.last_stats = dict(state.get("last_stats", self.last_stats))

    @property
    def lr(self) -> float:
        return self._lr


def make_optimizer(model_params, *,
                   kind: str = "shadow",
                   base_lr: Optional[float] = None,
                   uap_classical_floor: float = 0.15,
                   uap_classical_ceiling: float = 0.85,
                   uap_geometry_lr: float = 0.01,
                   uap_weight_decay: Optional[float] = None,
                   uap_max_update_rms: float = 1.0,
                   uap_trust_clip: float = 0.0,
                   hybrid_gate_lr: float = 0.02,
                   hybrid_min_adamw_weight: float = 0.15,
                   ) -> Any:
    """Optimizer factory.

    kind:
      "shadow" — UAP ShadowOptimizer (default; bilateral + AdamW-classicalized substrate)
      "adamw"  — AdamWWrapper (frontier scaling)
      "muon"   — MuonWrapper (orthogonalized matrix-update classical floor)

    Set base_lr if you want a non-default rate. Defaults:
      shadow: 3e-4
      adamw:  3e-4
    """
    kind = str(kind or "shadow").lower()
    if kind == "adamw":
        return AdamWWrapper(model_params, base_lr=base_lr or 3e-4)
    if kind == "muon":
        from tovah_v14.neural.muon import MuonWrapper
        return MuonWrapper(model_params, base_lr=base_lr or 3e-4)
    if kind in {"shadow", "uap_shadow", "uap"}:
        from tovah_v14.neural.optimizer import ShadowOptimizer
        return ShadowOptimizer(
            model_params,
            base_lr=base_lr or 3e-4,
            weight_decay=0.1 if uap_weight_decay is None else uap_weight_decay,
            classical_floor=uap_classical_floor,
            classical_ceiling=uap_classical_ceiling,
            geometry_lr=uap_geometry_lr,
            max_update_rms=uap_max_update_rms,
            trust_clip=uap_trust_clip,
        )
    if kind in {"hybrid", "adamw_shadow", "shadow_adamw"}:
        from tovah_v14.neural.hybrid_optimizer import HybridAdamWShadowOptimizer
        return HybridAdamWShadowOptimizer(
            model_params,
            base_lr=base_lr or 3e-4,
            gate_lr=hybrid_gate_lr,
            min_weight=hybrid_min_adamw_weight,
            uap_classical_floor=uap_classical_floor,
            uap_classical_ceiling=uap_classical_ceiling,
            uap_geometry_lr=uap_geometry_lr,
            uap_weight_decay=uap_weight_decay,
            uap_max_update_rms=uap_max_update_rms,
            uap_trust_clip=uap_trust_clip,
        )
    raise ValueError(f"unknown optimizer kind {kind!r}; use 'shadow', 'uap_shadow', 'adamw', 'muon', or 'hybrid'")
