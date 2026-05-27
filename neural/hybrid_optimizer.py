"""
TOVAH v14.3.1 neural/hybrid_optimizer.py — adaptive AdamW/Shadow mixer.

This module implements a checkpointable hybrid optimizer for TOVAH's
ShadowHoTT/AdamW split. AdamW proposes one update, ShadowOptimizer proposes
another, and a tiny 256-bit gate learns a scalar mixing relation between them.

v14.2.9 fixes the v14.2.8 frozen-gate defect: the previous gate used
``reward * (last_weight - 0.5)``. With initial weights exactly 0.5/0.5, the
update was identically zero forever. The new gate uses first-order proposal
quality, computed from the current accumulated gradient and each optimizer's
candidate update, so the split can move on the very first step.

Gate state is exactly 8 float32 values = 256 bits:
  [adam_logit, shadow_logit, prev_loss, reward_ema,
   last_adam_weight, last_shadow_weight, step_count, score_diff_ema]
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch

from tovah_v14.neural.adamw import AdamWWrapper
from tovah_v14.neural.optimizer import ShadowOptimizer


class HybridGate256:
    """Tiny online learner for the AdamW-vs-ShadowHoTT update split.

    The gate intentionally stays small: 8 float32s. It learns from two signals:
      1. first-order proposal advantage on the current gradient
      2. previous-step loss improvement/worsening as a weak stabilizer

    The first-order score is the important piece. For a proposed parameter
    displacement ``delta``, the local linearized loss change is ``grad·delta``;
    lower is better, so score is ``-grad·delta`` normalized to a cosine-like
    range. This gives the mixer immediate evidence about whether AdamW or
    ShadowHoTT produced the more descent-aligned update without doing extra
    forward passes.
    """

    def __init__(self, lr: float = 0.02, min_weight: float = 0.15,
                 score_ema_beta: float = 0.90,
                 phase_shadow_bias: float = 0.03):
        self.lr = float(lr)
        self.min_weight = float(min_weight)
        self.score_ema_beta = float(score_ema_beta)
        self.phase_shadow_bias = float(phase_shadow_bias)
        self.state = torch.zeros(8, dtype=torch.float32)
        self.state[0] = 0.0  # adam logit
        self.state[1] = 0.0  # shadow logit
        self.state[2] = float("nan")  # previous loss
        self.state[4] = 0.5
        self.state[5] = 0.5

    def weights(self) -> tuple[float, float]:
        logits = self.state[:2].clamp(-8.0, 8.0)
        w = torch.softmax(logits, dim=0)
        floor = self.min_weight
        w = w * (1.0 - 2.0 * floor) + floor
        aw, sw = float(w[0].item()), float(w[1].item())
        s = aw + sw
        return aw / s, sw / s

    @staticmethod
    def _clip_unit(x: float) -> float:
        if not math.isfinite(x):
            return 0.0
        return max(-1.0, min(1.0, float(x)))

    def observe(self, loss_value: Optional[float], *,
                adam_score: Optional[float] = None,
                shadow_score: Optional[float] = None,
                phase: str = "Active") -> Dict[str, float]:
        """Update gate logits and return current split statistics.

        ``adam_score`` and ``shadow_score`` are first-order proposal scores;
        higher is better. A positive score difference favors AdamW, a negative
        score difference favors ShadowHoTT. Phase bias is intentionally tiny and
        only nudges paradox-heavy batches toward ShadowHoTT.
        """
        reward = 0.0
        if loss_value is not None:
            loss = float(loss_value)
            prev = float(self.state[2].item())
            if prev == prev:  # not NaN
                reward = self._clip_unit(prev - loss)
                self.state[3] = 0.95 * self.state[3] + 0.05 * reward
            self.state[2] = loss

        a_score = 0.0 if adam_score is None else self._clip_unit(float(adam_score))
        s_score = 0.0 if shadow_score is None else self._clip_unit(float(shadow_score))
        raw_diff = self._clip_unit(a_score - s_score)  # + means AdamW better

        # EMA of proposal advantage. This smooths single-batch noise while still
        # allowing movement immediately from the first real proposal comparison.
        prev_diff = float(self.state[7].item())
        diff_ema = self.score_ema_beta * prev_diff + (1.0 - self.score_ema_beta) * raw_diff
        self.state[7] = diff_ema

        # Weak reward correction: if recent loss improved, reinforce the last
        # split direction; if it worsened, soften it. This no longer freezes at
        # 0.5/0.5 because the proposal advantage term is independent of weight.
        last_aw = float(self.state[4].item())
        last_sw = float(self.state[5].item())
        reinforce = reward * (last_aw - last_sw)

        phase_name = str(phase or "")
        bias = 0.0
        if phase_name == "Collapse-Resistant Paradox":
            # Negative advantage favors ShadowHoTT. This is a small prior, not
            # a hard override; proposal scores and loss feedback still dominate.
            bias -= self.phase_shadow_bias
        elif phase_name in {"Active", "Active Learning"}:
            bias += 0.5 * self.phase_shadow_bias

        advantage = self._clip_unit(diff_ema + 0.25 * reinforce + bias)
        self.state[0] += self.lr * advantage
        self.state[1] -= self.lr * advantage

        aw, sw = self.weights()
        self.state[4] = aw
        self.state[5] = sw
        self.state[6] += 1.0
        return {
            "adamw_weight": aw,
            "shadow_weight": sw,
            "hybrid_reward": reward,
            "hybrid_reward_ema": float(self.state[3].item()),
            "hybrid_score_diff": raw_diff,
            "hybrid_score_diff_ema": float(self.state[7].item()),
            "hybrid_gate_advantage": advantage,
            "adamw_score": a_score,
            "shadow_score": s_score,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.detach().cpu(),
            "lr": self.lr,
            "min_weight": self.min_weight,
            "score_ema_beta": self.score_ema_beta,
            "phase_shadow_bias": self.phase_shadow_bias,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.lr = float(state.get("lr", self.lr))
        self.min_weight = float(state.get("min_weight", self.min_weight))
        self.score_ema_beta = float(state.get("score_ema_beta", self.score_ema_beta))
        self.phase_shadow_bias = float(state.get("phase_shadow_bias", self.phase_shadow_bias))
        raw = state.get("state")
        if raw is not None:
            t = torch.as_tensor(raw, dtype=torch.float32)
            if t.numel() != 8:
                raise ValueError(f"HybridGate256 expected 8 floats, got {t.numel()}")
            self.state.copy_(t.reshape(8))


class HybridAdamWShadowOptimizer:
    """Mix AdamW and ShadowOptimizer proposed updates with a learned gate."""

    def __init__(self, params, *, base_lr: float = 3e-4,
                 adamw_lr: Optional[float] = None,
                 shadow_lr: Optional[float] = None,
                 gate_lr: float = 0.02,
                 min_weight: float = 0.15,
                 uap_classical_floor: float = 0.15,
                 uap_classical_ceiling: float = 0.85,
                 uap_geometry_lr: float = 0.01,
                 uap_weight_decay: Optional[float] = None,
                 uap_max_update_rms: float = 1.0,
                 uap_trust_clip: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.adamw = AdamWWrapper(self.params, base_lr=adamw_lr or max(3e-4, base_lr))
        self.shadow = ShadowOptimizer(
            self.params,
            base_lr=shadow_lr or base_lr,
            weight_decay=0.1 if uap_weight_decay is None else uap_weight_decay,
            classical_floor=uap_classical_floor,
            classical_ceiling=uap_classical_ceiling,
            geometry_lr=uap_geometry_lr,
            max_update_rms=uap_max_update_rms,
            trust_clip=uap_trust_clip,
        )
        self.gate = HybridGate256(lr=gate_lr, min_weight=min_weight)
        self.t = 0
        self._lr = float(base_lr)
        self.base_lr = float(base_lr)
        self.last_stats: Dict[str, Any] = {"mode": "hybrid_adamw_shadow", "lr": self._lr}

    def set_schedule(self, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> None:
        self.adamw.set_schedule(warmup_steps, total_steps, min_lr_ratio)
        self.shadow.set_schedule(warmup_steps, total_steps, min_lr_ratio)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    @staticmethod
    def _clone_params(params):
        return [p.detach().clone() for p in params]

    @staticmethod
    def _restore_params(params, saved) -> None:
        for p, v in zip(params, saved):
            p.data.copy_(v)

    @staticmethod
    def _clone_grads(params):
        return [None if p.grad is None else p.grad.detach().clone() for p in params]

    @staticmethod
    def _restore_grads(params, grads) -> None:
        for p, g in zip(params, grads):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach().clone()

    @staticmethod
    def _proposal_score(params, grads: Sequence[Optional[torch.Tensor]],
                        base: Sequence[torch.Tensor], proposal: Sequence[torch.Tensor]) -> float:
        """Return normalized first-order descent score for a proposal.

        Positive means the proposal is aligned with ``-grad``. Scores are
        normalized to avoid larger models/layers dominating purely by scale.
        """
        dot = 0.0
        g2 = 0.0
        d2 = 0.0
        for g, b, q in zip(grads, base, proposal):
            if g is None:
                continue
            gd = g.detach().float().reshape(-1)
            dd = (q.detach() - b.detach()).float().reshape(-1)
            if gd.numel() == 0 or dd.numel() == 0:
                continue
            dot += float(torch.dot(gd, dd).item())
            g2 += float(torch.dot(gd, gd).item())
            d2 += float(torch.dot(dd, dd).item())
        denom = math.sqrt(max(g2, 0.0) * max(d2, 0.0)) + 1e-12
        if denom <= 1e-12:
            return 0.0
        score = -dot / denom
        if not math.isfinite(score):
            return 0.0
        return max(-1.0, min(1.0, score))

    def step_grads(self, phase: str = "Active", loss_value: Optional[float] = None) -> Dict[str, Any]:
        base = self._clone_params(self.params)
        grads = self._clone_grads(self.params)

        # AdamW proposal.
        self._restore_grads(self.params, grads)
        adam_stats = self.adamw.step_grads(phase=phase, loss_value=loss_value)
        adam_params = self._clone_params(self.params)
        self._restore_params(self.params, base)

        # ShadowHoTT proposal.
        self._restore_grads(self.params, grads)
        shadow_stats = self.shadow.step_grads(phase=phase, loss_value=loss_value)
        shadow_params = self._clone_params(self.params)
        self._restore_params(self.params, base)

        adam_score = self._proposal_score(self.params, grads, base, adam_params)
        shadow_score = self._proposal_score(self.params, grads, base, shadow_params)
        gate_stats = self.gate.observe(
            loss_value,
            adam_score=adam_score,
            shadow_score=shadow_score,
            phase=phase,
        )
        aw = gate_stats["adamw_weight"]
        sw = gate_stats["shadow_weight"]

        # Mixed application.
        with torch.no_grad():
            for p, b, a, sh in zip(self.params, base, adam_params, shadow_params):
                p.data.copy_(b + aw * (a - b) + sw * (sh - b))

        self.zero_grad()
        self.t += 1
        self._lr = aw * float(adam_stats.get("lr", 0.0)) + sw * float(shadow_stats.get("lr", 0.0))
        self.last_stats = {
            "mode": "hybrid_adamw_shadow",
            "phase": phase,
            "lr": self._lr,
            "step": self.t,
            **gate_stats,
            "adamw_lr": adam_stats.get("lr"),
            "shadow_lr": shadow_stats.get("lr"),
            "adamw_grad_norm": adam_stats.get("grad_norm"),
            "shadow_paradox_mass": shadow_stats.get("paradox_mass"),
            "shadow_gap_mass": shadow_stats.get("gap_mass"),
            "shadow_uap_classical_weight": shadow_stats.get("uap_classical_weight"),
            "shadow_uap_shadow_weight": shadow_stats.get("uap_shadow_weight"),
            "shadow_uap_obstruction": shadow_stats.get("uap_obstruction"),
            "shadow_uap_obstruction_ema": shadow_stats.get("uap_obstruction_ema"),
            "shadow_uap_residue_mass": shadow_stats.get("uap_residue_mass"),
            "shadow_uap_collapse_pressure": shadow_stats.get("uap_collapse_pressure"),
            "shadow_uap_trust_ratio_mean": shadow_stats.get("uap_trust_ratio_mean"),
        }
        return self.last_stats

    def step(self, loss: torch.Tensor, phase: str = "Active") -> Dict[str, Any]:
        loss.backward()
        return self.step_grads(phase=phase, loss_value=float(loss.detach().item()))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": "hybrid_adamw_shadow",
            "t": self.t,
            "base_lr": self.base_lr,
            "lr": self._lr,
            "adamw": self.adamw.state_dict(),
            "shadow": self.shadow.state_dict(),
            "gate256": self.gate.state_dict(),
            "last_stats": dict(self.last_stats),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.t = int(state.get("t", self.t))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self._lr = float(state.get("lr", self._lr))
        self.adamw.load_state_dict(state.get("adamw") or {})
        self.shadow.load_state_dict(state.get("shadow") or {})
        self.gate.load_state_dict(state.get("gate256") or {})
        self.last_stats = dict(state.get("last_stats", self.last_stats))

    @property
    def lr(self) -> float:
        return self._lr
