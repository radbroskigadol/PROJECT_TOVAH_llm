"""
TOVAH v14.3.1 neural/optimizer.py — UAP/ShadowHoTT optimizer.

This file keeps the public class name ``ShadowOptimizer`` but upgrades the
optimizer from the early bilateral sketch into a UAP-shaped AdamW-classicalizing
optimizer.

Core idea
---------
AdamW is treated as the classical shadow/projection of the richer bilateral
optimizer, not as an external replacement.  The optimizer therefore has:

  - compact K_glut/R_obs bilateral EMA diagnostics
  - derived manifestation/gap/collapse diagnostics computed on demand
  - Adam-style first moment and second moment scaling
  - bias correction for both classical and bilateral support moments
  - decoupled weight decay
  - trust-ratio / update-norm magnitude control
  - a tiny UAP geometry gate that chooses the classicalization floor vs.
    ShadowHoTT correction using obstruction and loss-improvement signals

Compatibility
-------------
Existing callers still use ``ShadowOptimizer``. v14.3.4 stores only four
per-parameter buffers: m, rms2, K_glut_q, and R_obs_q.  Old checkpoints with
T_sup/F_sup/K_glut/R_obs are tolerated and compacted on load.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


class UAPGeometryGate:
    """Small online controller for classicalization-vs-ShadowHoTT geometry.

    State is 8 float32 values:
      [classical_logit, shadow_logit, prev_loss, reward_ema,
       obstruction_ema, last_classical_weight, last_shadow_weight, step]

    The gate is deliberately slow. AdamW's classical projection remains
    available as a floor, while obstruction-rich batches can shift weight to
    the ShadowHoTT correction.
    """

    def __init__(self, *, lr: float = 0.01, classical_floor: float = 0.15,
                 classical_ceiling: float = 0.85, shadow_bias: float = 0.05,
                 obstruction_beta: float = 0.95):
        self.lr = float(lr)
        self.classical_floor = float(classical_floor)
        self.classical_ceiling = float(classical_ceiling)
        self.shadow_bias = float(shadow_bias)
        self.obstruction_beta = float(obstruction_beta)
        self.state = torch.zeros(8, dtype=torch.float32)
        # Start AdamW-led but not AdamW-owned: classical projection is strong
        # early so language acquisition is stable, Shadow can earn control.
        self.state[0] = 0.35
        self.state[1] = -0.35
        self.state[2] = float("nan")
        cw, sw = self.weights()
        self.state[5] = cw
        self.state[6] = sw

    @staticmethod
    def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        if not math.isfinite(float(x)):
            return 0.0
        return max(lo, min(hi, float(x)))

    def weights(self) -> tuple[float, float]:
        logits = self.state[:2].clamp(-8.0, 8.0)
        w = torch.softmax(logits, dim=0)
        cw = float(w[0].item())
        cw = max(self.classical_floor, min(self.classical_ceiling, cw))
        sw = 1.0 - cw
        return cw, sw

    def observe(self, *, loss_value: Optional[float], obstruction: float,
                phase: str = "Active") -> Dict[str, float]:
        obstruction = self._clip(obstruction, 0.0, 1.0)
        prev_obs = float(self.state[4].item())
        obs_ema = self.obstruction_beta * prev_obs + (1.0 - self.obstruction_beta) * obstruction
        self.state[4] = obs_ema

        reward = 0.0
        if loss_value is not None:
            loss = float(loss_value)
            prev = float(self.state[2].item())
            if prev == prev:  # not NaN
                reward = self._clip(prev - loss)
                self.state[3] = 0.95 * self.state[3] + 0.05 * reward
            self.state[2] = loss

        last_c = float(self.state[5].item())
        last_s = float(self.state[6].item())

        # Positive advantage means "more classical projection"; negative means
        # "more ShadowHoTT correction". Obstruction-rich batches favor Shadow;
        # low-obstruction or worsening-loss batches favor the classical floor.
        advantage = 0.55 - obs_ema
        if reward < 0:
            advantage += 0.20 * abs(reward)  # stabilize after bad steps
        else:
            advantage += 0.08 * reward * (last_c - last_s)

        phase_name = str(phase or "")
        if phase_name == "Collapse-Resistant Paradox":
            advantage -= self.shadow_bias
        elif phase_name in {"Classical", "Active", "Active Learning"}:
            advantage += 0.5 * self.shadow_bias

        advantage = self._clip(advantage)
        self.state[0] += self.lr * advantage
        self.state[1] -= self.lr * advantage
        cw, sw = self.weights()
        self.state[5] = cw
        self.state[6] = sw
        self.state[7] += 1.0
        return {
            "uap_classical_weight": cw,
            "uap_shadow_weight": sw,
            "uap_gate_reward": reward,
            "uap_gate_reward_ema": float(self.state[3].item()),
            "uap_obstruction_ema": obs_ema,
            "uap_gate_advantage": advantage,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.detach().cpu(),
            "lr": self.lr,
            "classical_floor": self.classical_floor,
            "classical_ceiling": self.classical_ceiling,
            "shadow_bias": self.shadow_bias,
            "obstruction_beta": self.obstruction_beta,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.lr = float(state.get("lr", self.lr))
        self.classical_floor = float(state.get("classical_floor", self.classical_floor))
        self.classical_ceiling = float(state.get("classical_ceiling", self.classical_ceiling))
        self.shadow_bias = float(state.get("shadow_bias", self.shadow_bias))
        self.obstruction_beta = float(state.get("obstruction_beta", self.obstruction_beta))
        raw = state.get("state")
        if raw is not None:
            t = torch.as_tensor(raw, dtype=torch.float32)
            if t.numel() != 8:
                raise ValueError(f"UAPGeometryGate expected 8 floats, got {t.numel()}")
            self.state.copy_(t.reshape(8))


class ShadowOptimizer:
    """UAP/ShadowHoTT optimizer with AdamW as classicalization floor.

    The public name is intentionally unchanged: existing commands with
    ``--optimizer shadow`` now get the more mature UAP optimizer.
    """

    def __init__(self, params, base_lr: float = 2e-4, weight_decay: float = 0.1,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 support_beta: float = 0.90,
                 classical_floor: float = 0.15,
                 classical_ceiling: float = 0.85,
                 geometry_lr: float = 0.01,
                 trust_clip: float = 0.0,
                 max_update_norm: float = 0.0,
                 max_update_rms: float = 1.0,
                 grad_clip_norm: float = 1.0):
        self.params = [p for p in params if p.requires_grad]
        self._lr = float(base_lr)
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.support_beta = float(support_beta)
        # ``trust_clip`` is an optional LAMB-like relative trust ratio. It is
        # disabled by default because absolute tensor-norm trust ratios punish
        # large matrices and made the v14.3.0 Shadow update orders of magnitude
        # too small compared with AdamW. Stable magnitude control is now done by
        # update RMS, which is size-invariant and preserves AdamW-like elementwise
        # step scale when the classicalized component dominates.
        self.trust_clip = float(trust_clip)
        self.max_update_norm = float(max_update_norm)
        self.max_update_rms = float(max_update_rms)
        self.grad_clip_norm = float(grad_clip_norm)
        self.t = 0
        self._state_initialized = False
        self.state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.gate = UAPGeometryGate(
            lr=geometry_lr,
            classical_floor=classical_floor,
            classical_ceiling=classical_ceiling,
        )
        self._warmup_steps: Optional[int] = None
        self._total_steps: Optional[int] = None
        self._min_lr_ratio: float = 0.1
        self.last_stats: Dict[str, Any] = {
            "mode": "uap_shadow_hott",
            "lr": base_lr,
            "uap_classical_weight": classical_floor,
            "uap_shadow_weight": 1.0 - classical_floor,
        }

    def set_schedule(self, warmup_steps: int, total_steps: int,
                     min_lr_ratio: float = 0.1) -> None:
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

    def _ensure_state(self) -> None:
        if self._state_initialized:
            return
        self.state = {}
        for p in self.params:
            self.state[id(p)] = {
                # v14.3.4 compact state: four persistent buffers per parameter.
                # K/R are uint8 EMAs in [0,1]; derived T/F/M/D/G/C are computed
                # on demand instead of stored.
                "m": torch.zeros_like(p),
                "rms2": torch.zeros_like(p),
                "K_glut_q": torch.zeros(p.shape, dtype=torch.uint8, device=p.device),
                "R_obs_q": torch.zeros(p.shape, dtype=torch.uint8, device=p.device),
            }
        self._state_initialized = True

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @staticmethod
    def _safe_item(x: torch.Tensor) -> float:
        try:
            v = float(x.detach().float().mean().item())
            return v if math.isfinite(v) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _dequant01(q: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return q.to(device=like.device, dtype=like.dtype) / 255.0

    @staticmethod
    def _quant01_(target: torch.Tensor, value: torch.Tensor) -> None:
        target.copy_((value.detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8))

    def _global_clip_grads(self, grads) -> tuple[list[Optional[torch.Tensor]], float]:
        total = 0.0
        for g in grads:
            if g is not None:
                gd = g.detach().float()
                total += float(torch.dot(gd.reshape(-1), gd.reshape(-1)).item())
        norm = math.sqrt(max(0.0, total))
        coef = 1.0
        if self.grad_clip_norm > 0 and norm > self.grad_clip_norm:
            coef = self.grad_clip_norm / (norm + self.eps)
        out = [None if g is None else (g.detach() * coef) for g in grads]
        return out, norm

    def _apply_grads(self, grads, phase: str = "Active", loss_value: Optional[float] = None) -> Dict[str, Any]:
        self._ensure_state()
        self.t += 1
        scheduled = self._scheduled_lr()
        phase_mult = {
            "Classical": 0.7,
            "Active Learning": 1.0,
            "Active": 1.0,
            "Collapse-Resistant Paradox": 0.4,
        }.get(phase, 1.0)
        lr = scheduled * phase_mult
        self._lr = scheduled

        clipped_grads, raw_grad_norm = self._global_clip_grads(grads)

        # First pass: update compact moments and obstruction diagnostics.
        pm = gm = rm = cm = 0.0
        elem_count = 0
        for p, g in zip(self.params, clipped_grads):
            if g is None:
                continue
            g = torch.clamp(g, -5.0, 5.0)
            st = self.state[id(p)]
            prev_m = st["m"].detach()

            # Glut proxy: oscillating/opposed gradient evidence.  This retains
            # the genuinely novel bilateral signal without storing redundant
            # T_sup/F_sup/M/D/G/C tensors.
            conflict = torch.relu(-(g * prev_m)) * 2.0 / (g.square() + prev_m.square() + self.eps)
            conflict = conflict.clamp(0.0, 1.0)
            old_k = self._dequant01(st["K_glut_q"], p)
            old_r = self._dequant01(st["R_obs_q"], p)
            K = old_k.mul(self.support_beta).add(conflict, alpha=1.0 - self.support_beta).clamp(0.0, 1.0)
            R = old_r.mul(0.95).add(conflict, alpha=0.05).clamp(0.0, 1.0)
            self._quant01_(st["K_glut_q"], K)
            self._quant01_(st["R_obs_q"], R)

            st["m"].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            st["rms2"].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            # Gap/collapse are derived diagnostics. Gap is high where the
            # classical moment remains near zero; collapse is obstruction pressure
            # against a high-glut locus.
            M = st["m"].detach().abs()
            G = torch.exp(-M.float()).to(M.dtype)
            collapse = K * conflict

            n = max(1, p.numel())
            elem_count += n
            pm += float(K.detach().float().sum().item())
            gm += float(G.detach().float().sum().item())
            rm += float(R.detach().float().sum().item())
            cm += float(collapse.detach().float().sum().item())

        denom_elems = max(1, elem_count)
        paradox_mass = pm / denom_elems
        gap_mass = gm / denom_elems
        residue_mass = rm / denom_elems
        collapse_pressure = cm / denom_elems
        # Bounded obstruction scalar: supports the geometry gate, not a theorem.
        obstruction = max(0.0, min(1.0, 0.35 * paradox_mass + 0.15 * gap_mass + 0.35 * residue_mass + 0.15 * collapse_pressure))
        gate_stats = self.gate.observe(loss_value=loss_value, obstruction=obstruction, phase=phase)
        cw = gate_stats["uap_classical_weight"]
        sw = gate_stats["uap_shadow_weight"]

        # Second pass: apply decoupled decay + bias-corrected UAP/classical update.
        trust_sum = 0.0
        trust_n = 0
        update_rms_sum = 0.0
        update_rms_n = 0
        effective_scale_sum = 0.0
        effective_scale_n = 0
        beta1_corr = max(self.eps, 1.0 - self.beta1 ** self.t)
        beta2_corr = max(self.eps, 1.0 - self.beta2 ** self.t)
        for p, g in zip(self.params, clipped_grads):
            if g is None:
                continue
            st = self.state[id(p)]
            m_hat = st["m"] / beta1_corr
            v_hat = st["rms2"] / beta2_corr
            denom = torch.sqrt(v_hat) + self.eps
            classical_step = -m_hat / denom

            K_hat = self._dequant01(st["K_glut_q"], p)
            R = self._dequant01(st["R_obs_q"], p)
            G = torch.exp(-st["m"].detach().abs().float()).to(st["m"].dtype)
            damping = 1.0 / (1.0 + 0.75 * K_hat + 0.25 * G + 0.50 * R)
            # Compact Shadow correction: AdamW's classical descent remains the
            # floor/projection, while K/R-aware damping protects loci showing
            # contradictory or obstruction-rich gradient evidence.
            shadow_step = classical_step * damping
            update = cw * classical_step + sw * shadow_step

            # Stable magnitude control: trust ratio + update norm clip. This is
            # AdamW-like optimizer hygiene expressed after UAP mixing.
            with torch.no_grad():
                p_norm = float(p.detach().float().norm().item())
                u_norm = float(update.detach().float().norm().item())
                n_elem = max(1, p.numel())
                update_rms = u_norm / math.sqrt(n_elem)
                trust = 1.0

                # Optional relative trust ratio. Disabled by default. When
                # enabled, it is symmetric around 1.0 rather than a one-way
                # suppressor, so classical AdamW-scale steps are recoverable.
                if self.trust_clip > 0 and p_norm > 0 and u_norm > 0:
                    raw_trust = p_norm / (u_norm + self.eps)
                    lo = 1.0 / max(1.0, self.trust_clip)
                    hi = max(1.0, self.trust_clip)
                    trust = max(lo, min(hi, raw_trust))

                # Size-invariant magnitude hygiene. v14.3.0 used an absolute
                # tensor-norm cap; for large matrices that divided the update by
                # sqrt(numel) and prevented learning. RMS capping preserves the
                # AdamW sign-step scale while still preventing pathological
                # elementwise explosions.
                if self.max_update_rms > 0 and update_rms * trust > self.max_update_rms:
                    trust *= self.max_update_rms / (update_rms * trust + self.eps)
                if self.max_update_norm > 0 and u_norm * trust > self.max_update_norm:
                    trust *= self.max_update_norm / (u_norm * trust + self.eps)

                if self.weight_decay > 0:
                    p.data.mul_(1.0 - lr * self.weight_decay)
                p.data.add_(update, alpha=lr * trust)
                trust_sum += trust
                trust_n += 1
                update_rms_sum += update_rms
                update_rms_n += 1
                effective_scale_sum += trust
                effective_scale_n += 1

        self.zero_grad()
        self.last_stats = {
            "mode": "uap_shadow_hott",
            "phase": phase,
            "lr": lr,
            "scheduled_lr": scheduled,
            "phase_multiplier": phase_mult,
            "step": self.t,
            "grad_norm": raw_grad_norm,
            "paradox_mass": paradox_mass,
            "gap_mass": gap_mass,
            "uap_residue_mass": residue_mass,
            "uap_collapse_pressure": collapse_pressure,
            "uap_obstruction": obstruction,
            "uap_trust_ratio_mean": trust_sum / max(1, trust_n),
            "uap_update_rms_mean": update_rms_sum / max(1, update_rms_n),
            "uap_effective_scale_mean": effective_scale_sum / max(1, effective_scale_n),
            "uap_max_update_rms": self.max_update_rms,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            **gate_stats,
        }
        if loss_value is not None:
            self.last_stats["loss_value"] = float(loss_value)
        return self.last_stats

    def step_grads(self, phase: str = "Active", loss_value: Optional[float] = None) -> Dict[str, Any]:
        grads = [None if p.grad is None else p.grad for p in self.params]
        return self._apply_grads(grads, phase=phase, loss_value=loss_value)

    def step(self, loss: torch.Tensor, phase: str = "Active") -> Dict[str, Any]:
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        return self._apply_grads(grads, phase=phase, loss_value=float(loss.detach().item()))

    def state_dict(self) -> Dict[str, Any]:
        self._ensure_state()
        state_by_index = []
        legacy_state = {}
        for p in self.params:
            tensors = self.state.get(id(p), {})
            cpu_tensors = {name: val.detach().cpu() for name, val in tensors.items()}
            state_by_index.append(cpu_tensors)
            legacy_state[id(p)] = cpu_tensors
        return {
            "optimizer_family": "uap_shadow_hott_v14_3_4_compact",
            "optimizer_state_buffers_per_param": 4,
            "bilateral_state_format": "uint8_ema_0_1_for_K_glut_and_R_obs",
            "t": self.t,
            "lr": self._lr,
            "base_lr": self.base_lr,
            "weight_decay": self.weight_decay,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "support_beta": self.support_beta,
            "trust_clip": self.trust_clip,
            "max_update_norm": self.max_update_norm,
            "max_update_rms": self.max_update_rms,
            "grad_clip_norm": self.grad_clip_norm,
            "state_initialized": self._state_initialized,
            "state_by_index": state_by_index,
            "state": legacy_state,
            "uap_gate": self.gate.state_dict(),
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "min_lr_ratio": self._min_lr_ratio,
            "last_stats": dict(self.last_stats),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.t = int(state.get("t", self.t))
        self._lr = float(state.get("lr", self._lr))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        betas = state.get("betas")
        if betas is not None and len(betas) >= 2:
            self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(state.get("eps", self.eps))
        self.support_beta = float(state.get("support_beta", self.support_beta))
        self.trust_clip = float(state.get("trust_clip", self.trust_clip))
        self.max_update_norm = float(state.get("max_update_norm", self.max_update_norm))
        self.max_update_rms = float(state.get("max_update_rms", self.max_update_rms))
        self.grad_clip_norm = float(state.get("grad_clip_norm", self.grad_clip_norm))
        self._warmup_steps = state.get("warmup_steps", self._warmup_steps)
        self._total_steps = state.get("total_steps", self._total_steps)
        self._min_lr_ratio = float(state.get("min_lr_ratio", self._min_lr_ratio))
        self.gate.load_state_dict(state.get("uap_gate") or {})
        self.last_stats = dict(state.get("last_stats", self.last_stats))

        by_index = state.get("state_by_index")
        legacy = state.get("state") or {}
        if by_index or legacy:
            self._ensure_state()
            for i, p in enumerate(self.params):
                target = self.state[id(p)]
                old = None
                if isinstance(by_index, list) and i < len(by_index):
                    old = by_index[i]
                elif id(p) in legacy:
                    old = legacy.get(id(p))
                if not old:
                    continue
                # v14.3.4 compact load path.  New checkpoints store m/rms2 and
                # uint8 K/R.  Older checkpoints with full T/F/K/R tensors are
                # compacted here rather than rejected.
                for name in ("m", "rms2"):
                    if name in old:
                        val = old[name]
                        if tuple(val.shape) != tuple(target[name].shape):
                            raise ValueError(
                                f"ShadowOptimizer state shape mismatch for param {i} {name}: "
                                f"checkpoint {tuple(val.shape)} != current {tuple(target[name].shape)}"
                            )
                        target[name].copy_(val.to(target[name].device, dtype=target[name].dtype))
                if "K_glut_q" in old:
                    val = old["K_glut_q"]
                    if tuple(val.shape) == tuple(target["K_glut_q"].shape):
                        target["K_glut_q"].copy_(val.to(target["K_glut_q"].device, dtype=torch.uint8))
                elif "K_glut" in old:
                    val = old["K_glut"]
                    if tuple(val.shape) == tuple(target["K_glut_q"].shape):
                        self._quant01_(target["K_glut_q"], val.to(target["m"].device))
                elif "T_sup" in old and "F_sup" in old:
                    tv, fv = old["T_sup"], old["F_sup"]
                    if tuple(tv.shape) == tuple(target["K_glut_q"].shape) and tuple(fv.shape) == tuple(target["K_glut_q"].shape):
                        self._quant01_(target["K_glut_q"], torch.minimum(tv.to(target["m"].device), fv.to(target["m"].device)))
                if "R_obs_q" in old:
                    val = old["R_obs_q"]
                    if tuple(val.shape) == tuple(target["R_obs_q"].shape):
                        target["R_obs_q"].copy_(val.to(target["R_obs_q"].device, dtype=torch.uint8))
                elif "R_obs" in old:
                    val = old["R_obs"]
                    if tuple(val.shape) == tuple(target["R_obs_q"].shape):
                        self._quant01_(target["R_obs_q"], val.to(target["m"].device))

    @property
    def lr(self) -> float:
        return self._lr
