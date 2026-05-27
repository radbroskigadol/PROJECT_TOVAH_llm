"""Auxiliary UAP / ShadowHoTT losses for v14.3.2.

These losses are optional scaffolding around cross entropy.  They assume the
model eventually exposes auxiliary heads or probes named:
  t_support, f_support, glut_mass, gap_mass, obstruction_residue,
  collapse_pressure, classicalization_depth.

If no torch tensors are supplied, the functions fail loudly enough for wiring
mistakes but do not force torch import during corpus/eval tooling.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

try:  # pragma: no cover - depends on training env
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

PROFILE_KEYS = (
    "t_support", "f_support", "glut_mass", "gap_mass", "obstruction_residue",
    "collapse_pressure", "classicalization_depth",
)

DEFAULT_LOSS_WEIGHTS = {
    "collapse_penalty": 0.10,
    "residue_preservation_loss": 0.20,
    "glut_retention_loss": 0.15,
    "gap_recognition_loss": 0.15,
    "local_global_obstruction_loss": 0.20,
    "support_profile_consistency_loss": 0.20,
}


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("uap_aux_losses requires torch when computing training losses")


def _as_target_tensor(profile_targets: Mapping[str, Any], key: str, like: Any):
    _require_torch()
    value = profile_targets.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _expand_target_like(target: Any, pred: Any):
    _require_torch()
    if target is None:
        return None
    if target.ndim == 0:
        return target.expand_as(pred)
    while target.ndim < pred.ndim:
        target = target.unsqueeze(-1)
    return target.expand_as(pred)


def _mse(aux_outputs: Mapping[str, Any], profile_targets: Mapping[str, Any], key: str):
    pred = aux_outputs.get(key)
    if pred is None:
        return None
    target = _as_target_tensor(profile_targets, key, pred)
    if target is None:
        return None
    target = _expand_target_like(target, pred)
    return F.mse_loss(pred, target)


def uap_auxiliary_losses(
    aux_outputs: Mapping[str, Any],
    profile_targets: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Return individual auxiliary losses and weighted total.

    This should be added to cross entropy as:
      total = cross_entropy + losses["uap_aux_total"]
    """
    _require_torch()
    w = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    zero = None
    for v in aux_outputs.values():
        if torch.is_tensor(v):
            zero = v.sum() * 0.0
            break
    if zero is None:
        raise ValueError("aux_outputs contains no torch tensors")

    losses: Dict[str, Any] = {}
    # Collapse penalty: pressure should be low when target profile has glut/gap/obstruction.
    cp = aux_outputs.get("collapse_pressure")
    if cp is not None:
        target_glut = _as_target_tensor(profile_targets, "glut_mass", cp)
        target_gap = _as_target_tensor(profile_targets, "gap_mass", cp)
        target_obs = _as_target_tensor(profile_targets, "obstruction_residue", cp)
        anti_collapse_need = 0.0
        for t in (target_glut, target_gap, target_obs):
            if t is not None:
                anti_collapse_need = anti_collapse_need + t
        anti_collapse_need = torch.clamp(anti_collapse_need, 0.0, 1.0) if torch.is_tensor(anti_collapse_need) else torch.as_tensor(0.0, device=cp.device, dtype=cp.dtype)
        anti_collapse_need = _expand_target_like(anti_collapse_need, cp)
        losses["collapse_penalty"] = F.mse_loss(cp * anti_collapse_need, torch.zeros_like(cp))
    else:
        losses["collapse_penalty"] = zero

    def _or_zero(value):
        return value if value is not None else zero

    losses["residue_preservation_loss"] = _or_zero(_mse(aux_outputs, profile_targets, "obstruction_residue"))
    losses["glut_retention_loss"] = _or_zero(_mse(aux_outputs, profile_targets, "glut_mass"))
    losses["gap_recognition_loss"] = _or_zero(_mse(aux_outputs, profile_targets, "gap_mass"))
    losses["local_global_obstruction_loss"] = _or_zero(_mse(aux_outputs, profile_targets, "obstruction_residue"))

    ts = _or_zero(_mse(aux_outputs, profile_targets, "t_support"))
    fs = _or_zero(_mse(aux_outputs, profile_targets, "f_support"))
    losses["support_profile_consistency_loss"] = 0.5 * (ts + fs)

    total = zero
    for name, loss in losses.items():
        total = total + float(w.get(name, 0.0)) * loss
    losses["uap_aux_total"] = total
    return losses


def add_uap_losses_to_cross_entropy(cross_entropy_loss: Any, aux_outputs: Mapping[str, Any], profile_targets: Mapping[str, Any], *, weights: Optional[Mapping[str, float]] = None):
    losses = uap_auxiliary_losses(aux_outputs, profile_targets, weights=weights)
    losses["cross_entropy"] = cross_entropy_loss
    losses["total_loss"] = cross_entropy_loss + losses["uap_aux_total"]
    return losses



def semantic_outputs_from_supports(T: Any, Fv: Any, attention_mask: Optional[Any] = None) -> Dict[str, Any]:
    """Build UAP auxiliary outputs from existing T/F semantic supports.

    This gives v14.3.2 real training/eval wiring before dedicated learned heads
    exist.  It does not replace future heads; it exposes the current bilateral
    tensors as profile predictions: T-support, F-support, glut, gap, obstruction
    residue, collapse pressure, and classicalization depth.
    """
    _require_torch()
    t_support = T.mean(dim=-1) if T.ndim >= 3 else T
    f_support = Fv.mean(dim=-1) if Fv.ndim >= 3 else Fv
    glut_mass = torch.minimum(t_support, f_support)
    gap_mass = 1.0 - torch.maximum(t_support, f_support)
    gap_mass = torch.clamp(gap_mass, 0.0, 1.0)
    obstruction_residue = torch.clamp(0.40 * glut_mass + 0.35 * gap_mass + 0.25 * torch.abs(t_support - f_support), 0.0, 1.0)
    # Collapse pressure is high when one support dominates and low when K/G/residue
    # must be preserved.  This derived proxy is deliberately conservative.
    collapse_pressure = torch.clamp(torch.abs(t_support - f_support) * (1.0 - obstruction_residue), 0.0, 1.0)
    classicalization_depth = torch.clamp(1.0 - (0.38 * glut_mass + 0.32 * gap_mass + 0.30 * obstruction_residue), 0.0, 1.0)
    return {
        "t_support": t_support,
        "f_support": f_support,
        "glut_mass": glut_mass,
        "gap_mass": gap_mass,
        "obstruction_residue": obstruction_residue,
        "collapse_pressure": collapse_pressure,
        "classicalization_depth": classicalization_depth,
    }
