"""TOVAH v14.3.3 Shadow/UAP support-profile objective helpers.

These helpers formalize the token-profile fields used by the v14.3.x line:
truth support, falsity support, glut/gap mass, obstruction residue, collapse
pressure, and classicalization depth.  They are designed to be attached as an
auxiliary objective without disturbing the known-good cross-entropy path.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math

PROFILE_FIELDS: Tuple[str, ...] = (
    "T_support",
    "F_support",
    "glut_mass",
    "gap_mass",
    "obstruction_residue",
    "collapse_pressure",
    "classicalization_depth",
)

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "T_support": ("T_support", "t_support", "truth_support", "bilateral_t", "t"),
    "F_support": ("F_support", "f_support", "falsity_support", "bilateral_f", "f"),
    "glut_mass": ("glut_mass", "glut", "both_mass"),
    "gap_mass": ("gap_mass", "gap", "neither_mass"),
    "obstruction_residue": ("obstruction_residue", "residue", "uap_residue"),
    "collapse_pressure": ("collapse_pressure", "collapse", "classical_collapse_pressure"),
    "classicalization_depth": ("classicalization_depth", "classical_depth", "classicality"),
}


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _first_float(record: Mapping[str, object], aliases: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    for key in aliases:
        val = record.get(key)
        if isinstance(val, (int, float)):
            return _clamp01(float(val))
    profile = record.get("uap_profile") or record.get("shadow_profile") or record.get("profile")
    if isinstance(profile, Mapping):
        for key in aliases:
            val = profile.get(key)
            if isinstance(val, (int, float)):
                return _clamp01(float(val))
    return default


@dataclass(frozen=True)
class ShadowProfileTarget:
    T_support: float
    F_support: float
    glut_mass: float
    gap_mass: float
    obstruction_residue: float
    collapse_pressure: float
    classicalization_depth: float

    def as_vector(self) -> List[float]:
        return [float(getattr(self, f)) for f in PROFILE_FIELDS]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def infer_profile_from_record(record: Mapping[str, object]) -> ShadowProfileTarget:
    """Infer a canonical UAP profile from a JSONL corpus record.

    Explicit profile fields win.  If absent, infer a conservative profile from
    bilateral_t/bilateral_f and kind/probe_type.  This keeps old shards usable
    while v14.3.3 adds supervised profile heads.
    """
    t = _first_float(record, FIELD_ALIASES["T_support"], 0.5)
    f = _first_float(record, FIELD_ALIASES["F_support"], 0.5)
    assert t is not None and f is not None

    kind = str(record.get("kind") or record.get("probe_type") or record.get("family") or "").lower()
    glut_default = min(t, f)
    gap_default = min(1.0 - t, 1.0 - f)

    if "glut" in kind or "contradiction" in kind or "collision" in kind:
        glut_default = max(glut_default, 0.75)
    if "gap" in kind or "underdetermined" in kind or "category" in kind:
        gap_default = max(gap_default, 0.70)

    glut = _first_float(record, FIELD_ALIASES["glut_mass"], glut_default)
    gap = _first_float(record, FIELD_ALIASES["gap_mass"], gap_default)
    assert glut is not None and gap is not None

    obstruction_default = _clamp01(0.5 * glut + 0.35 * gap + 0.15 * abs(t - f))
    residue = _first_float(record, FIELD_ALIASES["obstruction_residue"], obstruction_default)
    assert residue is not None

    # Collapse pressure is low when a record is strongly paraconsistent or gappy;
    # it rises only when one support clearly dominates and obstruction is low.
    collapse_default = _clamp01(max(abs(t - f) - max(glut, gap) * 0.35, 0.0))
    collapse = _first_float(record, FIELD_ALIASES["collapse_pressure"], collapse_default)
    assert collapse is not None

    classical_default = _clamp01(1.0 - max(glut, gap, residue) * 0.7)
    classical = _first_float(record, FIELD_ALIASES["classicalization_depth"], classical_default)
    assert classical is not None

    return ShadowProfileTarget(
        T_support=_clamp01(t),
        F_support=_clamp01(f),
        glut_mass=_clamp01(glut),
        gap_mass=_clamp01(gap),
        obstruction_residue=_clamp01(residue),
        collapse_pressure=_clamp01(collapse),
        classicalization_depth=_clamp01(classical),
    )


def profile_distance(a: ShadowProfileTarget, b: ShadowProfileTarget) -> float:
    av = a.as_vector()
    bv = b.as_vector()
    return sum((x - y) ** 2 for x, y in zip(av, bv)) / len(av)


def support_profile_consistency_score(prompt_profile: ShadowProfileTarget, continuation_profile: ShadowProfileTarget) -> float:
    """0..1 score; higher means prompt/continuation profiles agree."""
    return _clamp01(1.0 - math.sqrt(profile_distance(prompt_profile, continuation_profile)))


def make_profile_tensor(records: Sequence[Mapping[str, object]], device=None, dtype=None):
    """Return [batch, 7] target tensor if torch is available."""
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyTorch is required for make_profile_tensor") from exc
    rows = [infer_profile_from_record(r).as_vector() for r in records]
    return torch.tensor(rows, device=device, dtype=dtype or torch.float32)


def shadow_profile_aux_loss(pred, target, weights: Optional[Sequence[float]] = None):
    """MSE auxiliary loss for predicted profile heads.

    `pred` and `target` are expected to have the final dimension ordered as
    PROFILE_FIELDS.  The helper intentionally does not create the head itself;
    attach it to your model using the hidden state you already expose.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyTorch is required for shadow_profile_aux_loss") from exc
    if pred.shape[-1] != len(PROFILE_FIELDS):
        raise ValueError(f"pred final dim must be {len(PROFILE_FIELDS)} for {PROFILE_FIELDS}")
    if target.shape[-1] != len(PROFILE_FIELDS):
        raise ValueError(f"target final dim must be {len(PROFILE_FIELDS)} for {PROFILE_FIELDS}")
    loss = (pred - target).pow(2)
    if weights is not None:
        w = torch.tensor(list(weights), device=pred.device, dtype=pred.dtype)
        if w.numel() != len(PROFILE_FIELDS):
            raise ValueError("weights length must match PROFILE_FIELDS")
        loss = loss * w
    return loss.mean()


def summarize_profiles(records: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    rows = [infer_profile_from_record(r) for r in records]
    if not rows:
        return {"n": 0.0}
    out: Dict[str, float] = {"n": float(len(rows))}
    for f in PROFILE_FIELDS:
        out[f"mean_{f}"] = sum(getattr(r, f) for r in rows) / len(rows)
    return out
