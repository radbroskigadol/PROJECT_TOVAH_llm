"""
TOVAH v14 neural/scoring.py — Text scoring via ShadowHoTT model.

SEMANTIC FIX (v13 bug resolved):
  In v13, _shadow_score_text got patched into returning ShadowScoreCompat,
  a dict subclass with __float__ and arithmetic operators. This caused
  cascading bugs when callers treated the return as either dict or scalar
  depending on context.

  v14 CONTRACT:
  - shadow_score_text() ALWAYS returns a plain dict.
  - shadow_score_scalar() ALWAYS returns a float.
  - No polymorphic return types. No ShadowScoreCompat. No ambiguity.

RETURN DICT SHAPE:
  {
    "entropy": float,
    "divergence": float,
    "lane_weights": {"A": float, "B": float, "C": float, "D": float},
    "top_bytes": [int, ...],
    "text_length": int,
  }
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from tovah_v14.neural.shadow_model import ShadowTokenCore


def encode_bytes(text: str, max_len: int = 320) -> torch.Tensor:
    """Encode text as byte tensor for ShadowTokenCore input.

    Preserved from v13 _encode_bytes.
    """
    raw = text.encode("utf-8", errors="ignore")[:max_len]
    if not raw:
        raw = b" "
    return torch.tensor(list(raw), dtype=torch.long).unsqueeze(0)


def shadow_score_text(
    model: ShadowTokenCore,
    text: str,
    *extra_parts: str,
    alpha: float = 1.0,
    temperature: float = 0.9,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Score text through the ShadowHoTT model.

    ALWAYS returns a plain dict. Never a scalar.

    Args:
      model: ShadowTokenCore instance
      text: primary text to score
      *extra_parts: additional context strings (concatenated)
      alpha: lane mixing strength
      temperature: softmax temperature
      device: torch device string

    Returns:
      Dict with keys: entropy, divergence, lane_weights, top_bytes, text_length
    """
    merged = " ".join(
        str(x) for x in ([text] + list(extra_parts)) if x is not None
    ).strip()

    try:
        ids = encode_bytes(merged, model.max_len).to(device)
        with torch.no_grad():
            mix, _, div, lw = model.next_token_distribution(
                ids, alpha=alpha, temperature=max(temperature, 1e-6)
            )

        ent = float((-mix * torch.log(mix + 1e-8)).sum(dim=-1).mean().item())
        top_k = min(5, int(mix.shape[-1])) if len(mix.shape) >= 2 else 5
        top_bytes = [int(i) for i in torch.topk(mix[0], k=top_k).indices.tolist()]

        return {
            "entropy": ent,
            "divergence": float(div.mean().item()),
            "lane_weights": lw,
            "top_bytes": top_bytes,
            "text_length": len(merged),
        }

    except Exception as e:
        logging.warning(f"shadow_score_text fallback: {e}")
        return {
            "entropy": 0.0,
            "divergence": 0.0,
            "lane_weights": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "top_bytes": [],
            "text_length": len(merged),
            "fallback": True,
            "error": str(e)[:160],
        }


def shadow_score_scalar(
    model: ShadowTokenCore,
    text: str,
    *extra_parts: str,
    alpha: float = 1.0,
    temperature: float = 0.9,
    device: str = "cpu",
) -> float:
    """Score text and return a single scalar quality metric.

    This is the ONLY way to get a float from scoring.
    Uses entropy and divergence to produce a quality estimate in [0, 1].

    Formula:
      quality = 0.55 + max(0, 3.5 - entropy) * 0.08 + min(0.20, divergence * 0.05)
      clamped to [0, 1]
    """
    metrics = shadow_score_text(
        model, text, *extra_parts,
        alpha=alpha, temperature=temperature, device=device,
    )
    ent = float(metrics.get("entropy", 4.0))
    div = float(metrics.get("divergence", 0.0))
    quality = 0.55 + max(0.0, 3.5 - ent) * 0.08 + min(0.20, div * 0.05)
    return max(0.0, min(1.0, quality))
