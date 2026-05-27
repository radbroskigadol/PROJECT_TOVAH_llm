"""
TOVAH v14 neural/shadow_model.py — ShadowTokenCore bilateral transformer.

SEMANTIC PRESERVATION:
  Every layer, dimension, forward pass formula, and next_token_distribution
  computation is identical to v13. No architectural changes.

Architecture:
  - Byte-level (vocab_size=256)
  - Dual embeddings: embed_T, embed_F, pos_T, pos_F
  - N BilateralBlocks, each containing:
    - BilateralAttention (dual multi-head attention + layernorm)
    - BilateralFFN (dual feedforward with cross-channel mixing + layernorm)
  - Dual output heads: head_T, head_F
  - Lane gate: learned 4-lane mixture weights
  - Causal mask for autoregressive generation
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualLinear(nn.Module):
    """Dual linear layer: separate T and F projections."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.T = nn.Linear(in_f, out_f)
        self.F = nn.Linear(in_f, out_f)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.T(x), self.F(x)


class BilateralAttention(nn.Module):
    """Dual-channel multi-head attention with independent T and F streams."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn_T = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_F = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_T = nn.LayerNorm(d_model)
        self.norm_F = nn.LayerNorm(d_model)

    def forward(
        self,
        xT: torch.Tensor,
        xF: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        yT, _ = self.attn_T(xT, xT, xT, attn_mask=attn_mask, need_weights=False)
        yF, _ = self.attn_F(xF, xF, xF, attn_mask=attn_mask, need_weights=False)
        return self.norm_T(xT + yT), self.norm_F(xF + yF)


class BilateralFFN(nn.Module):
    """Dual-channel feedforward with cross-channel mixing.

    The 0.35 cross-channel coefficient and 0.15 residual mixing
    are preserved exactly from v13.
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.inp = DualLinear(d_model, d_hidden)
        self.out = DualLinear(d_hidden, d_model)
        self.norm_T = nn.LayerNorm(d_model)
        self.norm_F = nn.LayerNorm(d_model)

    def forward(
        self, xT: torch.Tensor, xF: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aT, aF = self.inp(xT)
        bT, bF = self.inp(xF)
        midT = F.gelu(aT) + 0.35 * F.gelu(bF)
        midF = F.gelu(aF) + 0.35 * F.gelu(bT)
        oT, _ = self.out(midT)
        _, pF = self.out(midF)
        return (
            self.norm_T(xT + oT + 0.15 * pF),
            self.norm_F(xF + pF + 0.15 * oT),
        )


class BilateralBlock(nn.Module):
    """One bilateral transformer block: attention + FFN."""

    def __init__(self, d_model: int, d_hidden: int, n_heads: int):
        super().__init__()
        self.attn = BilateralAttention(d_model, n_heads)
        self.ffn = BilateralFFN(d_model, d_hidden)

    def forward(
        self,
        xT: torch.Tensor,
        xF: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xT, xF = self.attn(xT, xF, mask)
        return self.ffn(xT, xF)


class ShadowTokenCore(nn.Module):
    """ShadowHoTT bilateral byte-level transformer.

    SEMANTIC PRESERVATION: architecture identical to v13.

    Parameters:
      vocab_size: 256 (byte-level)
      d_model, d_hidden, n_heads, n_blocks: from MODEL_PROFILES
      max_len: maximum sequence length

    Forward returns: (t_logits, f_logits, gate_logits)
    next_token_distribution returns: (mix, probs, div, learned_weights)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 224,
        d_hidden: int = 896,
        n_heads: int = 7,
        n_blocks: int = 5,
        max_len: int = 320,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_T = nn.Embedding(vocab_size, d_model)
        self.embed_F = nn.Embedding(vocab_size, d_model)
        self.pos_T = nn.Embedding(max_len, d_model)
        self.pos_F = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [BilateralBlock(d_model, d_hidden, n_heads) for _ in range(n_blocks)]
        )
        self.head_T = nn.Linear(d_model, vocab_size)
        self.head_F = nn.Linear(d_model, vocab_size)
        self.lane_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 4)
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, L = token_ids.shape
        if L > self.max_len:
            token_ids = token_ids[:, -self.max_len :]
            L = token_ids.shape[1]
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        xT = self.embed_T(token_ids) + self.pos_T(pos)
        xF = self.embed_F(token_ids) + self.pos_F(pos)
        mask = self.causal_mask[:L, :L]
        for blk in self.blocks:
            xT, xF = blk(xT, xF, mask)
        return self.head_T(xT), self.head_F(xF), self.lane_gate(
            torch.cat([xT[:, -1, :], xF[:, -1, :]], dim=-1)
        )

    @torch.no_grad()
    def next_token_distribution(
        self,
        token_ids: torch.Tensor,
        alpha: float = 1.0,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
        """Compute next-token distribution across all four lanes.

        Returns:
          mix: weighted lane mixture distribution
          probs: per-lane probability distributions
          div: lane divergence tensor
          learned: learned lane weights {"A": float, ...}

        FORMULA (preserved exactly from v13).
        """
        t_logits, f_logits, gate_logits = self.forward(token_ids)
        t = torch.sigmoid(t_logits[:, -1, :])
        f = torch.sigmoid(f_logits[:, -1, :])
        diff = t_logits[:, -1, :] - f_logits[:, -1, :]
        scores = {
            "A": diff + alpha * (t * (1 - f) - f * (1 - t)),
            "B": diff + alpha * (torch.maximum(t, f) - f * (1 - t)),
            "C": diff
            + alpha * (t * (1 - f) - torch.maximum(f, 1 - t) * (1 - t * f)),
            "D": diff + alpha * (t - (1 - t)),
        }
        probs = {
            k: F.softmax(v / max(temperature, 1e-6), dim=-1) for k, v in scores.items()
        }
        gate = F.softmax(gate_logits, dim=-1)
        names = ["A", "B", "C", "D"]
        learned = {names[i]: float(gate[0, i].item()) for i in range(4)}
        mix = sum(probs[k] * learned[k] for k in names)
        div = sum(
            learned[k]
            * torch.sum(
                probs[k] * (torch.log(probs[k] + 1e-8) - torch.log(mix + 1e-8)),
                dim=-1,
            )
            for k in names
        )
        return mix, probs, div, learned
