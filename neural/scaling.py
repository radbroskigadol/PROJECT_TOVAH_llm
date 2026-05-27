"""
TOVAH v14.2.6 neural/scaling.py — Frontier-scale bilateral transformer.

The original ShadowTokenCore (neural/shadow_model.py) preserves the v13
architecture verbatim: byte vocab, standard attention, no RoPE, no GQA,
dimensions hardcoded for ~50M params at the `large` profile. It's the
right design for the bilateral research substrate but it doesn't scale
to frontier sizes.

This module provides ScalableBilateralCore, the bilateral architecture
re-implemented with the modern scaling tricks needed to compete with
frontier models:

  - RoPE (rotary positional encoding): no fixed positional embedding;
    positions encoded by rotating query/key channels. Lets us train at
    one context length and serve at another.
  - GQA (grouped-query attention): keys/values shared across query
    groups; cuts KV-cache memory by group factor. With n_heads=32 and
    n_kv_heads=8 the cache is 4x smaller, enabling longer context.
  - Gradient checkpointing: recompute activations during backward
    instead of storing them. Trades ~30% extra compute for ~50% memory
    savings.
  - Tied input/output embeddings: standard frontier-model trick that
    halves embedding-table memory.
  - RMSNorm (instead of LayerNorm): faster, no learnable bias, matches
    LLaMA/Mistral.
  - SwiGLU FFN: gate * up, with 2/3 hidden width to keep param count.

Bilateral architecture is PRESERVED: every block has parallel T and F
streams with cross-channel mixing in the FFN, dual heads, lane gate.

Frontier profiles in MODEL_PROFILES:
  frontier_1b:   ~1B params (d=2048, n=22, heads=16, kv_heads=8)
  frontier_7b:   ~7B params (d=4096, n=32, heads=32, kv_heads=8)

HONEST ACCOUNTING: at frontier_7b the bilateral architecture doubles
the parameter count vs a single-stream transformer of the same shape.
That's a real research cost. For frontier *compute parity* you'd run
single-stream classical attention with bilateral lane-mixing only at
the FFN and head, which we expose via `bilateral_mode="dual"` (default,
~doubles params) or `bilateral_mode="shared"` (heads-only, near-parity
with classical).

What this module is NOT:
  - A FlashAttention port. Use of FlashAttention is left to torch's
    SDPA fast path (torch>=2.0 picks it automatically on CUDA).
  - A pipeline/tensor-parallel layout. See neural/distributed.py for
    the DDP/FSDP integration points; pipeline-parallel is v15+.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- RoPE ------------------------------------------------------------------

def _rope_freqs(dim: int, max_seq: int, base: float = 10_000.0,
                device: Optional[torch.device] = None,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Precompute RoPE frequencies. Returns shape (max_seq, dim/2) of cis(θ)
    as a complex tensor — but for portability we return real (cos, sin)
    interleaved as shape (max_seq, dim/2, 2)."""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(max_seq, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (max_seq, dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.stack([cos, sin], dim=-1)  # (max_seq, dim/2, 2)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (..., seq, dim) using precomputed freqs
    of shape (seq, dim/2, 2)."""
    # Split into pairs along the last axis.
    *prefix, seq, dim = x.shape
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {dim}")
    x = x.reshape(*prefix, seq, dim // 2, 2)
    f = freqs[:seq].to(dtype=x.dtype, device=x.device)
    cos = f[..., 0]
    sin = f[..., 1]
    x_re = x[..., 0]
    x_im = x[..., 1]
    # Rotate: (x_re + i x_im)(cos + i sin) = (x_re cos - x_im sin) + i (x_re sin + x_im cos)
    new_re = x_re * cos - x_im * sin
    new_im = x_re * sin + x_im * cos
    out = torch.stack([new_re, new_im], dim=-1).reshape(*prefix, seq, dim)
    return out


# --- RMSNorm ---------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


# --- Grouped-query attention with RoPE -------------------------------------

class BilateralGQAttention(nn.Module):
    """Grouped-query attention with RoPE, dual T/F streams.

    Args:
      d_model:    model dim
      n_heads:    number of query heads
      n_kv_heads: number of key/value heads. Must divide n_heads.
                  GQA factor = n_heads / n_kv_heads.
      max_seq:    max sequence length for RoPE precompute
      bilateral_mode:
        "dual"   — separate T-attention and F-attention modules (matches
                   v13 architecture; doubles attention parameters)
        "shared" — single attention path; T/F mixing only at FFN/head
                   (near-parity with classical compute, the practical
                   choice for frontier scaling)
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq: int = 4096, bilateral_mode: str = "dual"):
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {n_kv_heads}")
        if max_seq <= 0:
            raise ValueError(f"max_seq must be > 0, got {max_seq}")
        if bilateral_mode not in {"dual", "shared"}:
            raise ValueError(f"bilateral_mode must be 'dual' or 'shared', got {bilateral_mode!r}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads}")
        if (d_model // n_heads) % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {d_model // n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = self.head_dim * n_kv_heads
        self.bilateral_mode = bilateral_mode

        # Q always per-head; K, V grouped.
        self.q_T = nn.Linear(d_model, d_model, bias=False)
        self.k_T = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_T = nn.Linear(d_model, self.kv_dim, bias=False)
        self.o_T = nn.Linear(d_model, d_model, bias=False)

        if bilateral_mode == "dual":
            self.q_F = nn.Linear(d_model, d_model, bias=False)
            self.k_F = nn.Linear(d_model, self.kv_dim, bias=False)
            self.v_F = nn.Linear(d_model, self.kv_dim, bias=False)
            self.o_F = nn.Linear(d_model, d_model, bias=False)
        # shared mode reuses the same projections for both streams; we just
        # cross-mix at output.

        # Persistent RoPE cache (registered as non-persistent buffer so it
        # doesn't bloat checkpoints).
        freqs = _rope_freqs(self.head_dim, max_seq)
        self.register_buffer("_rope_freqs", freqs, persistent=False)

    def _attn_head(self, x: torch.Tensor, q_proj, k_proj, v_proj, o_proj,
                   mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        if L > self._rope_freqs.shape[0]:
            raise ValueError(
                f"sequence length {L} exceeds RoPE cache length {self._rope_freqs.shape[0]}"
            )
        # Shape to (B, heads, L, head_dim) BEFORE RoPE. _apply_rope()
        # interprets the second-to-last axis as sequence position; applying
        # it while shaped (B, L, heads, dim) would rotate across heads.
        q = q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE on q and k over token positions.
        q = _apply_rope(q, self._rope_freqs)
        k = _apply_rope(k, self._rope_freqs)
        # GQA: expand k, v to n_heads.
        group = self.n_heads // self.n_kv_heads
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
        # SDPA — torch >= 2.0 picks Flash/Memory-Efficient automatically on CUDA.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return o_proj(out)

    def forward(self, T: torch.Tensor, Fv: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        T_out = self._attn_head(T, self.q_T, self.k_T, self.v_T, self.o_T, mask)
        if self.bilateral_mode == "dual":
            F_out = self._attn_head(Fv, self.q_F, self.k_F, self.v_F, self.o_F, mask)
        else:
            F_out = self._attn_head(Fv, self.q_T, self.k_T, self.v_T, self.o_T, mask)
        return T_out, F_out


# --- SwiGLU FFN with bilateral cross-mixing -------------------------------

class BilateralSwiGLU(nn.Module):
    """SwiGLU FFN with bilateral T/F cross-mixing.

    Architecture: gate * up, then down. The cross-mixing at the gate
    lets T and F streams interact (paradox / gap dynamics).
    """
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # T stream.
        self.gate_T = nn.Linear(d_model, d_hidden, bias=False)
        self.up_T = nn.Linear(d_model, d_hidden, bias=False)
        self.down_T = nn.Linear(d_hidden, d_model, bias=False)
        # F stream.
        self.gate_F = nn.Linear(d_model, d_hidden, bias=False)
        self.up_F = nn.Linear(d_model, d_hidden, bias=False)
        self.down_F = nn.Linear(d_hidden, d_model, bias=False)
        # Cross-mix at the gate (small).
        self.cross_T = nn.Linear(d_model, d_hidden, bias=False)
        self.cross_F = nn.Linear(d_model, d_hidden, bias=False)

    def forward(self, T: torch.Tensor, Fv: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # T gate sees a small contribution from F (paradox signal); vice versa.
        gT = F.silu(self.gate_T(T) + 0.1 * self.cross_F(Fv))
        gF = F.silu(self.gate_F(Fv) + 0.1 * self.cross_T(T))
        T_out = self.down_T(gT * self.up_T(T))
        F_out = self.down_F(gF * self.up_F(Fv))
        return T_out, F_out


class BilateralBelnapMoEFFN(nn.Module):
    """Top-k sparse-ish FFN with Belnap-typed experts.

    Experts are ordinary bilateral SwiGLU blocks, but the router sees compact
    A/B/K/G profile features.  This gives the formal taxonomy architectural
    teeth without changing default compute: set ``ffn_kind="belnap_moe"`` to
    enable it.  The implementation is dependency-free and uses dense expert
    evaluation masked by top-k weights; production sparse kernels can replace
    it later without changing the interface.
    """
    def __init__(self, d_model: int, d_hidden: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts must be positive")
        self.n_experts = int(n_experts)
        self.top_k = max(1, min(int(top_k), self.n_experts))
        self.experts = nn.ModuleList([BilateralSwiGLU(d_model, d_hidden) for _ in range(self.n_experts)])
        self.router = nn.Linear(d_model * 2 + 4, self.n_experts, bias=False)

    def _profile(self, T: torch.Tensor, Fv: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(T.mean(dim=-1, keepdim=True))
        f = torch.sigmoid(Fv.mean(dim=-1, keepdim=True))
        A = torch.relu(t - f)
        B = torch.relu(f - t)
        K = torch.minimum(t, f)
        G = 1.0 - torch.maximum(t, f)
        return torch.cat([A, B, K, G], dim=-1)

    def forward(self, T: torch.Tensor, Fv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prof = self._profile(T, Fv)
        logits = self.router(torch.cat([T, Fv, prof], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        if self.top_k < self.n_experts:
            vals, idx = torch.topk(weights, self.top_k, dim=-1)
            mask = torch.zeros_like(weights).scatter_(-1, idx, vals)
            weights = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out_T = torch.zeros_like(T)
        out_F = torch.zeros_like(Fv)
        for i, expert in enumerate(self.experts):
            eT, eF = expert(T, Fv)
            w = weights[..., i].unsqueeze(-1)
            out_T = out_T + w * eT
            out_F = out_F + w * eF
        return out_T, out_F


# --- Block -----------------------------------------------------------------

class ScalableBilateralBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int,
                 n_heads: int, n_kv_heads: int,
                 max_seq: int, bilateral_mode: str = "dual",
                 ffn_kind: str = "swiglu", n_experts: int = 4, moe_top_k: int = 2):
        super().__init__()
        self.attn_norm_T = RMSNorm(d_model)
        self.attn_norm_F = RMSNorm(d_model)
        self.attn = BilateralGQAttention(
            d_model, n_heads, n_kv_heads, max_seq=max_seq,
            bilateral_mode=bilateral_mode,
        )
        self.ffn_norm_T = RMSNorm(d_model)
        self.ffn_norm_F = RMSNorm(d_model)
        if ffn_kind == "swiglu":
            self.ffn = BilateralSwiGLU(d_model, d_hidden)
        elif ffn_kind == "belnap_moe":
            self.ffn = BilateralBelnapMoEFFN(d_model, d_hidden, n_experts=n_experts, top_k=moe_top_k)
        else:
            raise ValueError(f"ffn_kind must be 'swiglu' or 'belnap_moe', got {ffn_kind!r}")

    def forward(self, T: torch.Tensor, Fv: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm residual attention.
        nT = self.attn_norm_T(T)
        nF = self.attn_norm_F(Fv)
        aT, aF = self.attn(nT, nF, mask)
        T = T + aT
        Fv = Fv + aF
        # Pre-norm residual FFN.
        nT = self.ffn_norm_T(T)
        nF = self.ffn_norm_F(Fv)
        fT, fF = self.ffn(nT, nF)
        T = T + fT
        Fv = Fv + fF
        return T, Fv


# --- Full model -----------------------------------------------------------

class ScalableBilateralCore(nn.Module):
    """Frontier-scale bilateral transformer.

    Drop-in shape-compatible with ShadowTokenCore (vocab_size, d_model,
    n_heads, n_blocks, max_len, d_hidden) plus new args (n_kv_heads,
    bilateral_mode, gradient_checkpointing, tied_embeddings).

    Returns (T_logits, F_logits, lane_gate) of shapes (B, L, vocab) for
    drop-in compatibility with the training loop and eval harness. The
    optional frontier-training path can skip F_logits and use compact
    hidden-state semantic heads for K/G auxiliary losses, avoiding full
    B×L×V T/F semantic tensors at 7B/13B scale.
    """
    def __init__(self, *,
                 vocab_size: int = 50_257,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 n_kv_heads: Optional[int] = None,
                 n_blocks: int = 22,
                 max_len: int = 2048,
                 d_hidden: Optional[int] = None,
                 bilateral_mode: str = "dual",
                 gradient_checkpointing: bool = False,
                 tied_embeddings: bool = True,
                 init_bilateral_negation: bool = True,
                 ffn_kind: str = "swiglu",
                 n_experts: int = 4,
                 moe_top_k: int = 2):
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {n_kv_heads}")
        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len}")
        if bilateral_mode not in {"dual", "shared"}:
            raise ValueError(f"bilateral_mode must be 'dual' or 'shared', got {bilateral_mode!r}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads}")
        if (d_model // n_heads) % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {d_model // n_heads}")
        if d_hidden is None:
            # SwiGLU convention: 2.67 * d_model rounded to multiple of 64.
            d_hidden = ((int(d_model * 8 / 3) + 63) // 64) * 64
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_blocks = n_blocks
        self.max_len = max_len
        self.d_hidden = d_hidden
        self.bilateral_mode = bilateral_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.tied_embeddings = tied_embeddings
        self.init_bilateral_negation = bool(init_bilateral_negation)
        self.ffn_kind = str(ffn_kind)
        self.n_experts = int(n_experts)
        self.moe_top_k = int(moe_top_k)

        # Dual embeddings.
        self.embed_T = nn.Embedding(vocab_size, d_model)
        self.embed_F = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            ScalableBilateralBlock(
                d_model=d_model, d_hidden=d_hidden,
                n_heads=n_heads, n_kv_heads=n_kv_heads,
                max_seq=max_len, bilateral_mode=bilateral_mode,
                ffn_kind=ffn_kind, n_experts=n_experts, moe_top_k=moe_top_k,
            )
            for _ in range(n_blocks)
        ])
        self.final_norm_T = RMSNorm(d_model)
        self.final_norm_F = RMSNorm(d_model)

        # Vocabulary heads. v14.3.4 resolves the propositional/positional
        # type confusion in the frontier path: next-token prediction is a
        # single classical vocabulary proposition family, while bilateral
        # truth/falsity support is represented by compact per-position scalar
        # semantic heads below.  The F vocab head remains available only as a
        # diagnostic view of the F hidden stream and shares the same output
        # projection as the T/classical head.
        self.head_T = nn.Linear(d_model, vocab_size, bias=False)
        self.head_F = nn.Linear(d_model, vocab_size, bias=False)

        # 4-lane gate. Same form as ShadowTokenCore.
        self.lane_gate = nn.Linear(d_model * 2, 4)

        # Compact semantic-support heads for frontier training. These produce
        # B×L×1 supports used by K/G auxiliary losses.  They are the primary
        # bilateral semantics for ScalableBilateralCore; full-vocab F logits
        # are diagnostic and should not drive the main paraconsistent loss.
        self.semantic_T = nn.Linear(d_model, 1)
        self.semantic_F = nn.Linear(d_model, 1)

        self.apply(self._init_weights)
        if self.init_bilateral_negation:
            self._init_bilateral_negation()
        self._tie_vocab_projection()
        self._rescale_residual_outputs()

    def _init_weights(self, module: nn.Module) -> None:
        """Frontier-safe GPT/LLaMA-style initialization.

        PyTorch's default nn.Embedding initialization is N(0, 1), which makes
        random-init frontier profiles produce enormous initial logits.  All
        dense/embedding weights start at std=0.02; residual output projections
        are additionally rescaled in _rescale_residual_outputs().
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_bilateral_negation(self) -> None:
        """Initialize F stream as the logical complement of T stream.

        This is a computationally literal UAP/ShadowHoTT starting condition:
        before nonlinear interaction, the two streams begin in bilateral
        negation symmetry rather than unrelated random coordinates.
        """
        with torch.no_grad():
            self.embed_F.weight.copy_(-self.embed_T.weight)
            if hasattr(self, "semantic_F") and hasattr(self, "semantic_T"):
                self.semantic_F.weight.copy_(-self.semantic_T.weight)
                if self.semantic_F.bias is not None:
                    self.semantic_F.bias.zero_()

    def _tie_vocab_projection(self) -> None:
        """Use one shared vocabulary projection for both T and F views.

        When tied_embeddings=True the shared output projection is tied to the
        T embedding table, matching the standard language-model tying pattern.
        F keeps its own input embedding stream but uses the same vocabulary
        projection for diagnostic logits.  When tied_embeddings=False the two
        head modules remain distinct objects but share the same Parameter.
        """
        if self.tied_embeddings:
            self.head_T.weight = self.embed_T.weight
        self.head_F.weight = self.head_T.weight

    def _rescale_residual_outputs(self) -> None:
        """Scale residual-output projections by 1/sqrt(2*n_blocks)."""
        scale = 1.0 / math.sqrt(max(1, 2 * self.n_blocks))
        for block in self.blocks:
            block.attn.o_T.weight.data.mul_(scale)
            if hasattr(block.attn, "o_F"):
                block.attn.o_F.weight.data.mul_(scale)
            if hasattr(block.ffn, "down_T"):
                block.ffn.down_T.weight.data.mul_(scale)
                block.ffn.down_F.weight.data.mul_(scale)
            elif hasattr(block.ffn, "experts"):
                for expert in block.ffn.experts:
                    expert.down_T.weight.data.mul_(scale)
                    expert.down_F.weight.data.mul_(scale)

    def _encode_streams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return final normalized T/F hidden streams without vocab heads."""
        if x.shape[1] > self.max_len:
            x = x[:, -self.max_len:]
        T = self.embed_T(x)
        Fv = self.embed_F(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Recompute activations during backward.
                T, Fv = torch.utils.checkpoint.checkpoint(
                    block, T, Fv, use_reentrant=False,
                )
            else:
                T, Fv = block(T, Fv)
        T = self.final_norm_T(T)
        Fv = self.final_norm_F(Fv)
        return T, Fv

    def semantic_supports_from_hidden(self, T: torch.Tensor, Fv: torch.Tensor
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compact B×L×1 bilateral support estimates from hidden streams."""
        return torch.sigmoid(self.semantic_T(T)), torch.sigmoid(self.semantic_F(Fv))

    def forward(self, x: torch.Tensor, *,
                return_semantic_supports: bool = False,
                semantic_aux_mode: str = "logits",
                skip_f_logits: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Default compatibility mode returns ``(T_logits, F_logits, gate)``;
        both vocab heads share one projection, so F_logits is a diagnostic view
        rather than an independent proposition-level falsity tensor.

        Frontier training should call
        ``forward(x, return_semantic_supports=True, semantic_aux_mode="hidden",
        skip_f_logits=True)`` to receive ``(T_logits, None, gate, T_sup, F_sup)``
        where supports are compact B×L×1 tensors. This keeps CE over the shared
        vocabulary while avoiding full-vocab F logits and full-vocab semantic
        auxiliary tensors.
        """
        if semantic_aux_mode not in {"logits", "hidden"}:
            raise ValueError("semantic_aux_mode must be 'logits' or 'hidden'")
        T, Fv = self._encode_streams(x)
        T_logits = self.head_T(T)
        F_logits = None if skip_f_logits else self.head_F(Fv)
        gate = self.lane_gate(torch.cat([T, Fv], dim=-1))
        if not return_semantic_supports:
            if F_logits is None:
                raise ValueError("skip_f_logits=True requires return_semantic_supports=True")
            return T_logits, F_logits, gate
        if semantic_aux_mode == "hidden":
            T_sup, F_sup = self.semantic_supports_from_hidden(T, Fv)
        else:
            if F_logits is None:
                raise ValueError("logit semantic supports require F_logits")
            T_sup, F_sup = torch.sigmoid(T_logits), torch.sigmoid(F_logits)
        return T_logits, F_logits, gate, T_sup, F_sup

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- Profile builders -----------------------------------------------------

FRONTIER_PROFILES: Dict[str, Dict[str, int]] = {
    # Profiles are NAMED by actual parameter count with bilateral
    # dual-embedding overhead included. At vocab=50,257 with tied
    # embeddings + bilateral_mode='shared':
    #   frontier_2b ~ 2.5B params (target: GPT-2-XL / Llama-3B range)
    #   frontier_7b ~ 7B params   (target: Mistral-7B class)
    #   frontier_13b ~13B params  (target: Llama-13B class)
    # The bilateral T/F dual embeddings cost ~2× vocab × d_model on top
    # of a same-shape classical transformer. Use bilateral_mode='shared'
    # for attention compute-parity with classical at the same param budget.

    # ~2.5B params target. Fits on one 24GB GPU for inference/eval; for
    # training needs grad checkpointing + small batch on consumer GPUs.
    "frontier_2b": {
        "d_model": 2048, "n_heads": 16, "n_kv_heads": 8,
        "n_blocks": 22, "max_len": 2048,
    },
    # ~7B params target. Requires A100 80GB or FSDP for training.
    "frontier_7b": {
        "d_model": 3072, "n_heads": 24, "n_kv_heads": 8,
        "n_blocks": 28, "max_len": 4096,
    },
    # ~13B params target. Multi-GPU. Scaffolding only at this scope.
    "frontier_13b": {
        "d_model": 4096, "n_heads": 32, "n_kv_heads": 8,
        "n_blocks": 32, "max_len": 4096,
    },
    # Smaller dev profile for verifying the scaling path on consumer GPU.
    "frontier_dev": {
        "d_model": 512, "n_heads": 8, "n_kv_heads": 4,
        "n_blocks": 6, "max_len": 1024,
    },
}


def make_scalable_model(profile_name: str = "frontier_2b",
                        *,
                        vocab_size: int = 50_257,
                        bilateral_mode: str = "shared",
                        gradient_checkpointing: bool = True,
                        tied_embeddings: bool = True,
                        init_bilateral_negation: bool = True,
                        ffn_kind: str = "swiglu",
                        n_experts: int = 4,
                        moe_top_k: int = 2,
                        ) -> ScalableBilateralCore:
    """Build a ScalableBilateralCore from a named profile.

    Default `bilateral_mode='shared'` for compute-parity with classical
    transformers at the same parameter budget. Switch to 'dual' if you
    want the full bilateral attention architecture (~10% more params,
    full v13-style separate T-attn and F-attn streams).
    """
    if profile_name not in FRONTIER_PROFILES:
        raise ValueError(
            f"unknown profile {profile_name!r}; "
            f"available: {list(FRONTIER_PROFILES)}"
        )
    cfg = FRONTIER_PROFILES[profile_name]
    return ScalableBilateralCore(
        vocab_size=vocab_size,
        bilateral_mode=bilateral_mode,
        gradient_checkpointing=gradient_checkpointing,
        tied_embeddings=tied_embeddings,
        init_bilateral_negation=init_bilateral_negation,
        ffn_kind=ffn_kind,
        n_experts=n_experts,
        moe_top_k=moe_top_k,
        **cfg,
    )


def estimate_param_count(profile_name: str, vocab_size: int = 50_257,
                         bilateral_mode: str = "shared",
                         tied_embeddings: bool = True) -> int:
    """Estimate parameter count without instantiating the model.

    Useful for capacity-planning before triggering an OOM-prone allocation.
    Estimates within ~5% of actual.
    """
    if profile_name not in FRONTIER_PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}")
    cfg = FRONTIER_PROFILES[profile_name]
    d = cfg["d_model"]
    n = cfg["n_blocks"]
    h = cfg["n_heads"]
    kv = cfg["n_kv_heads"]
    head_dim = d // h
    kv_dim = head_dim * kv

    # Embeddings (dual streams). Tied → embed is shared with head.
    embed = 2 * vocab_size * d  # T + F embeddings
    head = 0 if tied_embeddings else 2 * vocab_size * d

    # Per-block:
    #   Attention: Q (d×d), K (d×kv_dim), V (d×kv_dim), O (d×d)
    #     dual: ×2;  shared: ×1
    attn_one = d * d + d * kv_dim + d * kv_dim + d * d
    attn = (2 if bilateral_mode == "dual" else 1) * attn_one

    # SwiGLU: gate (d×d_hidden), up (d×d_hidden), down (d_hidden×d), per stream
    # Plus cross-mixing: cross_T (d×d_hidden), cross_F (d×d_hidden)
    d_hidden = ((int(d * 8 / 3) + 63) // 64) * 64
    ffn = 2 * (3 * d * d_hidden) + 2 * d * d_hidden

    # Norms: 2 RMSNorm per attn-pair, 2 per ffn-pair. Each is `d` params.
    norms = 4 * d

    per_block = attn + ffn + norms
    total = embed + head + n * per_block
    # Final norms (T, F) + lane_gate + compact semantic heads.
    total += 2 * d + (2 * d) * 4 + (2 * d + 2)
    return total


# --- Frontier memory / capacity estimation ---------------------------------

def dtype_bytes(dtype: str = "bf16") -> int:
    """Return bytes per element for common training dtypes."""
    d = str(dtype).lower()
    if d in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if d in {"fp32", "float32", "float"}:
        return 4
    if d in {"fp8", "float8"}:
        return 1
    raise ValueError(f"unknown dtype {dtype!r}")


def estimate_frontier_memory(
    profile_name: str,
    *,
    vocab_size: int = 50_257,
    bilateral_mode: str = "shared",
    tied_embeddings: bool = True,
    dtype: str = "bf16",
    batch_size: int = 1,
    seq_len: Optional[int] = None,
    world_size: int = 1,
    use_fsdp: bool = False,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """Conservative 13B-readiness memory estimate without model allocation.

    This is not a substitute for a real profiler. It is a launch guard that
    estimates dominant memory terms: parameters, gradients, optimizer state,
    logits, and a coarse activation term. With FSDP, parameter/gradient/optimizer
    terms are divided by world size; activations and logits remain per-rank.
    """
    if profile_name not in FRONTIER_PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}")
    cfg = FRONTIER_PROFILES[profile_name]
    seq_len = int(seq_len or cfg["max_len"])
    if seq_len > cfg["max_len"]:
        raise ValueError(f"seq_len {seq_len} exceeds profile max_len {cfg['max_len']}")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    elem = dtype_bytes(dtype)
    params = estimate_param_count(
        profile_name,
        vocab_size=vocab_size,
        bilateral_mode=bilateral_mode,
        tied_embeddings=tied_embeddings,
    )
    shard = float(world_size if use_fsdp else 1)
    param_bytes = params * elem / shard
    grad_bytes = params * elem / shard
    opt_mult = 2 * 4 if optimizer.lower() == "adamw" else 2 * elem
    optimizer_bytes = params * opt_mult / shard

    d = cfg["d_model"]
    n = cfg["n_blocks"]
    # Coarse activation estimate for T/F hidden streams + residual intermediates.
    # Checkpointing keeps per-block saved activations much smaller.
    act_factor = 8 if gradient_checkpointing else 24
    activation_bytes = batch_size * seq_len * d * elem * act_factor
    # CE still needs T logits. Frontier hidden semantic mode avoids F logits for
    # auxiliary loss, so only one vocab-logit tensor is mandatory.
    t_logits_bytes = batch_size * seq_len * vocab_size * elem
    f_logits_bytes_if_full_aux = t_logits_bytes
    total_no_full_aux = param_bytes + grad_bytes + optimizer_bytes + activation_bytes + t_logits_bytes
    total_full_aux = total_no_full_aux + f_logits_bytes_if_full_aux
    gb = 1024 ** 3
    return {
        "profile": profile_name,
        "parameters": float(params),
        "world_size": float(world_size),
        "fsdp_sharded": bool(use_fsdp),
        "dtype_bytes": float(elem),
        "param_gb_per_rank": param_bytes / gb,
        "grad_gb_per_rank": grad_bytes / gb,
        "optimizer_gb_per_rank": optimizer_bytes / gb,
        "activation_gb_per_rank_est": activation_bytes / gb,
        "t_logits_gb_per_rank": t_logits_bytes / gb,
        "avoidable_f_logits_gb_per_rank": f_logits_bytes_if_full_aux / gb,
        "total_gb_per_rank_hidden_aux_est": total_no_full_aux / gb,
        "total_gb_per_rank_full_vocab_aux_est": total_full_aux / gb,
    }
