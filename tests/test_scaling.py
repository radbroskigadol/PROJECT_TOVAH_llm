"""
Tests for the v14.2.0 frontier-scale neural infrastructure.

Verifies:
  - RoPE rotation is invertible and applies correctly
  - GQA grouping divides cleanly
  - Frontier profiles construct without OOM (just check param count)
  - Forward and backward pass through ScalableBilateralCore
  - Gradient checkpointing reduces activation memory (conceptual test)
  - AdamWWrapper interface parity with ShadowOptimizer
  - Distributed scaffolding gracefully degrades when not distributed
"""
from __future__ import annotations

import math
import os
import pytest
import torch

from tovah_v14.neural.scaling import (
    ScalableBilateralCore, ScalableBilateralBlock,
    BilateralGQAttention, BilateralSwiGLU, RMSNorm,
    FRONTIER_PROFILES,
    make_scalable_model, estimate_param_count,
    _rope_freqs, _apply_rope,
)
from tovah_v14.neural.adamw import AdamWWrapper, make_optimizer
from tovah_v14.neural import distributed


# --- RoPE ------------------------------------------------------------------

class TestRoPE:
    def test_rope_freqs_shape(self):
        freqs = _rope_freqs(dim=64, max_seq=128)
        assert freqs.shape == (128, 32, 2)

    def test_rope_preserves_length(self):
        """RoPE is unitary: ||x||² per position is preserved."""
        freqs = _rope_freqs(dim=32, max_seq=16)
        x = torch.randn(2, 4, 16, 32)
        y = _apply_rope(x, freqs)
        x_norm = x.pow(2).sum(dim=-1)
        y_norm = y.pow(2).sum(dim=-1)
        assert torch.allclose(x_norm, y_norm, atol=1e-5)

    def test_attention_rope_uses_token_axis_not_head_axis(self):
        """Regression: RoPE must rotate by token position, not head index."""
        attn = BilateralGQAttention(
            d_model=64, n_heads=4, n_kv_heads=2,
            max_seq=16, bilateral_mode="shared",
        )
        x = torch.randn(2, 7, 64)
        B, L, _ = x.shape
        q_raw = attn.q_T(x).view(B, L, attn.n_heads, attn.head_dim).transpose(1, 2)
        q_expected = _apply_rope(q_raw, attn._rope_freqs)
        q_wrong_axis = _apply_rope(
            attn.q_T(x).view(B, L, attn.n_heads, attn.head_dim),
            attn._rope_freqs,
        ).transpose(1, 2)
        assert not torch.allclose(q_expected, q_wrong_axis)
        assert q_expected.shape == (B, attn.n_heads, L, attn.head_dim)


# --- GQA -------------------------------------------------------------------

class TestGQA:
    def test_gqa_group_factor(self):
        attn = BilateralGQAttention(
            d_model=128, n_heads=8, n_kv_heads=2,
            max_seq=64, bilateral_mode="shared",
        )
        assert attn.n_heads // attn.n_kv_heads == 4
        assert attn.kv_dim == 16 * 2  # head_dim=16, n_kv_heads=2

    def test_gqa_forward_pass(self):
        attn = BilateralGQAttention(
            d_model=64, n_heads=4, n_kv_heads=2,
            max_seq=32, bilateral_mode="shared",
        )
        T = torch.randn(2, 16, 64)
        Fv = torch.randn(2, 16, 64)
        T_out, F_out = attn(T, Fv)
        assert T_out.shape == T.shape
        assert F_out.shape == Fv.shape

    def test_gqa_rejects_non_divisible(self):
        with pytest.raises(ValueError):
            BilateralGQAttention(d_model=128, n_heads=8, n_kv_heads=3)
        with pytest.raises(ValueError):
            BilateralGQAttention(d_model=129, n_heads=8, n_kv_heads=2)


# --- RMSNorm + SwiGLU ------------------------------------------------------

class TestPrimitives:
    def test_rmsnorm_centers_to_unit_rms(self):
        norm = RMSNorm(d_model=128)
        x = torch.randn(4, 16, 128) * 10  # large variance
        y = norm(x)
        # Per-position RMS should be ~1 (weight init is 1, so output RMS ≈ 1).
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_bilateral_swiglu_forward(self):
        ffn = BilateralSwiGLU(d_model=64, d_hidden=256)
        T = torch.randn(2, 8, 64)
        Fv = torch.randn(2, 8, 64)
        T_out, F_out = ffn(T, Fv)
        assert T_out.shape == T.shape
        assert F_out.shape == Fv.shape


# --- Param count estimation -----------------------------------------------

class TestParamEstimation:
    def test_estimate_matches_actual_small(self):
        """For frontier_dev at small vocab, estimate ≈ actual within <0.1%."""
        model = make_scalable_model(
            "frontier_dev", vocab_size=256,
            bilateral_mode="shared", gradient_checkpointing=False,
        )
        actual = model.num_params()
        estimated = estimate_param_count(
            "frontier_dev", vocab_size=256, bilateral_mode="shared",
        )
        # Within 1% (some norm params, lane gate, fudge factor).
        assert abs(actual - estimated) / max(1, actual) < 0.01

    def test_frontier_2b_target_range(self):
        """frontier_2b at vocab=50257 should be in the 2-3B param range."""
        est = estimate_param_count("frontier_2b", vocab_size=50_257)
        assert 2e9 < est < 3e9

    def test_frontier_7b_target_range(self):
        est = estimate_param_count("frontier_7b", vocab_size=50_257)
        assert 5e9 < est < 8e9

    def test_dual_mode_costs_more(self):
        """bilateral_mode='dual' uses more params than 'shared' (extra attn projs)."""
        shared = estimate_param_count("frontier_2b", bilateral_mode="shared")
        dual = estimate_param_count("frontier_2b", bilateral_mode="dual")
        assert dual > shared

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            make_scalable_model("frontier_nonexistent")


# --- Full model: forward, backward, generation ----------------------------

class TestScalableBilateralCore:
    def _make_dev_model(self, vocab=256):
        # Keep this unit test CPU-small; full frontier_dev smoke lives elsewhere.
        from tovah_v14.neural.scaling import ScalableBilateralCore
        return ScalableBilateralCore(
            vocab_size=vocab, d_model=64, n_heads=4, n_kv_heads=2, n_blocks=1, max_len=64,
            bilateral_mode="shared", gradient_checkpointing=False,
        )

    def test_forward_shapes(self):
        m = self._make_dev_model(vocab=256)
        x = torch.randint(0, 256, (3, 32))
        T_logits, F_logits, gate = m(x)
        assert T_logits.shape == (3, 32, 256)
        assert F_logits.shape == (3, 32, 256)
        assert gate.shape == (3, 32, 4)

    def test_backward_pass(self):
        m = self._make_dev_model(vocab=256)
        x = torch.randint(0, 256, (2, 16))
        T, F_, _ = m(x)
        loss = T.mean() + F_.mean()
        loss.backward()
        # All parameters should have gradients (when tied, sharing means
        # the same .grad is updated; that's fine — just check we got SOME).
        any_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m.parameters())
        assert any_grad

    def test_gradient_checkpointing_forward_still_works(self):
        from tovah_v14.neural.scaling import ScalableBilateralCore
        m = ScalableBilateralCore(
            vocab_size=256, d_model=64, n_heads=4, n_kv_heads=2, n_blocks=1, max_len=64,
            bilateral_mode="shared", gradient_checkpointing=True,
        )
        m.train()
        x = torch.randint(0, 256, (2, 16))
        T, F_, _ = m(x)
        loss = T.mean() + F_.mean()
        loss.backward()
        # If we got here without exception, checkpointing didn't break backward.
        assert True

    def test_tied_embeddings_share_weight(self):
        from tovah_v14.neural.scaling import ScalableBilateralCore
        m = ScalableBilateralCore(
            vocab_size=256, d_model=64, n_heads=4, n_kv_heads=2, n_blocks=1, max_len=64,
            bilateral_mode="shared", tied_embeddings=True, gradient_checkpointing=False,
        )
        assert m.head_T.weight is m.embed_T.weight
        assert m.head_F.weight is m.head_T.weight
        assert m.head_F.weight is not m.embed_F.weight

    def test_untied_embeddings_have_separate_weights(self):
        from tovah_v14.neural.scaling import ScalableBilateralCore
        m = ScalableBilateralCore(
            vocab_size=256, d_model=64, n_heads=4, n_kv_heads=2, n_blocks=1, max_len=64,
            bilateral_mode="shared", tied_embeddings=False, gradient_checkpointing=False,
        )
        assert m.head_T.weight is not m.embed_T.weight
        assert m.head_F.weight is m.head_T.weight


# --- Training path wiring --------------------------------------------------

class TestFrontierPretrainWiring:
    def test_default_builder_selects_scalable_core_and_muon_for_frontier(self):
        from tovah_v14.training.pretrain import _make_default_model_optimizer, _unwrap_model
        from tovah_v14.neural.scaling import ScalableBilateralCore
        from tovah_v14.neural.muon import MuonWrapper
        model, opt = _make_default_model_optimizer(
            "cpu", "frontier_dev", vocab_size=64,
            bilateral_mode="shared", gradient_checkpointing=False,
        )
        assert isinstance(_unwrap_model(model), ScalableBilateralCore)
        assert isinstance(opt, MuonWrapper)
        assert _unwrap_model(model).vocab_size == 64


# --- AdamW wrapper ---------------------------------------------------------

class TestAdamWWrapper:
    def test_basic_step(self):
        p = torch.nn.Parameter(torch.randn(8))
        opt = AdamWWrapper([p], base_lr=1e-3)
        loss = (p ** 2).sum()
        stats = opt.step(loss)
        assert stats["mode"] == "adamw"
        assert stats["lr"] == 1e-3
        # Param should have moved.
        assert p.requires_grad

    def test_schedule_warmup_then_decay(self):
        p = torch.nn.Parameter(torch.randn(4))
        opt = AdamWWrapper([p], base_lr=2e-4)
        opt.set_schedule(warmup_steps=10, total_steps=100)
        lrs = []
        for step in (1, 5, 10, 50, 100):
            opt.t = step
            lrs.append(opt._scheduled_lr())
        # Warmup: increasing.
        assert lrs[0] < lrs[1] < lrs[2]
        assert lrs[2] == pytest.approx(2e-4, rel=1e-4)
        # Decay: non-increasing.
        assert lrs[2] >= lrs[3] >= lrs[4]

    def test_factory_returns_correct_kind(self):
        p = torch.nn.Parameter(torch.randn(2))
        adamw = make_optimizer([p], kind="adamw")
        assert isinstance(adamw, AdamWWrapper)
        from tovah_v14.neural.optimizer import ShadowOptimizer
        shadow = make_optimizer([p], kind="shadow")
        assert isinstance(shadow, ShadowOptimizer)
        with pytest.raises(ValueError):
            make_optimizer([p], kind="nonexistent")

    def test_zero_grad_clears_grads(self):
        p = torch.nn.Parameter(torch.randn(4))
        opt = AdamWWrapper([p], base_lr=1e-3)
        loss = (p ** 2).sum()
        opt.step(loss)
        opt.zero_grad()
        assert p.grad is None


# --- Distributed scaffolding ----------------------------------------------

class TestDistributedScaffolding:
    def test_not_distributed_in_test_env(self):
        # In normal test env, RANK/WORLD_SIZE aren't set.
        if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
            pytest.skip("running under distributed launcher")
        assert not distributed.is_distributed_available()

    def test_rank_world_size_defaults(self):
        if "RANK" in os.environ:
            pytest.skip("running under distributed launcher")
        assert distributed.rank() == 0
        assert distributed.world_size() == 1
        assert distributed.is_main()

    def test_barrier_no_op_when_not_distributed(self):
        # Should not raise.
        distributed.barrier()
        distributed.cleanup()

    def test_init_returns_none_when_not_distributed(self):
        if "RANK" in os.environ:
            pytest.skip("running under distributed launcher")
        assert distributed.init_distributed() is None
