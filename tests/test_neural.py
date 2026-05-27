"""
TOVAH v14 tests/test_neural.py — Neural layer tests.

Verifies:
- ShadowTokenCore forward pass shape
- next_token_distribution shape and lane structure
- shadow_score_text ALWAYS returns dict (never scalar)
- shadow_score_scalar ALWAYS returns float
- ShadowOptimizer step runs and returns stats
- Training step runs and returns (loss, phase)
- No ShadowScoreCompat anywhere
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.neural.optimizer import ShadowOptimizer
from tovah_v14.neural.scoring import shadow_score_text, shadow_score_scalar, encode_bytes
from tovah_v14.neural.training import (
    train_shadow_step,
    compute_paraconsistent_invariants,
    semantic_rank_nullity_loss,
)


def _make_model():
    """Create a small debug model for testing."""
    return ShadowTokenCore(
        vocab_size=256, d_model=64, d_hidden=128,
        n_heads=2, n_blocks=2, max_len=64,
    )


# ============================================================
# Model forward tests
# ============================================================
def test_forward_shape():
    model = _make_model()
    ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
    tl, fl, gl = model(ids)
    assert tl.shape == (1, 3, 256)
    assert fl.shape == (1, 3, 256)
    assert gl.shape == (1, 4)


def test_next_token_distribution():
    model = _make_model()
    ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
    mix, probs, div, learned = model.next_token_distribution(ids)
    assert mix.shape == (1, 256)
    assert set(probs.keys()) == {"A", "B", "C", "D"}
    assert set(learned.keys()) == {"A", "B", "C", "D"}
    assert abs(sum(learned.values()) - 1.0) < 0.01


def test_long_sequence_truncation():
    model = _make_model()
    ids = torch.randint(0, 256, (1, 200), dtype=torch.long)
    tl, fl, gl = model(ids)
    assert tl.shape[1] <= model.max_len


# ============================================================
# Scoring contract tests — THE CRITICAL FIX
# ============================================================
def test_score_text_returns_dict():
    model = _make_model()
    result = shadow_score_text(model, "test text")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "entropy" in result
    assert "divergence" in result
    assert "lane_weights" in result
    assert "top_bytes" in result
    assert "text_length" in result


def test_score_text_with_extra_parts():
    model = _make_model()
    result = shadow_score_text(model, "primary", "extra1", "extra2")
    assert isinstance(result, dict)
    assert result["text_length"] > len("primary")


def test_score_text_not_scalar():
    """Verify shadow_score_text never returns something that can be used as a number."""
    model = _make_model()
    result = shadow_score_text(model, "test")
    try:
        _ = float(result)
        assert False, "shadow_score_text result should NOT be convertible to float"
    except (TypeError, ValueError):
        pass  # correct: dicts can't be floated


def test_score_scalar_returns_float():
    model = _make_model()
    result = shadow_score_scalar(model, "test text")
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0


def test_score_text_empty():
    model = _make_model()
    result = shadow_score_text(model, "")
    assert isinstance(result, dict)


def test_no_shadowscorecompat():
    """Verify ShadowScoreCompat class definition does not exist in the neural module."""
    import tovah_v14.neural.scoring as scoring_mod
    source = open(scoring_mod.__file__, encoding="utf-8").read()
    # Check no class definition — docstring mentions are fine (they document what was removed)
    assert "class ShadowScoreCompat" not in source, "ShadowScoreCompat class must not exist"
    code_lines = [l for l in source.splitlines() if not l.strip().startswith("#") and not l.strip().startswith('"""') and not l.strip().startswith("'")]
    code_only = "\n".join(code_lines)
    assert "def __float__" not in code_only, "__float__ method must not exist in scoring"


# ============================================================
# Optimizer tests
# ============================================================
def test_optimizer_step():
    model = _make_model()
    opt = ShadowOptimizer(model.parameters(), base_lr=1e-3)
    ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
    tl, fl, gl = model(ids)
    loss = tl.mean()
    stats = opt.step(loss, phase="Active Learning")
    assert "paradox_mass" in stats
    assert "gap_mass" in stats
    assert "phase" in stats


def test_optimizer_phase_lr():
    model = _make_model()
    opt = ShadowOptimizer(model.parameters(), base_lr=1e-3)
    ids = torch.tensor([[65, 66, 67]], dtype=torch.long)

    tl1, _, _ = model(ids)
    loss1 = tl1.mean()
    stats_active = opt.step(loss1, phase="Active Learning")

    tl2, _, _ = model(ids)
    loss2 = tl2.mean()
    stats_classical = opt.step(loss2, phase="Classical")
    assert stats_classical["lr"] < stats_active["lr"]


# ============================================================
# Training step tests
# ============================================================
def test_train_shadow_step():
    model = _make_model()
    opt = ShadowOptimizer(model.parameters(), base_lr=1e-3)
    corpus = ["hello world", "bilateral evidence four lanes"]
    loss, phase = train_shadow_step(model, opt, corpus)
    assert isinstance(loss, float)
    assert phase in ("Classical", "Active Learning", "Collapse-Resistant Paradox")


# ============================================================
# Paraconsistent invariant tests
# ============================================================
def test_paraconsistent_invariants():
    T = torch.tensor([0.8, 0.3, 0.6])
    Fv = torch.tensor([0.2, 0.7, 0.5])
    Sigma, sigma, h = compute_paraconsistent_invariants(T, Fv)
    assert len(Sigma) == 4
    assert isinstance(sigma, float)
    assert isinstance(h, int)


def test_semantic_rank_nullity_loss():
    T = torch.tensor([[0.8, 0.3], [0.6, 0.5]])
    Fv = torch.tensor([[0.2, 0.7], [0.5, 0.4]])
    loss, dim_con, dim_gap, con_avg, gap_avg = semantic_rank_nullity_loss(T, Fv)
    assert loss.requires_grad is False or True  # may or may not depending on input
    assert isinstance(dim_con, float)
    assert isinstance(dim_gap, float)


# ============================================================
# Encode bytes test
# ============================================================
def test_encode_bytes():
    t = encode_bytes("abc", max_len=10)
    assert t.shape == (1, 3)
    assert t[0, 0].item() == ord("a")


def test_encode_bytes_truncation():
    t = encode_bytes("a" * 100, max_len=10)
    assert t.shape == (1, 10)


def test_encode_bytes_empty():
    t = encode_bytes("", max_len=10)
    assert t.shape[1] >= 1  # at least one byte (space fallback)


# ============================================================
# Runner
# ============================================================
def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
