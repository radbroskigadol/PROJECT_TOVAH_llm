from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def test_unknown_profile_no_silent_fallback():
    from tovah_v14.training.pretrain import _make_default_model_optimizer
    with pytest.raises(ValueError, match="unknown profile"):
        _make_default_model_optimizer("cpu", "tiny", vocab_size=256)


def test_shadow_optimizer_state_restores_by_parameter_order():
    from tovah_v14.neural.optimizer import ShadowOptimizer

    torch.manual_seed(1)
    m1 = torch.nn.Linear(3, 2)
    opt1 = ShadowOptimizer(m1.parameters())
    x = torch.randn(4, 3)
    loss = m1(x).pow(2).mean()
    loss.backward()
    opt1.step_grads(phase="Classical", loss_value=float(loss.item()))
    state = opt1.state_dict()
    assert "state_by_index" in state

    torch.manual_seed(2)
    m2 = torch.nn.Linear(3, 2)
    opt2 = ShadowOptimizer(m2.parameters())
    opt2.load_state_dict(state)
    st2 = opt2.state_dict()
    assert opt2.t == opt1.t
    assert len(st2["state_by_index"]) == len(state["state_by_index"])
    for a, b in zip(state["state_by_index"], st2["state_by_index"]):
        for key in ("m", "rms2", "K_glut_q", "R_obs_q"):
            if a[key].dtype.is_floating_point:
                assert torch.allclose(a[key], b[key])
            else:
                assert torch.equal(a[key], b[key])


def test_hybrid_optimizer_factory_and_gate_state():
    from tovah_v14.neural.adamw import make_optimizer
    m = torch.nn.Linear(4, 2)
    opt = make_optimizer(m.parameters(), kind="hybrid")
    x = torch.randn(3, 4)
    loss = m(x).pow(2).mean()
    loss.backward()
    stats = opt.step_grads(phase="Active Learning", loss_value=float(loss.item()))
    assert stats["mode"] == "hybrid_adamw_shadow"
    assert 0.0 < stats["adamw_weight"] < 1.0
    assert 0.0 < stats["shadow_weight"] < 1.0
    assert "adamw_score" in stats
    assert "shadow_score" in stats
    assert "hybrid_score_diff" in stats
    # v14.2.9: the gate must not freeze at the initial 0.5/0.5 split.
    assert abs(stats["adamw_weight"] - 0.5) > 1e-7 or abs(stats["shadow_weight"] - 0.5) > 1e-7
    sd = opt.state_dict()
    assert sd["gate256"]["state"].numel() == 8


def test_eval_is_tokenizer_aware(tmp_path):
    from tovah_v14.neural.shadow_model import ShadowTokenCore
    from tovah_v14.training.eval import held_out_perplexity, token_top1_accuracy
    from tovah_v14.training.tokenizer import ByteTokenizer

    shard = tmp_path / "val_eval.jsonl"
    shard.write_text("\n".join(json.dumps({"text": "hello world " * 3}) for _ in range(4)), encoding="utf-8")
    model = ShadowTokenCore(vocab_size=256, d_model=32, d_hidden=64, n_heads=4, n_blocks=1, max_len=32)
    tok = ByteTokenizer()
    ppl = held_out_perplexity(model, tmp_path, tokenizer=tok, val_fraction=1.0, max_examples=3)
    acc = token_top1_accuracy(model, tmp_path, tokenizer=tok, val_fraction=1.0, max_examples=3)
    assert ppl["tokenizer"]["name"] == "byte"
    assert "bits_per_token" in ppl
    assert acc["random_baseline"] == pytest.approx(1 / tok.vocab_size)


def test_uap_shadow_optimizer_exposes_adamw_classicalization_geometry():
    from tovah_v14.neural.optimizer import ShadowOptimizer

    torch.manual_seed(3)
    m = torch.nn.Linear(5, 3)
    opt = ShadowOptimizer(m.parameters(), classical_floor=0.15, geometry_lr=0.01)
    x = torch.randn(6, 5)
    loss = m(x).pow(2).mean()
    loss.backward()
    stats = opt.step_grads(phase="Collapse-Resistant Paradox", loss_value=float(loss.item()))
    assert stats["mode"] == "uap_shadow_hott"
    assert 0.15 <= stats["uap_classical_weight"] <= 0.85
    assert 0.0 <= stats["uap_shadow_weight"] <= 0.85
    assert "uap_obstruction" in stats
    assert "uap_residue_mass" in stats
    assert "uap_trust_ratio_mean" in stats
    sd = opt.state_dict()
    assert sd["optimizer_family"] in {"uap_shadow_hott_v1", "uap_shadow_hott_v14_3_4_compact"}
    assert sd.get("optimizer_state_buffers_per_param", 0) <= 4
    assert "uap_gate" in sd
    assert sd["uap_gate"]["state"].numel() == 8
