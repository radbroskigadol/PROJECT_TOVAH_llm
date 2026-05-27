from __future__ import annotations

import torch


def test_frontier_init_is_small_and_logits_are_sane():
    from tovah_v14.neural.scaling import make_scalable_model

    torch.manual_seed(123)
    model = make_scalable_model("frontier_dev", vocab_size=128, bilateral_mode="shared", gradient_checkpointing=False)
    assert 0.015 <= float(model.embed_T.weight.std().item()) <= 0.025
    assert float(model.embed_T.weight.abs().max().item()) < 0.20
    x = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        t_logits, f_logits, _gate = model(x)
    assert float(t_logits.abs().mean().item()) < 2.0
    assert float(f_logits.abs().mean().item()) < 2.0
    assert model.head_F.weight is model.head_T.weight


def test_shadow_optimizer_uses_four_compact_buffers():
    from tovah_v14.neural.optimizer import ShadowOptimizer

    lin = torch.nn.Linear(4, 3)
    opt = ShadowOptimizer(lin.parameters())
    loss = lin(torch.randn(5, 4)).pow(2).mean()
    loss.backward()
    opt.step_grads(loss_value=float(loss.item()))
    sd = opt.state_dict()
    assert sd["optimizer_state_buffers_per_param"] == 4
    st = sd["state_by_index"][0]
    assert set(st) == {"m", "rms2", "K_glut_q", "R_obs_q"}
    assert st["K_glut_q"].dtype == torch.uint8
    assert st["R_obs_q"].dtype == torch.uint8


def test_formal_hott_reward_suite_accepts_known_witnesses():
    from tovah_v14.training.formal_hott_rl import smoke_score_suite

    result = smoke_score_suite()
    assert result["accepted"] == result["n"]
    assert result["reward_mean"] == 1.0


def test_repetition_penalty_is_training_noop():
    from tovah_v14.training.loop_stability import repetition_penalty_from_logits

    logits = torch.randn(2, 3, 11, requires_grad=True)
    penalty = repetition_penalty_from_logits(logits, strength=999.0)
    assert float(penalty.detach().item()) == 0.0


def test_muon_factory_steps():
    from tovah_v14.neural.adamw import make_optimizer

    lin = torch.nn.Linear(4, 4)
    opt = make_optimizer(lin.parameters(), kind="muon")
    loss = lin(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    stats = opt.step_grads(loss_value=float(loss.item()))
    assert stats["mode"] == "muon"
    assert stats["matrix_updates"] >= 1
