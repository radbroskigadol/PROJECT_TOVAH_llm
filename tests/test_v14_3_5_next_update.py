import torch

from tovah_v14.neural.muon import MuonWrapper, zeropower_via_newtonschulz
from tovah_v14.neural.scaling import ScalableBilateralCore, make_scalable_model
from tovah_v14.training.formal_hott_rl import (
    default_pi_sigma_id_tasks, positive_candidate_strings, reward_task_candidate,
    smoke_score_suite, grpo_advantages,
)
from tovah_v14.training.sheaf_regularizer import sequence_sheaf_obstruction_loss
from tovah_v14.training.bilateral_lora_adapter import BilateralDoRALinear


def test_bilateral_negation_init_and_moe_forward():
    m = ScalableBilateralCore(vocab_size=64, d_model=32, n_heads=4, n_kv_heads=2, n_blocks=2, max_len=16, ffn_kind="belnap_moe", n_experts=4, moe_top_k=2)
    assert torch.allclose(m.embed_F.weight, -m.embed_T.weight, atol=1e-6)
    x = torch.randint(0, 64, (2, 8))
    tl, fl, gate, ts, fs = m(x, return_semantic_supports=True, semantic_aux_mode="hidden")
    assert tl.shape == (2, 8, 64)
    assert fl.shape == (2, 8, 64)
    assert gate.shape == (2, 8, 4)
    assert ts.shape == (2, 8, 1)
    assert fs.shape == (2, 8, 1)


def test_muon_shape_lr_and_nesterov_smoke():
    p = torch.nn.Parameter(torch.randn(16, 8) * 0.01)
    opt = MuonWrapper([p], base_lr=1e-3)
    p.grad = torch.randn_like(p)
    stats = opt.step_grads()
    assert stats["optimizer_family"] == "muon_v14_3_5"
    assert stats["nesterov"] is True
    assert stats["ns_steps"] == 3
    assert stats["matrix_updates"] == 1


def test_formal_hott_rl_parser_rewards():
    tasks = default_pi_sigma_id_tasks()
    seeds = positive_candidate_strings()
    accepted = 0
    for task in tasks:
        r = reward_task_candidate(task, seeds[task.name])
        accepted += int(r.accepted)
    smoke = smoke_score_suite(tasks)
    assert accepted >= 7
    assert smoke["reward_mean"] > 0.75
    adv = grpo_advantages([[reward_task_candidate(tasks[0], seeds[tasks[0].name]), reward_task_candidate(tasks[0], "bad")]])[0]
    assert adv.numel() == 2


def test_sheaf_regularizer_and_dora_shapes():
    T = torch.rand(2, 5, 1)
    F = torch.rand(2, 5, 1)
    loss, stats = sequence_sheaf_obstruction_loss(T, F, bilateral_t=torch.tensor([1.0, 0.8]), bilateral_f=torch.tensor([0.0, 0.8]))
    assert loss.ndim == 0
    assert stats["sheaf_edges"] > 0
    base = torch.nn.Linear(4, 3)
    dora = BilateralDoRALinear(base, r=2)
    y = dora(torch.randn(2, 4), gate_T=torch.ones(2, 1), gate_F=torch.zeros(2, 1))
    assert y.shape == (2, 3)
