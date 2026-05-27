"""
Tests for the v14.1.2 audit fixes.

Each P0/P1 fix gets at least one regression test so subsequent edits
cannot silently re-introduce the original bug.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch


# --- P0-1: semantic loss should be O(1), not O(B*L*V) ----------------------

def test_p0_1_semantic_loss_is_batch_invariant():
    from tovah_v14.neural.training import semantic_rank_nullity_loss
    # Small batch
    T_small = torch.sigmoid(torch.randn(2, 50, 256))
    F_small = torch.sigmoid(torch.randn(2, 50, 256))
    loss_small, *_ = semantic_rank_nullity_loss(T_small, F_small)
    # Big batch — should produce similar magnitude (averages, not sums).
    T_big = torch.sigmoid(torch.randn(32, 320, 256))
    F_big = torch.sigmoid(torch.randn(32, 320, 256))
    loss_big, *_ = semantic_rank_nullity_loss(T_big, F_big)
    # Both should be in O(1) range.
    assert 0.0 < loss_small.item() < 10.0
    assert 0.0 < loss_big.item() < 10.0
    # And the difference between them should be at most a constant factor
    # (no longer scales as numel ratio = 327680 / 25600 = 12.8x).
    assert abs(loss_big.item() / max(1e-6, loss_small.item())) < 5.0


def test_p0_1_semantic_loss_competes_with_task_loss():
    """sem_loss * 0.3 must NOT dominate task_loss (formerly 12,000x ratio)."""
    import torch.nn.functional as F
    from tovah_v14.neural.training import semantic_rank_nullity_loss
    batch, seq, vocab = 4, 200, 256
    logits = torch.randn(batch, seq, vocab)
    targets = torch.randint(0, vocab, (batch, seq))
    task_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
    T = torch.sigmoid(logits)
    Fv = torch.sigmoid(torch.randn_like(logits))
    sem_loss, *_ = semantic_rank_nullity_loss(T, Fv)
    ratio = (0.3 * sem_loss.item()) / max(1e-6, task_loss.item())
    # In v14.1.1 this was ~3600x. In v14.1.2 the regularizer is at most
    # the same order of magnitude as task_loss.
    assert ratio < 1.0, f"sem_loss dominates task_loss by {ratio}x (P0-1 regression)"


# --- P0-2: K-class reachable when both T and F evidence are high ------------

def test_p0_2_k_class_reachable_via_independent_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tovah_v14.config.paths import ensure_directories
    ensure_directories()
    from tovah_v14.selfmodel.experience import ExperienceStore
    es = ExperienceStore()
    rec = es.record(
        "experiment_1", "research",
        context={"text": "evidence for and against the hypothesis"},
        outcome="contradictory",
        truth_evidence=0.8,
        falsity_evidence=0.7,
    )
    # Both T and F should be set independently from the provided evidence.
    assert rec.bilateral_assessment.t == pytest.approx(0.8, abs=1e-6)
    assert rec.bilateral_assessment.f == pytest.approx(0.7, abs=1e-6)
    # And this should classify as K-class.
    assert rec.bilateral_assessment.t >= 0.55
    assert rec.bilateral_assessment.f >= 0.55


def test_p0_2_legacy_reward_still_yields_complementary_tf():
    """When the user gives reward_signal alone, T + F = 1 (legacy behaviour)."""
    from tovah_v14.selfmodel.experience import ExperienceStore
    es = ExperienceStore()
    rec = es.record("legacy", "research", reward_signal=0.6)
    # T+F=1 in legacy mode.
    assert rec.bilateral_assessment.t + rec.bilateral_assessment.f == pytest.approx(1.0, abs=1e-6)


# --- P0-3: dedup collapses across timestamps -------------------------------

def test_p0_3_dedup_collapses_same_record_across_timestamps():
    from tovah_v14.training.corpus_builder import _experience_to_example
    from tovah_v14.training import deduplicate, DedupStrategy
    base = {
        "record_id": "same_id",
        "action_type": "research",
        "context": {"text": "identical body"},
        "outcome": "useful",
        "reward_signal": 0.8,
        "bilateral_assessment": {"t": 0.9, "f": 0.1},
    }
    examples = []
    for ts in (100.0, 200.0, 300.0, 400.0):
        rec = dict(base); rec["created_at"] = ts
        examples.append(_experience_to_example(rec))
    # All four should share one lineage_id (P0-3 fix: time excluded from hash).
    ids = {e.lineage_id for e in examples}
    assert len(ids) == 1, f"P0-3 regression: same record, 4 timestamps, got {len(ids)} ids"
    # And dedup should collapse them.
    unique, stats = deduplicate(examples, strategy=DedupStrategy.KEEP_MOST_RECENT)
    assert len(unique) == 1
    assert stats["duplicates_collapsed"] == 3


def test_p0_3_dedup_keeps_distinct_records_separate():
    from tovah_v14.training.corpus_builder import _experience_to_example
    from tovah_v14.training import deduplicate, DedupStrategy
    examples = []
    for rid in ("a", "b", "c"):
        rec = {"record_id": rid, "action_type": "research",
               "context": {"text": f"body {rid}"}, "outcome": "useful",
               "reward_signal": 0.8, "created_at": 100.0,
               "bilateral_assessment": {"t": 0.9, "f": 0.1}}
        examples.append(_experience_to_example(rec))
    unique, _ = deduplicate(examples, strategy=DedupStrategy.KEEP_BEST_QUALITY)
    assert len(unique) == 3


# --- P0-4: envelope stripped from training text ----------------------------

def test_p0_4_envelope_not_in_training_text():
    from tovah_v14.training.corpus_builder import _experience_to_example
    rec = {
        "record_id": "r1",
        "action_type": "research",
        "context": {"text": "the actual content"},
        "outcome": "useful",
        "reward_signal": 0.5,
        "bilateral_assessment": {"t": 0.8, "f": 0.2},
    }
    ex = _experience_to_example(rec)
    # The text body must not begin with the structural envelope.
    assert not ex.text.startswith("[experience"), \
        f"P0-4 regression: text starts with envelope: {ex.text[:60]!r}"
    # But envelope information must still be available in metadata.
    assert "envelope" in ex.metadata


def test_p0_4_strip_envelope_helper_works():
    from tovah_v14.training import strip_envelope
    assert strip_envelope("[experience kind=research]\nbody") == "body"
    assert strip_envelope("[packet kind=status from=hub to=main]\npayload here") == "payload here"
    # No-op when no envelope.
    assert strip_envelope("just body") == "just body"


def test_p0_4_chunking_helper_splits_long_text():
    from tovah_v14.training.corpus_builder import _chunk_text
    long_text = "x" * 3000
    chunks = _chunk_text(long_text, chunk_bytes=1024, overlap_bytes=128)
    assert len(chunks) >= 3
    # Each chunk fits the byte budget.
    for c in chunks:
        assert len(c.encode("utf-8")) <= 1024


# --- P0-4: max_len raised across all profiles ------------------------------

def test_p0_4_max_len_raised_in_all_profiles():
    from tovah_v14.config.constants import MODEL_PROFILES
    # Every profile must have max_len >= 512 (no more 320-byte cap).
    for name, profile in MODEL_PROFILES.items():
        assert profile["max_len"] >= 512, \
            f"P0-4 regression: profile {name} has max_len={profile['max_len']}"


# --- P0-5: eval harness produces sensible numbers --------------------------

def test_p0_5_held_out_perplexity_returns_finite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tovah_v14.config.paths import ensure_directories, CORPUS_STREAM_DIR
    ensure_directories()
    from tovah_v14.kernel.kernel import ProtozoanKernel
    k = ProtozoanKernel(api={}, is_original=True)
    for i in range(30):
        k.experience_store.record(
            f"e_{i}", "research",
            context={"text": f"sample sample sample {i} content content content"},
            outcome="useful", reward_signal=0.8,
        )
    from tovah_v14.training import held_out_perplexity
    result = held_out_perplexity(
        k.shadow_model, CORPUS_STREAM_DIR,
        val_fraction=1.0,  # treat all available shards as val
        max_examples=20, device=k.device,
    )
    # Either we got valid metrics, or a clean warning string.
    if "warning" in result:
        # Acceptable when only one shard exists (split impossible).
        assert isinstance(result["warning"], str)
    else:
        assert math.isfinite(result["perplexity"])
        assert result["n_tokens"] > 0


def test_p0_5_detect_divergence_flags_nan_and_blowup():
    from tovah_v14.training import detect_divergence
    # NaN.
    out = detect_divergence([1.0, 1.2, 1.1, float("nan")])
    assert out["diverging"] and out["reason"] == "non_finite_loss"
    # Blowup.
    history = [1.0] * 50 + [50.0]
    out = detect_divergence(history)
    assert out["diverging"] and out["reason"] == "blowup"
    # Clean trajectory.
    out = detect_divergence([1.0 - 0.001 * i for i in range(100)])
    assert not out["diverging"]


def test_p0_5_gen_sample_returns_string():
    from tovah_v14.training import gen_sample
    from tovah_v14.neural.shadow_model import ShadowTokenCore
    model = ShadowTokenCore(vocab_size=256, d_model=64, d_hidden=128,
                            n_heads=4, n_blocks=2, max_len=128)
    out = gen_sample(model, "hello ", max_tokens=20, temperature=0.8, device="cpu")
    assert isinstance(out, str) and out.startswith("hello ")


# --- P1-1: tokenizer abstraction --------------------------------------------

def test_p1_1_byte_tokenizer_round_trip():
    from tovah_v14.training import ByteTokenizer
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
    ids = tok.encode("hello world", max_len=64)
    assert len(ids) == 11
    text = tok.decode(ids)
    assert text == "hello world"


def test_p1_1_load_tokenizer_fallback_to_byte():
    from tovah_v14.training import load_tokenizer
    tok = load_tokenizer("byte")
    assert tok.name == "byte"
    tok2 = load_tokenizer("nonexistent_spec_xxx")
    # Falls back to byte for unknown specs.
    assert tok2.name == "byte"


# --- P1-2: dataset streaming + collate -------------------------------------

def test_p1_2_corpus_shard_dataset_yields_examples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tovah_v14.config.paths import ensure_directories, CORPUS_STREAM_DIR
    ensure_directories()
    # Write a fake shard.
    shard_dir = CORPUS_STREAM_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / "tovah_stream_000.jsonl", "w") as fh:
        for i in range(20):
            d = {"text": f"sample {i} body body body content content",
                 "bilateral_t": 0.8 if i % 2 == 0 else 0.3,
                 "bilateral_f": 0.1 if i % 2 == 0 else 0.6,
                 "paraconsistent_class": "A" if i % 2 == 0 else "B",
                 "kind": "experience",
                 "outcome_label": "useful"}
            fh.write(json.dumps(d) + "\n")
    from tovah_v14.training import CorpusShardDataset
    ds = CorpusShardDataset(shard_dir, max_len=128, shuffle_shards=False)
    items = list(iter(ds))
    assert len(items) == 20
    # With class_filter.
    ds_a = CorpusShardDataset(shard_dir, max_len=128, class_filter={"A"},
                              shuffle_shards=False)
    items_a = list(iter(ds_a))
    assert len(items_a) == 10


def test_p1_2_collate_fn_builds_tensor_batches():
    from tovah_v14.training import ByteTokenizer, build_collate_fn
    tok = ByteTokenizer()
    collate = build_collate_fn(tok, max_len=64)
    batch = [{"text": "hello", "bilateral_t": 0.8, "bilateral_f": 0.1,
              "paraconsistent_class": "A", "kind": "experience"},
             {"text": "world wide", "bilateral_t": 0.3, "bilateral_f": 0.6,
              "paraconsistent_class": "B", "kind": "experience"}]
    out = collate(batch)
    assert out["input_ids"].shape[0] == 2
    assert out["target_ids"].shape == out["input_ids"].shape
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["bilateral_t"].shape == (2,)


# --- P1-4: LR schedule -----------------------------------------------------

def test_p1_4_lr_schedule_warms_up_then_decays():
    from tovah_v14.neural.optimizer import ShadowOptimizer
    p = torch.nn.Parameter(torch.randn(4))
    opt = ShadowOptimizer([p], base_lr=2e-4)
    opt.set_schedule(warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    # Manually advance the step counter to probe schedule shape.
    lrs = []
    for step in (1, 5, 10, 25, 50, 100):
        opt.t = step
        lrs.append(opt._scheduled_lr())
    # Warmup: lrs[0] < lrs[1] < lrs[2] == base_lr.
    assert lrs[0] < lrs[1] < lrs[2]
    assert lrs[2] == pytest.approx(2e-4, rel=1e-4)
    # Decay: lrs[2] >= lrs[3] >= lrs[4] >= lrs[5].
    assert lrs[2] >= lrs[3] >= lrs[4] >= lrs[5]
    # Minimum hit at end.
    assert lrs[5] >= 2e-4 * 0.1 - 1e-8
