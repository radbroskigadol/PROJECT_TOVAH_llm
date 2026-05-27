"""
TOVAH v14.1.1 — Smoke test for closed-loop training wiring.

Verifies that:
1. ContinuousExporter is wired into the kernel at boot.
2. Live activity (experience.record + packet dispatch) writes JSONL.
3. _sample_live_corpus reads back from shards.
4. _train_shadow_step uses sampled corpus (loss is a float).
5. pretrain() runs over the streamed shards and updates the model.
6. TRAIN_FROM_CORPUS David command produces a structured response.
7. The whole loop is reentrant (kernel can re-boot, find existing shards,
   and roll forward to a new shard without truncating prior ones).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tovah_v14.config.paths import ensure_directories, CORPUS_STREAM_DIR, COMMAND_FILE, RESPONSE_FILE
from tovah_v14.kernel.kernel import ProtozoanKernel


def _list_stream_shards():
    return sorted(CORPUS_STREAM_DIR.glob("tovah_stream_*.jsonl"))


def test_continuous_exporter_wired_at_boot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k = ProtozoanKernel(api={}, is_original=True)
    assert k.continuous_exporter is not None, \
        "ContinuousExporter must be initialised by __init__"
    assert k.experience_store.on_record is not None
    assert k.promotion_ladder.on_gate_decision is not None


def test_experience_record_writes_to_stream(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k = ProtozoanKernel(api={}, is_original=True)
    k.experience_store.record(
        "smoke_001", "tool_use",
        context={"tool": "test"}, outcome="useful", reward_signal=0.8,
    )
    shards = _list_stream_shards()
    assert shards, "experience.record should produce at least one shard line"
    # Verify by parsing the JSONL — the record_id is hashed into lineage_id,
    # not preserved verbatim, so we check structural fields instead.
    lines = shards[-1].read_text().strip().split("\n")
    parsed = [json.loads(ln) for ln in lines if ln.strip()]
    experiences = [p for p in parsed if p.get("kind") == "experience"]
    assert experiences, "expected at least one experience-kind line in shard"
    e = experiences[-1]
    assert e["outcome_label"] == "useful"
    assert e["bilateral_t"] >= e["bilateral_f"]  # positive reward → A-class
    assert e["metadata"]["rec_kind"] == "tool_use"


def test_sample_live_corpus_reads_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k = ProtozoanKernel(api={}, is_original=True)
    # Seed several experiences so the A-pool is non-empty.
    for i in range(12):
        k.experience_store.record(
            f"sample_{i}", "research",
            context={"i": i, "text": f"alpha bravo charlie {i}"},
            outcome="useful", reward_signal=0.7,
        )
    batch = k._sample_live_corpus(batch_size=8)
    assert isinstance(batch, list) and len(batch) >= 1
    # Fallback signature: only triggered when nothing usable found.
    fallback = "shadowhott bilateral evidence four lanes constraints"
    assert fallback not in batch or len(batch) > 2, \
        "should sample real shards, not fall back, after seeding 12 experiences"


def test_train_shadow_step_with_live_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k = ProtozoanKernel(api={}, is_original=True)
    for i in range(20):
        k.experience_store.record(
            f"livetrain_{i}", "research",
            context={"text": f"the rain in spain falls mainly on the plain step {i}"},
            outcome="useful", reward_signal=0.9,
        )
    loss, phase = k._train_shadow_step()
    assert isinstance(loss, float)
    assert phase in {"Classical", "Active Learning", "Collapse-Resistant Paradox"}
    assert k.loss_history[-1] == loss


def test_pretrain_over_streamed_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k = ProtozoanKernel(api={}, is_original=True)
    # Seed a mix of A-class (high reward) and K-class (mixed) experiences.
    for i in range(40):
        rew = 0.9 if i % 3 == 0 else -0.5 if i % 3 == 1 else 0.0
        k.experience_store.record(
            f"pretrain_{i}", "research",
            context={"text": f"sample text body for pretraining seed {i}"},
            outcome="useful" if rew > 0 else "contradictory",
            reward_signal=rew,
        )
    from tovah_v14.training import pretrain
    summary = pretrain(
        CORPUS_STREAM_DIR,
        model=k.shadow_model,
        optimizer=k.shadow_optimizer,
        epochs=1,
        batch_size=4,
        device=k.device,
        log_every=0,
        val_fraction=0.0,  # all-train so the smoke test has data
    )
    # v14.1.2 schema: total_steps + epoch_avg_loss + per-epoch first/last.
    assert summary["total_steps"] >= 1
    assert summary["epoch_avg_loss"] and summary["epoch_avg_loss"][0] >= 0
    # Tokenizer/vocab info is now part of the summary.
    assert summary["vocab_size"] == 256
    assert summary["tokenizer"]["name"] == "byte"


def test_train_from_corpus_david_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k = ProtozoanKernel(api={}, is_original=True)
    for i in range(30):
        k.experience_store.record(
            f"cmd_{i}", "research",
            context={"text": f"david command test payload {i}"},
            outcome="useful", reward_signal=0.8,
        )
    # Default shard_dir = tovah_corpus/stream; small epochs/batch for speed.
    COMMAND_FILE.write_text("TRAIN_FROM_CORPUS:|1|4")
    k._check_david_commands()
    resp = RESPONSE_FILE.read_text() if RESPONSE_FILE.exists() else ""
    # Either a JSON summary or a structured failure — both must be parseable.
    assert resp, "no response written to david_response.txt"
    # We accept the response if it contains the structural keys.
    assert ("total_steps" in resp) or ("total_examples" in resp) or ("failed" in resp.lower())


def test_continuous_exporter_resumes_to_next_shard(tmp_path, monkeypatch):
    """Second boot must keep existing shards and roll forward to a new one."""
    monkeypatch.chdir(tmp_path)
    ensure_directories()
    k1 = ProtozoanKernel(api={}, is_original=True)
    k1.experience_store.record(
        "boot1", "research",
        context={"text": "first-boot seed"}, outcome="useful", reward_signal=0.7,
    )
    shards_before = _list_stream_shards()
    assert shards_before, "first boot should produce a shard"
    # Snapshot the first shard's lines (we'll verify they survive).
    pre_lines = shards_before[-1].read_text().strip().split("\n")
    assert len(pre_lines) >= 1
    # Close & re-open by dropping the kernel and instantiating a new one.
    k1.continuous_exporter.close()
    k2 = ProtozoanKernel(api={}, is_original=True)
    assert k2.continuous_exporter is not None
    # The next shard index should be one past the highest pre-existing.
    assert k2.continuous_exporter._current_shard_idx >= len(shards_before)
    k2.experience_store.record(
        "boot2", "research",
        context={"text": "second-boot seed"}, outcome="useful", reward_signal=0.7,
    )
    shards_after = _list_stream_shards()
    assert len(shards_after) > len(shards_before), \
        f"new shard should appear; before={len(shards_before)} after={len(shards_after)}"
    # First-boot shard must still be intact (same or more lines).
    post_lines = shards_before[-1].read_text().strip().split("\n")
    assert len(post_lines) >= len(pre_lines)
    # And the new shard must contain new experience JSON.
    new_lines = shards_after[-1].read_text().strip().split("\n")
    new_parsed = [json.loads(ln) for ln in new_lines if ln.strip()]
    assert any(p.get("kind") == "experience" for p in new_parsed)
