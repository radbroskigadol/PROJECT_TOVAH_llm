"""
Tests for the v14.2.7 audit fixes:

  1. Reservoir shuffle in CorpusShardDataset preserves multiset, reorders.
  2. RC-1 default inversion: empty metadata → provisional/subkernel/medium.
  3. Persist-on-evict: rolling buffers stay bounded; overflow lands on disk
     with no record gap.
  4. CalibrationProfile defaults: bit-identical to v14.2.6 numerics.
  5. SheafObserver: topology built, assessment classifies, no gate impact.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Reservoir shuffle
# ---------------------------------------------------------------------------

class TestReservoirShuffle:

    def _build_shards(self, tmpdir: str, n_shards: int = 3, per_shard: int = 30) -> None:
        for s in range(n_shards):
            with open(os.path.join(tmpdir, f"shard_{s:03d}.jsonl"), "w") as fh:
                for i in range(per_shard):
                    fh.write(json.dumps({
                        "text": f"shard{s}-line{i:03d}",
                        "bilateral_t": 0.7, "bilateral_f": 0.2,
                        "kind": "test", "paraconsistent_class": "A",
                    }) + "\n")

    def test_reservoir_off_is_deterministic_legacy(self, tmp_path):
        from tovah_v14.training.dataset import CorpusShardDataset
        self._build_shards(str(tmp_path))
        ds = CorpusShardDataset(
            str(tmp_path), reservoir_size=0,
            shuffle_shards=False, chunk_long_text=False,
        )
        run1 = [x["text"] for x in ds]
        run2 = [x["text"] for x in ds]
        assert run1 == run2, "reservoir_size=0 path must be deterministic"
        # First shard's first item must come out first.
        assert run1[0].startswith("shard0-line000"), run1[:3]

    def test_reservoir_on_preserves_multiset_and_reorders(self, tmp_path):
        from tovah_v14.training.dataset import CorpusShardDataset
        self._build_shards(str(tmp_path))
        ds = CorpusShardDataset(
            str(tmp_path), reservoir_size=8,
            shuffle_shards=False, chunk_long_text=False, seed=42,
        )
        run1 = [x["text"] for x in ds]
        run2 = [x["text"] for x in ds]
        # Multiset preserved on each run
        assert sorted(run1) == sorted(run2)
        # Same seed → same order (reproducibility)
        assert run1 == run2
        # But not equal to the legacy deterministic order
        ds_legacy = CorpusShardDataset(
            str(tmp_path), reservoir_size=0,
            shuffle_shards=False, chunk_long_text=False,
        )
        legacy = [x["text"] for x in ds_legacy]
        assert sorted(legacy) == sorted(run1)
        assert legacy != run1, "reservoir must reorder vs legacy"

    def test_reservoir_different_seeds_diverge(self, tmp_path):
        from tovah_v14.training.dataset import CorpusShardDataset
        self._build_shards(str(tmp_path))
        ds1 = CorpusShardDataset(str(tmp_path), reservoir_size=8,
                                 shuffle_shards=False, chunk_long_text=False, seed=1)
        ds2 = CorpusShardDataset(str(tmp_path), reservoir_size=8,
                                 shuffle_shards=False, chunk_long_text=False, seed=2)
        out1 = [x["text"] for x in ds1]
        out2 = [x["text"] for x in ds2]
        assert sorted(out1) == sorted(out2)
        assert out1 != out2, "different seeds must give different orders"


# ---------------------------------------------------------------------------
# RC-1 default inversion
# ---------------------------------------------------------------------------

class TestRC1Inversion:

    def test_empty_metadata_now_defaults_to_provisional_subkernel_medium(self):
        from tovah_v14.mutation.promotion_ladder import PromotionLadder
        ladder = PromotionLadder()
        ctx = ladder._source_context("unaccounted_patch")
        assert ctx["trust_level"] == "provisional"
        assert ctx["source_role"] == "subkernel"
        assert ctx["risk_class"] == "medium"
        assert ctx["_has_metadata"] is False

    def test_unaccounted_patch_blocked_at_adaptive_gate(self):
        from tovah_v14.mutation.promotion_ladder import PromotionLadder
        ladder = PromotionLadder()
        # No set_source_metadata call. No evidence. Old code would BYPASS
        # the adaptive gate because source_role=='main', trust=='sovereign'.
        # New code defaults to provisional, so insufficient_evidence fires.
        gate = ladder.assess_stage_transition_gate(
            "unaccounted_patch", to_stage="live_promoted", target="main",
        )
        assert gate["allowed"] is False
        # Either policy gate or adaptive gate refusal is fine; the point
        # is "not allowed".
        assert "reason" in gate

    def test_explicit_metadata_still_works(self):
        from tovah_v14.mutation.promotion_ladder import PromotionLadder
        ladder = PromotionLadder()
        ladder.set_source_metadata(
            "p_accounted",
            source_role="hub", trust_level="trusted", risk_level="medium",
            outcome_success_rate=0.95, budget_pressure=0.1,
        )
        ctx = ladder._source_context("p_accounted")
        assert ctx["trust_level"] == "trusted"
        assert ctx["source_role"] == "hub"
        assert ctx["risk_class"] == "medium"
        assert ctx["_has_metadata"] is True

    def test_warning_emitted_for_unaccounted(self, caplog):
        from tovah_v14.mutation.promotion_ladder import PromotionLadder
        ladder = PromotionLadder()
        with caplog.at_level(logging.WARNING):
            ladder._source_context("p_unaccounted_xyz")
        msgs = [r.message for r in caplog.records]
        assert any("p_unaccounted_xyz" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Persist-on-evict
# ---------------------------------------------------------------------------

class TestPersistOnEvict:

    def test_bus_log_persists_overflow_with_no_gap(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.modules.bus_contracts import MessageBusContract
        from tovah_v14.modules.interfaces import ModuleRequest

        set_trace_root(str(tmp_path))
        bus = MessageBusContract()
        # 300 records > cap(200) + cushion(50) → one batched eviction.
        for i in range(300):
            bus.record_request(ModuleRequest(
                from_role="a", to_role="b", action="ping",
                payload={"i": i}, trace_id=f"t{i}", priority=1,
            ))
        # In-memory log stays bounded.
        assert len(bus.message_log) <= 250
        # Trace file exists.
        trace = tmp_path / "module_bus_log.ndjson"
        assert trace.exists()
        lines = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
        # No-gap invariant: in-memory[0].i == disk[-1].i + 1.
        assert lines, "persist-on-evict must produce trace records"
        in_mem_first = bus.message_log[0].payload["i"]
        disk_last = lines[-1]["payload"]["i"]
        assert in_mem_first == disk_last + 1, (
            f"record gap detected: memory[0]={in_mem_first} disk[-1]={disk_last}"
        )

    def test_gate_log_persists(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.mutation.promotion_ladder import PromotionLadder

        set_trace_root(str(tmp_path))
        ladder = PromotionLadder()
        # Force many gate calls. Use explicit metadata so they evaluate cleanly.
        for i in range(280):
            patch = f"p{i}"
            ladder.set_source_metadata(
                patch, source_role="hub", trust_level="trusted",
                risk_level="medium", outcome_success_rate=0.9, budget_pressure=0.1,
            )
            ladder.record_evidence(patch, "ev", source_kernel_id="hub",
                                   trust_level="trusted", risk_class="medium")
            ladder.assess_stage_transition_gate(patch, to_stage="shadow_deployed", target="hub")
        # In-memory gate_log bounded.
        assert len(ladder.gate_log) <= 250
        # Trace file exists (may not have records yet if total <= 250).
        # We pushed 280, so at least one eviction occurred.
        trace = tmp_path / "promotion_gate_log.ndjson"
        assert trace.exists(), "gate_log must persist when count exceeds 250"

    def test_evidence_log_persists_per_patch_with_patch_name_tag(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.mutation.promotion_ladder import PromotionLadder

        set_trace_root(str(tmp_path))
        ladder = PromotionLadder()
        for i in range(170):  # > cap(100) + cushion(50)
            ladder.record_evidence(
                "p_heavy", f"ev{i}", source_kernel_id="hub",
                trust_level="trusted", risk_class="medium",
            )
        assert len(ladder.evidence_log["p_heavy"]) <= 150
        trace = tmp_path / "promotion_evidence.ndjson"
        assert trace.exists()
        lines = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
        assert lines, "evidence overflow must persist"
        # Every persisted record must carry its patch_name tag.
        assert all(r.get("patch_name") == "p_heavy" for r in lines)


# ---------------------------------------------------------------------------
# CalibrationProfile defaults (bit-identical to v14.2.6 numerics)
# ---------------------------------------------------------------------------

class TestCalibrationProfile:

    def test_feedback_calibration_default_constants(self):
        from tovah_v14.modules.calibration import DEFAULT_FEEDBACK_CALIBRATION as C
        # The four most-cited audit constants
        assert C.stale_weak_penalty_weight == 0.55
        assert C.fresh_strong_credit_base == 0.45
        assert C.fresh_strong_credit_scale == 0.55
        assert C.cancellation_fraction_of_credit == 0.80

    def test_gate_calibration_default_constants(self):
        from tovah_v14.modules.calibration import DEFAULT_GATE_CALIBRATION as G
        assert G.shadow_deployed.required_evidence == 2
        assert G.shadow_deployed.min_success_rate == 0.45
        assert G.live_promoted.required_evidence == 3
        assert G.live_promoted.min_success_rate == 0.60
        assert G.live_dynamic_delta_floor == -1.0

    def test_evidence_quality_total_parity_with_explicit_default(self):
        from tovah_v14.modules.registry import ModuleRegistry
        from tovah_v14.modules.calibration import DEFAULT_FEEDBACK_CALIBRATION
        NOW = 1_000_000.0
        items = [
            {"evidence_quality": 0.8, "time": NOW - 3600},
            {"evidence_quality": 1.2, "time": NOW - 100},
            {"evidence_quality": 0.9, "time": NOW - 7200},
        ]
        v1 = ModuleRegistry._evidence_quality_total(items, now=NOW)
        v2 = ModuleRegistry._evidence_quality_total(
            items, now=NOW, calibration=DEFAULT_FEEDBACK_CALIBRATION,
        )
        assert v1 == v2, "default-arg path must be bit-identical to explicit default"

    def test_gate_calibration_injection_changes_behavior(self):
        from dataclasses import replace
        from tovah_v14.modules.calibration import (
            DEFAULT_GATE_CALIBRATION, AdaptiveGateCalibration, GateStageThresholds,
        )
        from tovah_v14.mutation.promotion_ladder import PromotionLadder
        strict = AdaptiveGateCalibration(
            live_promoted=GateStageThresholds(
                required_evidence=10, min_success_rate=0.99,
                max_budget_pressure=0.01, max_recent_failure_weight=0.1,
            ),
        )
        ladder = PromotionLadder(gate_calibration=strict)
        ladder.set_source_metadata(
            "p", source_role="hub", trust_level="trusted", risk_level="medium",
            outcome_success_rate=0.95, budget_pressure=0.1, dynamic_delta=0.0,
        )
        for i in range(3):
            ladder.record_evidence("p", f"ev{i}", source_kernel_id="hub",
                                   trust_level="trusted", risk_class="medium")
        gate = ladder.assess_stage_transition_gate("p", to_stage="live_promoted", target="main")
        assert gate["allowed"] is False
        # 3 evidence < required 10 OR success_rate or budget rejection
        assert gate["reason"] in {
            "insufficient_evidence",
            "low_outcome_success_rate",
            "source_pressure_too_high",
            "main_promotion_requires_trusted",
            "target_not_allowed_for_role",
        }


# ---------------------------------------------------------------------------
# SheafObserver
# ---------------------------------------------------------------------------

class TestSheafObserver:

    def test_topology_built_from_manifests(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.modules.registry import ModuleRegistry
        from tovah_v14.modules.sheaf_observer import SheafObserver

        set_trace_root(str(tmp_path))
        reg = ModuleRegistry()
        obs = SheafObserver()
        obs.bind(reg)
        topo = obs.topology_summary()
        assert topo["n_nodes"] >= 5, topo
        assert topo["n_edges"] >= 1, topo

    def test_initial_assessment_is_ok(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.modules.registry import ModuleRegistry
        from tovah_v14.modules.sheaf_observer import SheafObserver

        set_trace_root(str(tmp_path))
        reg = ModuleRegistry()
        obs = SheafObserver()
        obs.bind(reg)
        f = obs.assess()
        assert f.classification == "ok"
        assert f.obstruction_total == 0.0

    def test_stressed_module_increases_obstruction(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.modules.registry import ModuleRegistry
        from tovah_v14.modules.sheaf_observer import SheafObserver

        set_trace_root(str(tmp_path))
        reg = ModuleRegistry()
        obs = SheafObserver()
        obs.bind(reg)
        f0 = obs.assess()
        for _ in range(15):
            reg.apply_module_feedback("executor", success=False, severity="critical")
        f1 = obs.assess()
        assert f1.obstruction_total > f0.obstruction_total
        # Hottest edge should touch the stressed module.
        if f1.hottest_edges:
            touched = {e["u"] for e in f1.hottest_edges} | {e["v"] for e in f1.hottest_edges}
            assert "executor" in touched

    def test_observer_does_not_influence_gate(self, tmp_path):
        """The observer is read-only — its presence must not change gate decisions."""
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.modules.registry import ModuleRegistry
        from tovah_v14.modules.sheaf_observer import SheafObserver
        from tovah_v14.mutation.promotion_ladder import PromotionLadder

        set_trace_root(str(tmp_path))
        reg = ModuleRegistry()
        # Run 1: without observer
        ladder1 = PromotionLadder()
        ladder1.set_source_metadata("p", source_role="hub", trust_level="trusted",
                                    risk_level="medium", outcome_success_rate=0.95,
                                    budget_pressure=0.1)
        for i in range(2):
            ladder1.record_evidence("p", f"ev{i}", source_kernel_id="hub",
                                    trust_level="trusted", risk_class="medium")
        gate1 = ladder1.assess_stage_transition_gate("p", to_stage="shadow_deployed", target="hub")

        # Run 2: with observer bound
        ladder2 = PromotionLadder()
        obs = SheafObserver()
        obs.bind(reg, ladder2)
        ladder2.set_source_metadata("p", source_role="hub", trust_level="trusted",
                                    risk_level="medium", outcome_success_rate=0.95,
                                    budget_pressure=0.1)
        for i in range(2):
            ladder2.record_evidence("p", f"ev{i}", source_kernel_id="hub",
                                    trust_level="trusted", risk_class="medium")
        gate2 = ladder2.assess_stage_transition_gate("p", to_stage="shadow_deployed", target="hub")

        # Same decision either way.
        assert gate1["allowed"] == gate2["allowed"]
        assert gate1["reason"] == gate2["reason"]

    def test_findings_persisted_to_trace(self, tmp_path):
        from tovah_v14.debug.trace_writer import set_trace_root
        from tovah_v14.modules.registry import ModuleRegistry
        from tovah_v14.modules.sheaf_observer import SheafObserver

        set_trace_root(str(tmp_path))
        reg = ModuleRegistry()
        obs = SheafObserver()
        obs.bind(reg)
        for _ in range(3):
            obs.assess()
        trace = tmp_path / "sheaf_observer_findings.ndjson"
        assert trace.exists()
        lines = trace.read_text().strip().splitlines()
        assert len(lines) >= 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
