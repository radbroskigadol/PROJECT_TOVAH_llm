"""
TOVAH v14 tests/test_persistence.py — Persistence layer tests.

Verifies:
- save_json / load_json roundtrip
- State serialization field inventory
- v13 state migration (dict beta coercion, missing fields)
- Snapshot save/cleanup
- Boot validation
- Migration preserves crypto fields
- Migration preserves all v13 data
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.persistence.state_io import save_json, load_json, serialize_state_for_save, save_kernel_ecology_to_file, load_kernel_ecology_from_file, serialize_kernel_ecology_state
from tovah_v14.persistence.migrations import migrate_state, ALL_DEFAULT_BETA_KEYS
from tovah_v14.persistence.boot import validate_boot, BootValidationResult
from tovah_v14.persistence.snapshots import save_branch_checkpoint, load_branch_checkpoint
from tovah_v14.memory.provenance_graph import ProvenanceGraph
from tovah_v14.memory.sync import apply_memory_sync_request
from tovah_v14.memory.store import MemoryStore
from tovah_v14.core.primitives import BilateralValue, coerce_bilateral_value


def _close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ============================================================
# JSON I/O tests
# ============================================================
def test_save_load_json_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.json"
        data = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        assert save_json(path, data)
        loaded = load_json(path)
        assert loaded == data


def test_load_json_missing_file():
    result = load_json(Path("/nonexistent/path.json"), {"default": True})
    assert result == {"default": True}


def test_save_json_atomic():
    """Verify atomic write: no .tmp file left behind on success."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.json"
        save_json(path, {"ok": True})
        tmp = path.with_suffix(".json.tmp")
        assert not tmp.exists()
        assert path.exists()


# ============================================================
# State serialization tests
# ============================================================
def test_serialize_state_has_all_v13_fields():
    """Verify serialize_state_for_save produces all v13 fields."""
    d = serialize_state_for_save(
        completed_goals=[], pending_tool_actions=[], staged_patches={},
        patch_history=[], loss_history=[], research_memory=[], trace_index=[],
        unresolved=[], last_research_time=0.0, improvement_count=0,
        autonomy_level=0, current_goal=None, goal_attempts=0,
        shelved_goals=[], domain_history=[], installed_packages=[],
        state_snapshot={"c": {}, "beta": {}, "nu": {}, "pi": {}},
        alpha=1.0, temperature=0.9, api_usage={}, lab_registry={},
        crypto_wallet=None, beneficiary_sol_address="test_addr",
        profile_name="standard", topic_last_research_time={},
        recent_research_topics=[], active_plans=[], completed_plans=[],
        capabilities={}, rewrite_queue=[], rewrite_history=[],
        memory_episodic=[], memory_semantic=[], memory_procedural=[],
        task_queue=[], completed_tasks=[], failure_clusters=[],
        resource_budgets={}, curriculum=[], promotion_state={},
        workbench_notes={}, state_version="14.0.0", tool_contracts={},
    )
    # Check all v13 fields present
    v13_fields = [
        "completed_goals", "pending_tool_actions", "staged_patches",
        "patch_history", "loss_history", "research_memory", "trace_index",
        "unresolved", "last_research_time", "improvement_count",
        "autonomy_level", "current_goal", "goal_attempts",
        "shelved_goals", "domain_history", "installed_packages",
        "state", "alpha", "temperature", "api_usage", "lab_registry",
        "crypto_wallet", "beneficiary_sol_address", "profile_name",
        "topic_last_research_time", "recent_research_topics",
        "active_plans", "completed_plans", "capabilities",
        "rewrite_queue", "rewrite_history",
        "memory_episodic", "memory_semantic", "memory_procedural",
        "task_queue", "completed_tasks", "failure_clusters",
        "resource_budgets", "curriculum", "promotion_state",
        "workbench_notes", "state_version", "tool_contracts",
    ]
    for field in v13_fields:
        assert field in d, f"missing v13 field: {field}"


# ============================================================
# Migration tests
# ============================================================
def test_migrate_empty_state():
    """Fresh/empty state should migrate without error."""
    d = migrate_state({})
    assert d["state_version"] == "14.0.0"
    assert isinstance(d["state"]["beta"], dict)
    assert len(d["state"]["beta"]) >= len(ALL_DEFAULT_BETA_KEYS)


def test_migrate_v13_state_with_dict_beta():
    """v13 state files may have beta values as plain dicts."""
    v13_state = {
        "state_version": "13.0.0",
        "state": {
            "c": {"active_goal": "test", "last_tool": "", "last_action": "",
                   "cycle": 42, "mode": "local", "paused": False},
            "beta": {
                "goal.active": {"t": 0.9, "f": 0.0},
                "runtime.stability": {"t": 0.7, "f": 0.2},
            },
            "nu": {"goal.active": "T", "runtime.stability": "T"},
            "pi": {"step": 100, "tags": [], "history": [], "refresh_count": 50},
        },
        "improvement_count": 15,
        "crypto_wallet": "test_wallet",
        "beneficiary_sol_address": "De3PtGBEx1XEP7wH6pXbcynSRw9xKNyj8S9k7rG5jRiW",
    }
    d = migrate_state(v13_state)
    assert d["state_version"] == "14.0.0"
    # Beta values coerced
    beta = d["state"]["beta"]
    assert _close(beta["goal.active"]["t"], 0.9)
    assert _close(beta["runtime.stability"]["f"], 0.2)
    # v14 keys added
    assert "tool.discovery_quality" in beta
    assert "module.planner_health" in beta
    # Existing data preserved
    assert d["improvement_count"] == 15
    assert d["crypto_wallet"] == "test_wallet"
    assert d["beneficiary_sol_address"] == "De3PtGBEx1XEP7wH6pXbcynSRw9xKNyj8S9k7rG5jRiW"


def test_migrate_preserves_carrier_fields():
    v13_state = {
        "state": {
            "c": {"active_goal": "research AI", "cycle": 100, "paused": True},
            "beta": {}, "nu": {}, "pi": {},
        },
    }
    d = migrate_state(v13_state)
    c = d["state"]["c"]
    assert c["active_goal"] == "research AI"
    assert c["cycle"] == 100
    assert c["paused"] is True
    assert c["degraded"] is False  # v14 default


def test_migrate_corrupted_beta():
    """Handle corrupted beta entries gracefully."""
    v13_state = {
        "state": {
            "c": {},
            "beta": {
                "good": {"t": 0.8, "f": 0.1},
                "bad_dict": {"garbage": "data"},
                "bad_type": "not_a_dict_or_bv",
                "nan_values": {"t": float("nan"), "f": 0.5},
            },
            "nu": {}, "pi": {},
        },
    }
    d = migrate_state(v13_state)
    beta = d["state"]["beta"]
    assert _close(beta["good"]["t"], 0.8)
    # Corrupted entries get defaults
    assert 0.0 <= beta["bad_dict"]["t"] <= 1.0
    assert 0.0 <= beta["bad_type"]["t"] <= 1.0
    # NaN gets clamped to 0
    import math
    assert math.isfinite(beta["nan_values"]["t"])


def test_migrate_no_data_loss():
    """Migration should never delete existing keys."""
    v13_state = {
        "state_version": "13.0.0",
        "state": {"c": {}, "beta": {"custom_key": {"t": 0.6, "f": 0.3}}, "nu": {}, "pi": {}},
        "improvement_count": 42,
        "completed_goals": ["goal1", "goal2"],
        "lab_registry": {"tool1": {"status": "active"}},
    }
    d = migrate_state(v13_state)
    assert "custom_key" in d["state"]["beta"]
    assert d["improvement_count"] == 42
    assert len(d["completed_goals"]) == 2
    assert "tool1" in d["lab_registry"]


def test_migrate_missing_numeric_fields():
    """Missing numeric fields get proper defaults."""
    d = migrate_state({})
    assert d["alpha"] == 1.0
    assert d["temperature"] == 0.9
    assert d["improvement_count"] == 0
    assert d["autonomy_level"] == 0


# ============================================================
# Boot validation tests
# ============================================================
def test_boot_validation_passes():
    """Boot validation should pass in a normal environment."""
    result = validate_boot()
    assert isinstance(result, BootValidationResult)
    # bilateral_coercion should always work
    assert result.checks.get("bilateral_coercion") is True


def test_boot_validation_structure():
    result = validate_boot()
    assert isinstance(result.checks, dict)
    assert isinstance(result.errors, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.ok, bool)


# ============================================================
# Full roundtrip: serialize → save → load → migrate
# ============================================================
def test_full_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        # Serialize
        d = serialize_state_for_save(
            completed_goals=["g1"], pending_tool_actions=[],
            staged_patches={"p1": {"status": "staged", "target": "research_topic"}},
            patch_history=[], loss_history=[1.0, 0.8],
            research_memory=[], trace_index=[], unresolved=[],
            last_research_time=100.0, improvement_count=5,
            autonomy_level=2, current_goal={"goal": "test"},
            goal_attempts=1, shelved_goals=[], domain_history=["research"],
            installed_packages=["requests"],
            state_snapshot={
                "c": {"active_goal": "test", "cycle": 10},
                "beta": {"k1": {"t": 0.8, "f": 0.1}},
                "nu": {"k1": "T"}, "pi": {"step": 10},
            },
            alpha=1.0, temperature=0.9, api_usage={}, lab_registry={},
            crypto_wallet=None, beneficiary_sol_address="test_addr",
            profile_name="standard", topic_last_research_time={},
            recent_research_topics=[], active_plans=[], completed_plans=[],
            capabilities={}, rewrite_queue=[], rewrite_history=[],
            memory_episodic=[], memory_semantic=[], memory_procedural=[],
            task_queue=[], completed_tasks=[], failure_clusters=[],
            resource_budgets={}, curriculum=[], promotion_state={},
            workbench_notes={}, state_version="13.0.0", tool_contracts={},
        )
        # Save
        assert save_json(path, d)
        # Load
        loaded = load_json(path, {})
        assert loaded["improvement_count"] == 5
        # Migrate
        migrated = migrate_state(loaded)
        assert migrated["state_version"] == "14.0.0"
        assert migrated["improvement_count"] == 5
        assert _close(migrated["state"]["beta"]["k1"]["t"], 0.8)
        # v14 keys present
        assert "module.planner_health" in migrated["state"]["beta"]


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


# ============================================================
# Kernel ecology persistence tests
# ============================================================
def test_kernel_ecology_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ecology.json"
        data = serialize_kernel_ecology_state(
            boot_mode="main_with_hub",
            packet_log=[{"packet_id": "p1", "packet_kind": "status_packet"}],
            child_kernel_registry={"main": {"role": "main"}},
            hub_state={"kernel_id": "hub"},
            subkernel_states={"sub_math": {"kernel_id": "sub_math"}},
            goal_lineage={"g1": {"goal_id": "g1"}},
            module_proposals=[{"proposal_id": "m1"}],
            resource_requests=[{"request_id": "r1"}],
            tool_requests=[{"request_id": "t1"}],
            memory_sync_requests=[{"request_id": "ms1"}],
            promotion_requests=[{"request_id": "p1"}],
            module_registry_state={"proposal_history": []},
            message_bus_state={"routes": []},
            promotion_ladder_state={"state": {}},
            delegation_state={"leases": {}},
            branch_provenance={"nodes": {}, "edges": []},
            branch_checkpoints=[{"path": "x"}],
        )
        assert save_kernel_ecology_to_file(path, data)
        loaded = load_kernel_ecology_from_file(path)
        assert loaded["boot_mode"] == "main_with_hub"
        assert loaded["hub_state"]["kernel_id"] == "hub"
        assert loaded["subkernel_states"]["sub_math"]["kernel_id"] == "sub_math"


def test_branch_checkpoint_save_load():
    with tempfile.TemporaryDirectory() as td:
        meta = save_branch_checkpoint("hub_test", {"ok": True, "depth": 2}, Path(td))
        assert meta["saved"] is True
        payload = load_branch_checkpoint(Path(meta["path"]))
        assert payload["checkpoint"]["ok"] is True
        assert payload["meta"]["branch_name"] == "hub_test"


def test_memory_sync_promote_and_summary_with_provenance():
    store = MemoryStore()
    graph = ProvenanceGraph()
    branch_memory = [
        {"kind": "episodic", "key": "proof_attempt_1", "data": {"result": "partial"}, "tags": ["math"], "goal_context": "prove theorem"},
        {"kind": "semantic", "key": "lemma_cache", "data": {"lemma": "X"}, "tags": ["math"]},
    ]
        
    promote_request = {
        "request_id": "ms_promote",
        "requester_kernel_id": "hub",
        "target_kernel_id": "main",
        "sync_mode": "promote",
        "memory_kinds": ["episodic"],
        "rationale": "promote proof attempt",
    }
    decision, remaining = apply_memory_sync_request(store, graph, promote_request, branch_memory, cycle=3)
    assert decision.promoted_count == 1
    assert decision.promoted_keys == ["proof_attempt_1"]
    assert len(remaining) == 1
    assert any(e.key == "proof_attempt_1" for e in store.get_bank("episodic"))

    summarize_request = {
        "request_id": "ms_summary",
        "requester_kernel_id": "hub",
        "target_kernel_id": "main",
        "sync_mode": "summarize",
        "memory_kinds": ["semantic"],
        "rationale": "keep only summary",
    }
    decision2, remaining2 = apply_memory_sync_request(store, graph, summarize_request, remaining, cycle=4)
    assert decision2.summarized_count == 1
    assert decision2.summary_key
    assert remaining2 == []
    assert graph.summary()["node_count"] >= 3
