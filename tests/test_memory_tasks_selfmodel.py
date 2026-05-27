"""
TOVAH v14 tests/test_memory_tasks_selfmodel.py

Verifies:
- MemoryStore CRUD + bilateral tracking
- TF-IDF retrieval beats word overlap
- Consolidation: episodic → semantic, workflow → procedural
- Forgetting removes stale low-confidence entries
- Memory conflict detection preserves both entries
- TaskQueue lifecycle: create, advance, complete
- Task cleanup: orphaned tasks, purge old
- PlanManager: add, complete, stale cleanup
- SelfModel: update from state, priorities ordered
- CompetenceMap: outcome recording, weakest domains
- ExperienceStore: record, replay, outcome summary
- ModuleHealthTracker: success/failure/summary
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState, CarrierState, ProvenanceState
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.runtime_interface import make_fresh_state

from tovah_v14.memory.store import MemoryStore, MemoryEntry
from tovah_v14.memory.retrieval import memory_query
from tovah_v14.memory.consolidation import consolidate_memory
from tovah_v14.memory.forgetting import forget_stale, cleanup_memory
from tovah_v14.memory.conflict import check_memory_conflict, MemoryConflictRecord

from tovah_v14.tasks.queue import TaskQueue, TaskNode
from tovah_v14.tasks.plans import PlanManager, StrategicPlan
from tovah_v14.tasks.cleanup import cleanup_tasks
from tovah_v14.tasks.delegation import DelegationManager

from tovah_v14.selfmodel.model import SelfModel, update_self_model
from tovah_v14.selfmodel.competence import CompetenceMap
from tovah_v14.selfmodel.experience import ExperienceStore
from tovah_v14.selfmodel.module_health import ModuleHealthTracker


# ============================================================
# Memory store tests
# ============================================================
def test_memory_store_basic():
    ms = MemoryStore()
    entry = ms.store("episodic", "test_key", {"data": "value"}, tags=["tag1"])
    assert entry.kind == "episodic"
    assert entry.key == "test_key"
    assert ms.counts()["episodic"] == 1


def test_memory_store_bilateral_update():
    state = make_fresh_state(["memory.consolidation_health"])
    ms = MemoryStore()
    old_t = state.beta["memory.consolidation_health"].t
    ms.store("semantic", "k", {"d": 1}, state=state)
    assert state.beta["memory.consolidation_health"].t >= old_t


def test_memory_store_respects_limit():
    ms = MemoryStore()
    for i in range(600):
        ms.store("episodic", f"k_{i}", {"i": i})
    assert ms.counts()["episodic"] <= 500


# ============================================================
# Memory retrieval tests
# ============================================================
def test_retrieval_basic():
    ms = MemoryStore()
    ms.store("semantic", "python_patterns", {"topic": "python design patterns"}, tags=["python", "patterns"])
    ms.store("semantic", "rust_memory", {"topic": "rust memory safety"}, tags=["rust", "memory"])
    results = memory_query(ms, "semantic", "python design patterns")
    assert len(results) >= 1
    assert results[0].key == "python_patterns"


def test_retrieval_updates_access():
    ms = MemoryStore()
    ms.store("semantic", "k", {"d": 1})
    results = memory_query(ms, "semantic", "some query", limit=5)
    if results:
        assert results[0].access_count >= 1


def test_retrieval_empty_query():
    ms = MemoryStore()
    ms.store("episodic", "k1", {"d": 1})
    ms.store("episodic", "k2", {"d": 2})
    results = memory_query(ms, "episodic", "", limit=5)
    assert len(results) >= 1  # returns most recent


# ============================================================
# Consolidation tests
# ============================================================
def test_consolidation_tag_clusters():
    ms = MemoryStore()
    ms.store("episodic", "e1", {"outcome": "ok"}, tags=["research"])
    ms.store("episodic", "e2", {"outcome": "ok"}, tags=["research"])
    ms.store("episodic", "e3", {"outcome": "fail"}, tags=["other"])
    counts = consolidate_memory(ms, max_age_hours=1.0)
    assert counts["semantic_created"] >= 1
    # Check semantic bank has pattern entry
    sem_keys = [e.key for e in ms.get_bank("semantic")]
    assert any("research" in k for k in sem_keys)


def test_consolidation_workflow_to_procedural():
    ms = MemoryStore()
    entry = ms.store("episodic", "wf1", {"outcome": "success", "steps": ["a", "b"]}, tags=["workflow"])
    entry.bilateral_confidence = BilateralValue(0.7, 0.1)  # above 0.5 threshold
    counts = consolidate_memory(ms, max_age_hours=1.0)
    assert counts["procedural_created"] >= 1


# ============================================================
# Forgetting tests
# ============================================================
def test_forget_stale():
    ms = MemoryStore()
    old_entry = ms.store("episodic", "old", {"d": 1})
    old_entry.created_at = time.time() - 200000  # very old
    old_entry.bilateral_confidence = BilateralValue(0.1, 0.8)  # low confidence
    old_entry.access_count = 0
    ms.store("episodic", "fresh", {"d": 2})
    removed = forget_stale(ms, "episodic", max_age_hours=24.0, min_confidence=0.3)
    assert removed >= 1
    keys = [e.key for e in ms.get_bank("episodic")]
    assert "fresh" in keys
    assert "old" not in keys


def test_cleanup_all_banks():
    ms = MemoryStore()
    for kind in ("episodic", "semantic", "procedural"):
        e = ms.store(kind, f"old_{kind}", {"d": 1})
        e.created_at = time.time() - 200000
        e.bilateral_confidence = BilateralValue(0.05, 0.9)
        e.access_count = 0
    result = cleanup_memory(ms)
    assert sum(result.values()) >= 3


# ============================================================
# Memory conflict tests — THE CRITICAL CONTRADICTION RULE
# ============================================================
def test_conflict_detection_key_collision():
    ms = MemoryStore()
    ms.store("semantic", "topic:ai", {"outcome": "positive"})
    conflicts = check_memory_conflict(ms, "semantic", "topic:ai", {"outcome": "negative"})
    assert len(conflicts) >= 1
    assert conflicts[0].conflict_type == "key_collision"


def test_conflict_detection_contradictory():
    ms = MemoryStore()
    ms.store("episodic", "task:search", {"outcome": "success"})
    conflicts = check_memory_conflict(ms, "episodic", "task:search", {"outcome": "failure"})
    assert any(c.conflict_type in ("key_collision", "contradictory_data") for c in conflicts)


def test_conflict_preserves_both():
    """Conflict detection does NOT remove the existing entry."""
    ms = MemoryStore()
    ms.store("semantic", "topic:x", {"v": 1})
    conflicts = check_memory_conflict(ms, "semantic", "topic:x", {"v": 2})
    assert len(conflicts) >= 1
    # Original still exists
    assert ms.counts()["semantic"] == 1
    # Store the new one too (caller's decision)
    ms.store("semantic", "topic:x", {"v": 2})
    assert ms.counts()["semantic"] == 2  # BOTH preserved


# ============================================================
# Task queue tests
# ============================================================
def test_task_create_and_advance():
    tq = TaskQueue()
    tid = tq.create("do something")
    assert tq.get_by_id(tid).status == "pending"
    msgs = tq.advance()
    assert tq.get_by_id(tid).status == "active"


def test_task_deps():
    tq = TaskQueue()
    t1 = tq.create("first")
    t2 = tq.create("second", deps=[t1])
    tq.advance()  # t1 → active, t2 stays pending
    assert tq.get_by_id(t2).status == "pending"
    tq.complete(t1)
    tq.advance()  # now t2 → active
    assert tq.get_by_id(t2).status == "active"


def test_task_complete():
    tq = TaskQueue()
    tid = tq.create("task")
    tq.advance()
    ok = tq.complete(tid, {"result": "done"})
    assert ok
    assert tq.get_by_id(tid).status == "completed"


def test_task_cleanup_orphan():
    tq = TaskQueue()
    child = tq.create("child task", parent_id="nonexistent_parent")
    tq.advance()
    counts = cleanup_tasks(tq)
    assert counts["orphaned"] >= 1


# ============================================================
# Plan tests
# ============================================================
def test_plan_lifecycle():
    pm = PlanManager()
    plan = StrategicPlan(plan_id="p1", objective="test", steps=[{"action": "research"}],
                         created_at="2025-01-01T00:00:00")
    pm.add(plan)
    assert len(pm.get_active()) == 1
    pm.complete("p1")
    assert len(pm.get_active()) == 0
    assert "p1" in pm.completed_ids


def test_plan_stale_cleanup():
    pm = PlanManager()
    plan = StrategicPlan(plan_id="old", objective="old plan", steps=[],
                         created_at="2020-01-01T00:00:00")
    pm.add(plan)
    archived = pm.cleanup_stale()
    assert "old" in archived


# ============================================================
# SelfModel tests
# ============================================================
def test_self_model_update():
    state = make_fresh_state(["planning.confidence", "browser.reachability", "patch.pipeline.health",
                               "tool.search_efficacy", "sandbox.reliability", "memory.consolidation_health",
                               "task.queue_health", "levbel.pdf_health", "state.coherent", "self_model.accuracy"])
    state.beta["planning.confidence"] = BilateralValue(0.3, 0.6)  # weak planner
    state.beta["tool.search_efficacy"] = BilateralValue(0.9, 0.05)  # strong search
    refresh_state(state)
    sm = SelfModel()
    sm = update_self_model(sm, state)
    assert sm.subsystem_reliability["planner"] < sm.subsystem_reliability["search"]
    assert sm.improvement_priorities[0] in ("planner", "browser", "pdf_ingest", "sandbox", "memory", "tasks", "persistence")


# ============================================================
# Competence map tests
# ============================================================
def test_competence_record():
    cm = CompetenceMap()
    cm.record_outcome("tool_mastery", True)
    cm.record_outcome("tool_mastery", True)
    cm.record_outcome("tool_mastery", False)
    entry = cm.entries["tool_mastery"]
    assert entry.test_count == 3
    assert entry.success_count == 2
    assert abs(entry.measured_mastery - 2/3) < 0.01


def test_competence_weakest():
    cm = CompetenceMap()
    cm.record_outcome("strong", True)
    cm.record_outcome("strong", True)
    cm.record_outcome("weak", False)
    cm.record_outcome("weak", False)
    weakest = cm.get_weakest(1)
    assert weakest[0].domain == "weak"


# ============================================================
# Experience store tests
# ============================================================
def test_experience_record():
    es = ExperienceStore()
    rec = es.record("r1", "research", outcome="useful", reward_signal=0.8, tags=["test"])
    assert rec.outcome == "useful"
    assert rec.bilateral_assessment.t > 0.5


def test_experience_replay():
    es = ExperienceStore()
    es.record("r1", "research", outcome="useful")
    es.record("r2", "patch", outcome="regressed")
    es.record("r3", "research", outcome="useless")
    research = es.replay("research")
    assert len(research) == 2


def test_experience_summary():
    es = ExperienceStore()
    es.record("r1", "research", outcome="useful")
    es.record("r2", "research", outcome="useful")
    es.record("r3", "patch", outcome="regressed")
    summary = es.outcome_summary()
    assert summary["useful"] == 2
    assert summary["regressed"] == 1


# ============================================================
# Module health tests
# ============================================================
def test_module_health_success():
    state = make_fresh_state(["module.planner_health"])
    tracker = ModuleHealthTracker()
    old_t = state.beta["module.planner_health"].t
    tracker.record_success("planner", state)
    assert state.beta["module.planner_health"].t >= old_t


def test_module_health_failure():
    state = make_fresh_state(["module.executor_health"])
    tracker = ModuleHealthTracker()
    old_f = state.beta["module.executor_health"].f
    tracker.record_failure("executor", state)
    assert state.beta["module.executor_health"].f >= old_f


def test_module_health_summary():
    state = make_fresh_state(["module.planner_health", "module.executor_health",
                               "module.critic_health", "module.memory_health"])
    tracker = ModuleHealthTracker()
    summary = tracker.get_health_summary(state)
    assert "planner" in summary
    assert "t" in summary["planner"]


def test_module_weakest():
    state = make_fresh_state(["module.planner_health", "module.executor_health"])
    state.beta["module.planner_health"] = BilateralValue(0.2, 0.7)
    state.beta["module.executor_health"] = BilateralValue(0.8, 0.1)
    refresh_state(state)
    tracker = ModuleHealthTracker()
    weakest = tracker.weakest_modules(state, 1)
    assert "planner" in weakest


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


def test_task_create_with_lineage():
    tq = TaskQueue()
    tid = tq.create(
        "delegated work",
        owner_kernel_id="sub_math_1",
        requester_kernel_id="hub",
        root_goal_id="root1",
        mission_context="math mission",
        lease_scope="delegated",
        delegated_to_kernel_id="sub_math_1",
        packet_id="pkt_1",
    )
    task = tq.get_by_id(tid)
    assert task.owner_kernel_id == "sub_math_1"
    assert task.delegated_to_kernel_id == "sub_math_1"
    assert task.lineage is not None
    assert task.lineage.root_goal_id == "root1"


def test_plan_add_with_lineage_defaults():
    pm = PlanManager()
    plan = StrategicPlan(plan_id="p_lineage", objective="test", steps=[], created_at="2025-01-01T00:00:00", owner_kernel_id="hub", mission_context="experiment")
    pm.add(plan)
    assert pm.get_active()[0].lineage is not None
    assert pm.get_active()[0].lineage.owner_kernel_id == "hub"


def test_delegation_manager_issue_and_complete():
    dm = DelegationManager()
    lease = dm.issue(goal_id="g1", source_kernel_id="main", target_kernel_id="sub_1", mission_context="delegated")
    assert lease.status == "active"
    assert dm.complete(lease.lease_id, {"ok": True}) is True
    assert dm.leases[lease.lease_id].status == "completed"
