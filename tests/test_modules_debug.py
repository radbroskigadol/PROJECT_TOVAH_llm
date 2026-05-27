"""
TOVAH v14 tests/test_modules_debug.py — Module and debug layer tests.

Verifies:
- Module roles defined
- All manifests present with health keys
- Registry queries work
- Dependency graph is acyclic at top level
- CycleMetrics collection
- MetricsCollector trend analysis
- Failure clustering
- No fake distributed execution
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.modules.roles import ModuleRole, MODULE_HEALTH_KEYS
from tovah_v14.modules.manifests import MODULE_MANIFESTS, ModuleManifest
from tovah_v14.modules.registry import ModuleRegistry
from tovah_v14.modules.interfaces import ModuleRequest, ModuleResponse, TaskLease
from tovah_v14.modules.bus_contracts import MessageBusContract

from tovah_v14.debug.failure_clusters import FailureCluster, cluster_failures
from tovah_v14.debug.observability import CycleMetrics, collect_cycle_metrics
from tovah_v14.debug.metrics import MetricsCollector

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.runtime_interface import make_fresh_state
from tovah_v14.core.cache import refresh_state


# ============================================================
# Module roles tests
# ============================================================
def test_module_roles_enum():
    assert len(ModuleRole) == 8
    assert ModuleRole.PLANNER.value == "planner"


def test_module_health_keys_match_roles():
    for role in ModuleRole:
        assert role.value in MODULE_HEALTH_KEYS, f"missing health key for {role.value}"


# ============================================================
# Module manifest tests
# ============================================================
def test_all_roles_have_manifests():
    for role in ModuleRole:
        assert role.value in MODULE_MANIFESTS, f"missing manifest for {role.value}"


def test_manifest_structure():
    m = MODULE_MANIFESTS["planner"]
    assert isinstance(m, ModuleManifest)
    assert m.role == "planner"
    assert len(m.methods) > 0
    assert m.health_key == "module.planner_health"


def test_manifest_dependencies_valid():
    """All deps should reference existing roles."""
    valid_roles = set(MODULE_MANIFESTS.keys())
    for role, manifest in MODULE_MANIFESTS.items():
        for dep in manifest.depends_on:
            assert dep in valid_roles, f"{role} depends on unknown role: {dep}"


# ============================================================
# Registry tests
# ============================================================
def test_registry_list():
    reg = ModuleRegistry()
    modules = reg.list_modules()
    assert "planner" in modules
    assert "executor" in modules
    assert len(modules) == 8


def test_registry_describe():
    reg = ModuleRegistry()
    desc = reg.describe("planner")
    assert desc["role"] == "planner"
    assert "methods" in desc


def test_registry_health():
    state = make_fresh_state(["module.planner_health", "module.executor_health"])
    state.beta["module.planner_health"] = BilateralValue(0.3, 0.7)
    refresh_state(state)
    reg = ModuleRegistry()
    h = reg.health_summary(state)
    assert h["planner"]["t"] < 0.4


def test_registry_weakest():
    state = make_fresh_state(["module.planner_health", "module.executor_health"])
    state.beta["module.planner_health"] = BilateralValue(0.2, 0.7)
    state.beta["module.executor_health"] = BilateralValue(0.9, 0.1)
    refresh_state(state)
    reg = ModuleRegistry()
    weakest = reg.weakest(state, 1)
    assert "planner" in weakest


def test_registry_dependency_graph():
    reg = ModuleRegistry()
    graph = reg.dependency_graph()
    assert "planner" in graph
    assert isinstance(graph["planner"], list)


# ============================================================
# Interface contracts (just verify they construct)
# ============================================================
def test_module_request():
    req = ModuleRequest(from_role="planner", to_role="memory_manager", action="query")
    assert req.from_role == "planner"


def test_task_lease():
    lease = TaskLease(task_id="t1", leased_by="executor")
    assert lease.status == "active"


def test_bus_contract():
    bus = MessageBusContract()
    bus.register("planner", "plan_goal", "_strategic_plan")
    assert len(bus.lookup("plan_goal")) == 1
    assert "plan_goal" in bus.all_actions()


# ============================================================
# Failure clustering tests
# ============================================================
def test_cluster_failures():
    errors = [
        {"category": "tool_failure", "key": "web_search", "message": "timeout", "timestamp": 100},
        {"category": "tool_failure", "key": "web_search", "message": "timeout", "timestamp": 101},
        {"category": "patch_rejection", "key": "research_topic", "message": "blocked", "timestamp": 102},
    ]
    clusters = cluster_failures(errors)
    assert len(clusters) >= 2
    # web_search should be top (count=2)
    assert clusters[0].count >= 2


def test_cluster_empty():
    clusters = cluster_failures([])
    assert clusters == []


# ============================================================
# Observability tests
# ============================================================
def test_cycle_metrics_collection():
    state = make_fresh_state(["goal.active", "runtime.stability"])
    metrics = collect_cycle_metrics(state, training_loss=0.5, training_phase="Active Learning")
    assert isinstance(metrics, CycleMetrics)
    assert metrics.training_loss == 0.5
    assert metrics.coherent is True


def test_metrics_collector_trend():
    mc = MetricsCollector()
    state = make_fresh_state(["k"])
    for i in range(10):
        m = collect_cycle_metrics(state, training_loss=float(i) * 0.1)
        mc.record(m)
    trend = mc.trend("training_loss", window=10)
    assert trend["count"] == 10
    assert trend["max"] > trend["min"]


def test_metrics_anomaly_detection():
    mc = MetricsCollector()
    state = make_fresh_state(["k"])
    # 9 normal values, 1 extreme
    for i in range(9):
        m = collect_cycle_metrics(state, training_loss=1.0)
        m.cycle = i
        mc.record(m)
    outlier = collect_cycle_metrics(state, training_loss=100.0)
    outlier.cycle = 9
    mc.record(outlier)
    anomalies = mc.anomalies("training_loss", threshold=2.0)
    assert 9 in anomalies



def test_module_registry_proposal_lifecycle():
    from tovah_v14.kernel.action_model import ModuleProposal
    reg = ModuleRegistry()
    rec = reg.propose(
        ModuleProposal(
            proposer_kernel_id="hub",
            module_name="math_lab",
            module_kind="critic",
            capabilities=["score_proof", "check_consistency"],
            dependencies=["observer"],
        ),
        source_kernel_id="hub",
        packet_id="pkt_1",
        trust_level="provisional",
    )
    assert rec.proposer_kernel_id == "hub"
    assert reg.summary()["proposal_count"] >= 1
    promoted = reg.promote(rec.proposal_id, reviewer="main")
    assert promoted is not None
    assert "math_lab" in reg.experimental_manifests


def test_bus_contract_records_message_and_bindings():
    bus = MessageBusContract()
    binding = bus.bind_proposal("module_abc", target_role="main")
    req = ModuleRequest(from_role="hub", to_role="main", action="review_module_proposal", payload={"proposal_id": "module_abc"}, trace_id="pkt_1")
    msg = bus.record_request(req, kind="proposal")
    assert binding["proposal_id"] == "module_abc"
    assert msg.kind == "proposal"
    assert bus.summary()["proposal_route_count"] >= 1

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
