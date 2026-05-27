"""
TOVAH v14 tests/test_kernel.py — Kernel orchestrator tests.

Tests kernel construction, subsystem composition, command surface,
and the fundamental contract: promotion ladder is the only path.
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Need dirs to exist for kernel init
from tovah_v14.config.paths import ensure_directories
ensure_directories()

from tovah_v14.kernel.kernel import ProtozoanKernel
from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.cache import is_cache_coherent
from tovah_v14.tools.result import ToolResult


def _make_kernel():
    """Create a kernel without API for testing."""
    return ProtozoanKernel(api={}, is_original=True)


# ============================================================
# Construction tests
# ============================================================
def test_kernel_constructs():
    k = _make_kernel()
    assert k.identity.version == "14.2.6"
    assert k.identity.name == "tovah betzer"
    assert k.protect_core_goal()


def test_kernel_state_coherent():
    k = _make_kernel()
    assert is_cache_coherent(k.state)


def test_kernel_has_subsystems():
    k = _make_kernel()
    assert k.tools is not None
    assert k.invariants is not None
    assert k.certs is not None
    assert k.budget_manager is not None
    assert k.promotion_ladder is not None
    assert k.quarantine_manager is not None
    assert k.mutation_logger is not None
    assert k.memory_store is not None
    assert k.task_queue is not None
    assert k.plan_manager is not None
    assert k.self_model is not None
    assert k.competence_map is not None
    assert k.experience_store is not None
    assert k.module_health is not None


def test_kernel_beta_keys():
    k = _make_kernel()
    assert "goal.active" in k.state.beta
    assert "runtime.stability" in k.state.beta
    assert "module.planner_health" in k.state.beta
    assert "promotion.ladder_health" in k.state.beta


# ============================================================
# Scoring contract
# ============================================================
def test_kernel_score_returns_dict():
    k = _make_kernel()
    result = k._shadow_score_text("test")
    assert isinstance(result, dict)
    assert "entropy" in result


def test_kernel_score_scalar_returns_float():
    k = _make_kernel()
    result = k._shadow_score_scalar("test")
    assert isinstance(result, float)


# ============================================================
# Tool dispatch
# ============================================================
def test_kernel_tool_dispatch_unknown():
    k = _make_kernel()
    tr = k._perform_tool_action({"tool": "nonexistent"})
    assert isinstance(tr, ToolResult)
    assert not tr.ok


def test_kernel_budget_enforcement():
    k = _make_kernel()
    # Exhaust web_search budget
    k.budget_manager.budgets["web_search"]["used"] = k.budget_manager.budgets["web_search"]["limit"]
    tr = k._perform_tool_action({"tool": "web_search", "arg": "test"})
    assert not tr.ok
    assert "budget" in tr.summary.lower()


# ============================================================
# Capability tests
# ============================================================
def test_kernel_capability_tests():
    k = _make_kernel()
    passed, total, details = k.run_capability_tests()
    assert passed > 0
    assert total > 0
    assert "identity" in details
    assert details["identity"] is True
    assert "shadow_model_exists" in details
    assert details["shadow_model_exists"] is True


# ============================================================
# Patch staging
# ============================================================
def test_kernel_stage_patch():
    k = _make_kernel()
    raw = json.dumps({
        "patch_name": "test_p",
        "target": "research_topic",
        "code": "def research_topic(self, topic, context=''):\n    self.state.beta['x'] = BilateralValue(0.5, 0.5)\n    refresh_state(self.state)\n    return []\n",
        "rationale": "test",
    })
    ok, msg = k.stage_patch(raw, "test")
    assert ok, msg
    assert "test_p" in k.staged_patches
    assert k.promotion_ladder.current_stage("test_p") == "proposed"


def test_kernel_apply_through_ladder():
    k = _make_kernel()
    from tovah_v14.kernel.kernel import ProtozoanKernel
    original = ProtozoanKernel.research_topic
    raw = json.dumps({
        "patch_name": "ladder_p",
        "target": "research_topic",
        "code": "def research_topic(self, topic, context=''):\n    self.state.beta['research.novelty'] = BilateralValue(0.6, 0.1)\n    refresh_state(self.state)\n    return {'findings':[], 'raw_results':[]}\n",
        "rationale": "test",
    })
    ok, msg = k.stage_patch(raw, "test")
    assert ok
    ok2, msg2 = k.apply_staged_patch("ladder_p")
    assert ok2, msg2
    assert k.promotion_ladder.current_stage("ladder_p") == "live_promoted"
    ProtozoanKernel.research_topic = original


# ============================================================
# Assess patch
# ============================================================
def test_kernel_assess_good_patch():
    k = _make_kernel()
    raw = json.dumps({
        "target": "research_topic",
        "code": "def research_topic(self, topic, context=''):\n    self.state.beta['x'] = BilateralValue(0.5, 0.5)\n    refresh_state(self.state)\n    return []\n",
        "rationale": "well-reasoned improvement",
    })
    bv = k.assess_patch_json(raw)
    assert isinstance(bv, BilateralValue)
    assert bv.t > bv.f


def test_kernel_assess_bad_patch():
    k = _make_kernel()
    bv = k.assess_patch_json("not json")
    assert bv.f > bv.t


# ============================================================
# Self summary
# ============================================================
def test_kernel_self_summary():
    k = _make_kernel()
    s = k.get_self_summary()
    assert s["version"] == "14.2.6"
    assert "memory" in s
    assert "promotion_queue" in s


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
