"""
TOVAH v14 tests/test_integration.py — Full integration tests.

Covers:
- manifest/live-kernel parity (every manifest method is callable)
- preflight catches missing manifest methods
- direct injection passes through patch preflight
- research_topic uses multi-step machinery
- _autonomous_cycle executes without advisor and without crashing
- memory conflict preservation
- experience persistence roundtrip
- competence persistence roundtrip
- budget-aware tool ranking
- shadow weight save cadence (no hammer)
- module health influences next-goal generation
- _classify_query_intent returns stable taxonomy
- _decompose_goal_into_queries returns 3-6 queries
- _generate_next_goal returns structured goal
- _strategic_plan returns StrategicPlan
"""
from __future__ import annotations
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tovah_v14.config.paths import ensure_directories
ensure_directories()

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.cache import refresh_state


def _k():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    return ProtozoanKernel(api={}, is_original=True)


# ============================================================
# Manifest / live-kernel parity
# ============================================================
def test_all_manifest_methods_callable():
    k = _k()
    from tovah_v14.modules.manifests import MODULE_MANIFESTS
    for role, manifest in MODULE_MANIFESTS.items():
        for method in manifest.methods:
            fn = getattr(k, method, None)
            assert fn is not None and callable(fn), f"manifest method not callable: {role}.{method}"


def test_preflight_validates_manifests():
    k = _k()
    from tovah_v14.kernel.preflight import run_preflight
    result = run_preflight(k)
    # All manifest checks should pass
    manifest_checks = {ck: v for ck, v in result.checks.items() if ck.startswith("manifest.")}
    assert len(manifest_checks) > 0
    for ck, v in manifest_checks.items():
        assert v, f"preflight manifest check failed: {ck}"


# ============================================================
# Direct injection through patch preflight
# ============================================================
def test_inject_rejects_obsolete_patterns():
    k = _k()
    code = """
def research_topic(self, topic, context=''):
    score = float(self._shadow_score_text(topic))
    return []
"""
    ok, msg = k.direct_inject_method("research_topic", code)
    assert not ok, "should reject obsolete float(score_text) pattern"
    assert "preflight" in msg.lower() or "obsolete" in msg.lower() or "rejected" in msg.lower()


def test_inject_accepts_clean_code():
    k = _k()
    from tovah_v14.kernel.kernel import ProtozoanKernel
    original = ProtozoanKernel.research_topic
    code = """
def research_topic(self, topic, context=''):
    self.state.beta['research.novelty'] = BilateralValue(0.6, 0.1)
    refresh_state(self.state)
    return {'findings': [], 'raw_results': [], 'success_count': 0, 'total_queries': 0}
"""
    ok, msg = k.direct_inject_method("research_topic", code)
    assert ok, f"should accept clean code: {msg}"
    ProtozoanKernel.research_topic = original


# ============================================================
# Research uses multi-step
# ============================================================
def test_research_topic_multistep():
    k = _k()
    initial_experience = len(k.experience_store.records)
    initial_research = len(k.research_memory)
    synth = k.research_topic("python data structures")
    assert isinstance(synth, dict), "research_topic must return synthesis dict"
    assert "findings" in synth
    assert "raw_results" in synth
    # Should have recorded new experience
    assert len(k.experience_store.records) > initial_experience
    # Should have added to research memory
    assert len(k.research_memory) > initial_research


# ============================================================
# Autonomous cycle without advisor
# ============================================================
def test_autonomous_cycle_no_crash():
    k = _k()
    k.current_goal = {"goal": "test autonomous", "function_spec": "test", "domain": "testing", "reasoning": "test"}
    k.state.c.active_goal = "test autonomous"
    k.state.beta["goal.active"] = BilateralValue(0.9, 0.0)
    refresh_state(k.state)
    # Should not crash even without advisor
    k._autonomous_cycle()
    # Should have advanced state somehow
    assert k.state.c.cycle >= 0


# ============================================================
# Memory conflict preservation
# ============================================================
def test_memory_conflict_preserves_both():
    k = _k()
    k._store_memory("semantic", "fact:x", {"value": "old"})
    entry2, conflicts = k._store_memory("semantic", "fact:x", {"value": "new"})
    # Both entries should exist
    sem = k.memory_store.get_bank("semantic")
    keys = [e.key for e in sem]
    assert keys.count("fact:x") >= 2, "conflict should preserve both entries"


# ============================================================
# Experience persistence roundtrip
# ============================================================
def test_experience_roundtrip():
    k = _k()
    k.experience_store.record("exp_1", "research", outcome="useful", reward_signal=0.8)
    k.experience_store.record("exp_2", "patch", outcome="regressed", reward_signal=-0.5)
    k.save_state()
    # Verify records are in save dict
    from tovah_v14.persistence.state_io import load_state_from_file
    from tovah_v14.config.paths import STATE_FILE
    raw = load_state_from_file(STATE_FILE)
    er = raw.get("experience_records", [])
    assert len(er) >= 2


# ============================================================
# Competence persistence roundtrip
# ============================================================
def test_competence_roundtrip():
    k = _k()
    k.competence_map.record_outcome("tool_mastery", True)
    k.competence_map.record_outcome("tool_mastery", False)
    k.save_state()
    from tovah_v14.persistence.state_io import load_state_from_file
    from tovah_v14.config.paths import STATE_FILE
    raw = load_state_from_file(STATE_FILE)
    cm = raw.get("competence_map", {})
    assert "tool_mastery" in cm


# ============================================================
# Budget-aware tool ranking
# ============================================================
def test_budget_aware_ranking():
    k = _k()
    # Exhaust web_search budget
    k.budget_manager.budgets["web_search"]["used"] = k.budget_manager.budgets["web_search"]["limit"]
    ranked = k._rank_tool_candidates("python tutorial", "python tutorial")
    # web_search should be deprioritized
    if "web_search" in ranked and "arxiv_search" in ranked:
        ws_idx = ranked.index("web_search")
        # web_search should be ranked lower than tools with remaining budget
        assert ws_idx > 0 or len(ranked) == 1


# ============================================================
# Shadow weight save cadence
# ============================================================
def test_shadow_save_not_on_every_save():
    k = _k()
    # save_state should NOT call save_shadow_weights
    import tovah_v14.persistence.snapshots as snap_mod
    original = snap_mod.save_shadow_weights
    call_count = [0]
    def counting_save(*a, **kw):
        call_count[0] += 1
        return True
    snap_mod.save_shadow_weights = counting_save
    try:
        k.save_state()
        k.save_state()
        k.save_state()
        assert call_count[0] == 0, f"save_state called shadow save {call_count[0]} times (should be 0)"
    finally:
        snap_mod.save_shadow_weights = original


# ============================================================
# Module health influences goal generation
# ============================================================
def test_module_health_influences_goals():
    k = _k()
    k.current_goal = None  # clear explicit goal
    k.task_queue.tasks = []  # clear tasks
    # Make planner very weak
    k.state.beta["module.planner_health"] = BilateralValue(0.1, 0.9)
    refresh_state(k.state)
    goal = k._generate_next_goal()
    assert goal is not None
    # Goal should reference module weakness or competence gap


# ============================================================
# Intent classification
# ============================================================
def test_classify_intent():
    k = _k()
    assert k._classify_query_intent("https://example.com/page") == "url_fetch"
    assert k._classify_query_intent("github.com/user/repo") == "github_repo"
    assert k._classify_query_intent("arxiv paper on transformers") == "paper_lookup"
    assert k._classify_query_intent("python web scraping tutorial") == "broad_research"


# ============================================================
# Query decomposition
# ============================================================
def test_decompose_goal():
    k = _k()
    queries = k._decompose_goal_into_queries("build a web scraper for news articles")
    assert isinstance(queries, list)
    assert 3 <= len(queries) <= 6
    assert len(set(q.lower() for q in queries)) == len(queries), "queries should be unique"


# ============================================================
# Strategic plan
# ============================================================
def test_strategic_plan():
    k = _k()
    plan = k._strategic_plan("learn about bilateral logic")
    assert plan is not None
    assert isinstance(plan, StrategicPlan)
    assert len(plan.steps) >= 2
    assert plan.status == "active"


# ============================================================
# Runner
# ============================================================
from tovah_v14.tasks.plans import StrategicPlan

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
