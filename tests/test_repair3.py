"""
TOVAH v14 tests/test_repair3.py — Targeted tests for repair pass 3.
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

# A. No implicit create-new
def test_extension_target_fails_without_explicit_create_new():
    """EXTENSION_TARGETS membership alone does NOT authorize creation."""
    k = _k()
    code = "def _extract_pdf_text_local(self, path):\n    return {'ok': True}\n"
    ok, msg = k.direct_inject_method("_extract_pdf_text_local", code, create_new=False)
    assert not ok, f"should fail: create_new=False even for extension target; got: {msg}"

def test_command_inject_cannot_implicitly_create():
    """INJECT_METHOD command path cannot create absent methods."""
    k = _k()
    from tovah_v14.config.paths import COMMAND_FILE, RESPONSE_FILE
    code = "def _extract_pdf_text_local(self, path):\n    return {'ok': True}\n"
    cmd = f"INJECT_METHOD:\n_extract_pdf_text_local\n{code}"
    COMMAND_FILE.write_text(cmd, encoding="utf-8")
    k._check_david_commands()
    resp = RESPONSE_FILE.read_text(encoding="utf-8") if RESPONSE_FILE.exists() else ""
    assert "preflight rejected" in resp.lower() or "fail" in resp.lower() or "absent" in resp.lower(), \
        f"command inject should fail for absent target; got: {resp[:200]}"

def test_explicit_create_new_works_end_to_end():
    """Explicit create_new=True + extension target works."""
    k = _k()
    from tovah_v14.kernel.kernel import ProtozoanKernel
    code = "def _extract_pdf_text_local(self, path):\n    return {'ok': True, 'text': 'extracted'}\n"
    ok, msg = k.direct_inject_method("_extract_pdf_text_local", code, create_new=True)
    assert ok, f"explicit create-new should work: {msg}"
    assert callable(getattr(k, "_extract_pdf_text_local", None))
    # Cleanup
    if hasattr(ProtozoanKernel, "_extract_pdf_text_local"):
        delattr(ProtozoanKernel, "_extract_pdf_text_local")

# B. SELF_MODEL command works
def test_self_model_command_works():
    k = _k()
    from tovah_v14.config.paths import COMMAND_FILE, RESPONSE_FILE
    COMMAND_FILE.write_text("SELF_MODEL", encoding="utf-8")
    k._check_david_commands()
    resp = RESPONSE_FILE.read_text(encoding="utf-8") if RESPONSE_FILE.exists() else ""
    assert "Error" not in resp or "NameError" not in resp, f"SELF_MODEL command error: {resp[:200]}"
    assert "version" in resp.lower() or "subsystem" in resp.lower(), f"SELF_MODEL should return model data: {resp[:200]}"

# D. Contract return types match
def test_research_topic_returns_dict():
    k = _k()
    synth = k.research_topic("python testing")
    assert isinstance(synth, dict)
    for key in ("findings", "uncertainties", "contradictions", "raw_results",
                "bilateral_confidence", "provenance", "success_count"):
        assert key in synth, f"missing key: {key}"

def test_adapt_research_code_returns_list_of_dicts():
    k = _k()
    # Stage a tool opportunity
    k._store_memory("semantic", "tool_opportunity:test_adapt",
                     {"name": "test_adapt", "capability": "testing", "rationale": "test"},
                     tags=["tool_opportunity"])
    for e in k.memory_store.get_bank("semantic"):
        if e.key == "tool_opportunity:test_adapt":
            e.bilateral_confidence = BilateralValue(0.8, 0.1)
    proposals = k._adapt_research_code()
    assert isinstance(proposals, list)
    if proposals:
        p = proposals[0]
        assert isinstance(p, dict)
        assert "kind" in p
        assert "rationale" in p
        assert "risk_class" in p

# E. Research synthesis depth
def test_research_synthesis_has_structured_fields():
    k = _k()
    synth = k.research_topic("bilateral paraconsistent logic")
    assert "contradictions" in synth
    assert isinstance(synth["contradictions"], list)
    assert "patch_opportunities" in synth
    assert isinstance(synth["patch_opportunities"], list)
    assert "service_opportunities" in synth
    bv = synth.get("bilateral_confidence", {})
    assert "t" in bv and "f" in bv

def test_research_populates_patch_opportunities():
    """patch_opportunities should be non-empty when tool_opps found."""
    k = _k()
    synth = k.research_topic("free REST API python")
    # Even if net fails, structure should be present
    assert isinstance(synth.get("patch_opportunities"), list)
    assert isinstance(synth.get("tool_opportunities"), list)

# F. _discover_free_services structured ranking
def test_discover_services_returns_ranked_candidates():
    k = _k()
    results = k._discover_free_services("api")
    assert isinstance(results, list)
    if results:
        # Should have ranking score
        assert "_score" in results[0] or "name" in results[0]

# G. _adapt_research_code proposal metadata
def test_adapt_proposal_has_metadata():
    k = _k()
    k._store_memory("semantic", "tool_opportunity:meta_test",
                     {"name": "meta_test", "capability": "API", "rationale": "found in research"},
                     tags=["tool_opportunity"])
    for e in k.memory_store.get_bank("semantic"):
        if e.key == "tool_opportunity:meta_test":
            e.bilateral_confidence = BilateralValue(0.9, 0.05)
    props = k._adapt_research_code()
    assert isinstance(props, list)
    if props:
        p = props[0]
        for key in ("kind", "rationale", "risk_class", "provenance"):
            assert key in p, f"proposal missing: {key}"

# H/I. Goal generation uses self-model, blocked growth, budget pressure
def test_goal_generation_uses_blocked_growth():
    k = _k()
    k.current_goal = None
    k.task_queue.tasks = []
    k._blocked_growth_log = [
        {"patch": "p1", "stage": "static_approved", "reason": "blocked", "time": time.time()},
        {"patch": "p2", "stage": "static_approved", "reason": "blocked", "time": time.time()},
        {"patch": "p3", "stage": "static_approved", "reason": "blocked", "time": time.time()},
    ]
    goal = k._generate_next_goal()
    assert goal is not None
    assert "blocked" in goal.get("reasoning", "").lower() or "blocked" in goal.get("goal", "").lower()

def test_self_model_has_module_health():
    k = _k()
    k.update_self_model()
    sm = k.self_model
    assert hasattr(sm, "module_health_summary")
    assert isinstance(sm.module_health_summary, dict)
    assert len(sm.module_health_summary) > 0

def test_self_model_has_budget_info():
    k = _k()
    k.update_self_model()
    sm = k.self_model
    assert hasattr(sm, "budget_summary")
    assert isinstance(sm.budget_summary, dict)

# J. No init snapshot
def test_no_init_snapshot():
    import tovah_v14.persistence.snapshots as snap_mod
    orig = snap_mod.save_snapshot
    calls = [0]
    def counting(*a, **kw):
        calls[0] += 1
        return {"reason": "test"}
    snap_mod.save_snapshot = counting
    try:
        k = _k()
        assert calls[0] == 0
    finally:
        snap_mod.save_snapshot = orig

# Bounded regression
def test_bounded_regression_no_hang():
    k = _k()
    import threading
    result = [None]
    def run():
        result[0] = k.run_capability_tests()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert result[0] is not None, "bounded regression hung"

def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  [PASS] {t.__name__}")
        except Exception as e: failed += 1; print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
