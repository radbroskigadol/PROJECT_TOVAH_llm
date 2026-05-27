"""
TOVAH v14 tests/test_smoke.py — Critical-path smoke tests.

Fast, deterministic, no browser/runtime dependencies.

Covers:
1. Kernel boot
2. Command parse and dispatch shape
3. Tool wrapper call shapes
4. Patch preflight validation
5. Persistence roundtrip
6. Browser command parsing (parse_only, no Playwright)
7. State schema validation
8. Research topic typed result
9. Preflight check
10. ToolActionResult contract
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.config.paths import ensure_directories
ensure_directories()


# ============================================================
# 1. Kernel boot
# ============================================================
def test_kernel_boot():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    k = ProtozoanKernel(api={}, is_original=True)
    assert k.identity.version == "14.2.6"
    assert k.state is not None
    assert k.tools is not None


# ============================================================
# 2. Browser command parsing (no Playwright, no hang)
# ============================================================
def test_browser_parse_only_simple():
    from tovah_v14.tools.browser import parse_browser_command
    cmd = parse_browser_command("navigate", "http://example.com")
    assert cmd.action == "navigate"
    assert cmd.url == "http://example.com"


def test_browser_parse_only_pipe():
    from tovah_v14.tools.browser import parse_browser_command
    cmd = parse_browser_command("extract_text|http://example.com|#content")
    assert cmd.action == "extract_text"
    assert cmd.url == "http://example.com"
    assert cmd.selector == "#content"


def test_browser_parse_only_complex():
    from tovah_v14.tools.browser import parse_browser_command
    cmd = parse_browser_command("fill|http://x.com|#email|test@test.com")
    assert cmd.action == "fill"
    assert cmd.url == "http://x.com"
    assert cmd.selector == "#email"
    assert cmd.text == "test@test.com"


def test_browser_action_parse_only_mode():
    from tovah_v14.tools.browser import browser_action
    result = browser_action("extract_text|http://example.com", mode="parse_only")
    assert result["ok"] is True
    assert result["action"] == "extract_text"
    assert result["url"] == "http://example.com"
    assert result["latency_ms"] == 0
    assert result["error"] is None


def test_browser_action_dry_run_no_playwright():
    """dry_run with no playwright should fail fast, not hang."""
    from tovah_v14.tools.browser import browser_action
    result = browser_action("navigate|http://example.com", mode="dry_run")
    assert isinstance(result, dict)
    assert "ok" in result
    # Either playwright is available or it fails fast
    assert isinstance(result.get("latency_ms"), int)


# ============================================================
# 3. Tool wrapper call shapes
# ============================================================
def test_tool_result_shape():
    from tovah_v14.tools.result import ToolResult
    tr = ToolResult(ok=True, tool="test", summary="ok", payload={"k": "v"}, url="http://x")
    assert tr.ok is True
    assert tr.tool == "test"


def test_tool_action_result_shape():
    from tovah_v14.tools.result import ToolActionResult
    tar = ToolActionResult(ok=True, tool="web_search", action="search", summary="found 5 results")
    assert tar.ok
    assert tar.error is None
    assert tar.retryable is False
    # Convert to v13 ToolResult
    tr = tar.to_tool_result()
    assert isinstance(tr.ok, bool)
    assert tr.tool == "web_search"


# ============================================================
# 4. Patch preflight validation
# ============================================================
def test_patch_preflight_good():
    from tovah_v14.kernel.patch_preflight import validate_patch_preflight
    from tovah_v14.kernel.kernel import ProtozoanKernel
    code = """
def research_topic(self, topic, context=''):
    self.state.beta['research.novelty'] = BilateralValue(0.6, 0.1)
    refresh_state(self.state)
    return []
"""
    report = validate_patch_preflight("research_topic", code, ProtozoanKernel)
    assert report.accepted, f"expected accepted, errors: {report.errors}"
    assert report.analysis_ok
    assert len(report.obsolete_patterns) == 0


def test_patch_preflight_obsolete_pattern():
    from tovah_v14.kernel.patch_preflight import validate_patch_preflight
    from tovah_v14.kernel.kernel import ProtozoanKernel
    code = """
def research_topic(self, topic, context=''):
    score = float(self._shadow_score_text(topic))
    self.state.beta['x'] = BilateralValue(score, 0.1)
    refresh_state(self.state)
    return []
"""
    report = validate_patch_preflight("research_topic", code, ProtozoanKernel)
    assert not report.accepted
    assert any("obsolete" in e.lower() for e in report.errors)


def test_patch_preflight_protected():
    from tovah_v14.kernel.patch_preflight import validate_patch_preflight
    from tovah_v14.kernel.kernel import ProtozoanKernel
    code = "def __init__(self): pass"
    report = validate_patch_preflight("__init__", code, ProtozoanKernel)
    assert not report.accepted
    assert any("PROTECTED" in e for e in report.errors)


# ============================================================
# 5. Persistence roundtrip
# ============================================================
def test_persistence_roundtrip():
    from tovah_v14.persistence.state_io import save_json, load_json
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.json"
        data = {"state_version": "14.0.0", "state": {"c": {}, "beta": {}, "nu": {}, "pi": {}}, "completed_goals": []}
        assert save_json(path, data)
        loaded = load_json(path)
        assert loaded["state_version"] == "14.0.0"


def test_state_schema_validation():
    from tovah_v14.persistence.schema import validate_state_schema
    good = {
        "state_version": "14.0.0", "state": {"c": {}, "beta": {}, "nu": {}, "pi": {}},
        "completed_goals": [], "pending_tool_actions": [], "staged_patches": {},
        "patch_history": [], "loss_history": [], "research_memory": [],
        "improvement_count": 0, "autonomy_level": 0, "alpha": 1.0, "temperature": 0.9,
    }
    result = validate_state_schema(good)
    assert result.ok, f"expected ok, errors: {result.missing_required} {result.type_mismatches}"

    # Bad: missing required field
    bad = {"state_version": "14.0.0"}
    result2 = validate_state_schema(bad)
    assert not result2.ok
    assert len(result2.missing_required) > 0


# ============================================================
# 6. Command registry
# ============================================================
def test_command_registry_populated():
    from tovah_v14.kernel.preflight import COMMAND_REGISTRY
    assert len(COMMAND_REGISTRY) >= 50
    assert "STATUS" in COMMAND_REGISTRY
    assert "GOAL:<text>" in COMMAND_REGISTRY
    assert COMMAND_REGISTRY["INGEST_LEVBEL"].status == "deferred"


# ============================================================
# 7. Preflight
# ============================================================
def test_kernel_preflight():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.preflight import run_preflight
    k = ProtozoanKernel(api={}, is_original=True)
    result = run_preflight(k)
    assert result.checks.get("tool.web_search") is True
    assert result.checks.get("tool.browser_action") is True
    assert result.checks.get("tool.extract_text") is True
    assert result.checks.get("command_handler") is True
    assert result.checks.get("shadow_model") is True


# ============================================================
# 8. Research topic returns typed result
# ============================================================
def test_research_topic_typed(monkeypatch):
    """Research synthesis should return the typed result shape without touching live tools.

    The production method may try ranked tool calls. A smoke test must remain
    deterministic and fast, so we stub decomposition/tool execution while still
    exercising the synthesis/body of research_topic().
    """
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.tools.result import ToolResult

    k = ProtozoanKernel(api={}, is_original=True)
    monkeypatch.setattr(k, "_decompose_goal_into_queries", lambda topic: ["stub query"])
    monkeypatch.setattr(k, "_rank_tool_candidates", lambda query, context: ["stub_tool"])
    monkeypatch.setattr(
        k, "_perform_tool_action",
        lambda action: ToolResult(
            ok=True, tool=str(action.get("tool", "stub_tool")),
            summary="stubbed research summary", payload={}, url=""
        ),
    )
    monkeypatch.setattr(k, "_discover_free_services", lambda seed=None: [])

    synth = k.research_topic("test topic", "context")
    assert isinstance(synth, dict)
    assert "findings" in synth
    assert "raw_results" in synth
    assert synth["success_count"] == 1


# ============================================================
# 9. Typed action model
# ============================================================
def test_goal_object():
    from tovah_v14.kernel.action_model import Goal
    g = Goal(goal="learn Python", domain="learning", reasoning="growth")
    assert g.goal == "learn Python"
    assert g.bilateral_confidence.t == 0.5


def test_patch_proposal():
    from tovah_v14.kernel.action_model import PatchProposal
    pp = PatchProposal(patch_name="p1", target="research_topic", code="def research_topic(self): pass",
                        rationale="test", risk_level="low")
    assert pp.approval_required is True
    assert pp.preflight_passed is False


# ============================================================
# 10. Packaging / launcher smoke test
# ============================================================
def test_launcher_imports_and_boots():
    """Verify the documented launch path works: import + kernel boot + preflight.

    This prevents future launcher regressions where run_tovah.py
    claims to work but fails with ModuleNotFoundError.
    """
    # The inner run_tovah.py must be importable
    from tovah_v14.run_tovah import main, build_api
    assert callable(main)
    assert callable(build_api)

    # Kernel must boot
    from tovah_v14.kernel.kernel import ProtozoanKernel
    k = ProtozoanKernel(api={}, is_original=True)
    assert k.identity.version == "14.2.6"

    # Preflight must pass
    from tovah_v14.kernel.preflight import run_preflight
    result = run_preflight(k)
    assert result.ok, f"preflight failed: {result.errors}"
    assert sum(1 for v in result.checks.values() if v) == len(result.checks), \
        f"not all checks passed: {[k for k, v in result.checks.items() if not v]}"


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
