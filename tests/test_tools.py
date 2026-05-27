"""
TOVAH v14 tests/test_tools.py — Tool layer tests.

Verifies:
- ToolResult shape preserved from v13
- ToolLayer constructs and has all builtins
- BudgetManager check/spend/reset
- Tool contracts exist for all builtins
- Browser action returns dict (not ToolResult)
- Extract text returns dict (not ToolResult)
- Budget enforcement prevents overspend

Does NOT test live network calls (those are integration tests).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.tools.result import ToolResult
from tovah_v14.tools.layer import ToolLayer
from tovah_v14.tools.budgets import BudgetManager
from tovah_v14.tools.contracts import TOOL_CONTRACTS, ToolContract
from tovah_v14.tools.browser import browser_action
from tovah_v14.tools.extraction import extract_text
from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import CarrierState, ProvenanceState, ShadowState
from tovah_v14.core.cache import refresh_state


# ============================================================
# ToolResult tests
# ============================================================
def test_tool_result_shape():
    tr = ToolResult(ok=True, tool="test", summary="ok", payload={"k": "v"}, url="http://x")
    assert tr.ok is True
    assert tr.tool == "test"
    assert tr.summary == "ok"
    assert tr.payload == {"k": "v"}
    assert tr.url == "http://x"


def test_tool_result_defaults():
    tr = ToolResult(ok=False, tool="t", summary="fail")
    assert tr.payload is None
    assert tr.url == ""


# ============================================================
# ToolLayer tests
# ============================================================
def test_tool_layer_builtins():
    tl = ToolLayer(timeout=5)
    builtins = tl.builtins
    assert "web_search" in builtins
    assert "fetch_url" in builtins
    assert "github_repo" in builtins
    assert "github_file" in builtins
    assert "robots_ok" in builtins
    assert "wikipedia_summary" in builtins
    assert "arxiv_search" in builtins
    assert "rss_fetch" in builtins
    assert "json_api_fetch" in builtins
    assert "sitemap_fetch" in builtins
    assert "browser_action" in builtins
    assert "extract_text" in builtins
    assert len(builtins) == 12


def test_tool_layer_session():
    tl = ToolLayer(timeout=10)
    assert tl.session is not None
    assert "TOVAH" in tl.session.headers.get("User-Agent", "")


# ============================================================
# Budget tests
# ============================================================
def test_budget_check_under_limit():
    bm = BudgetManager({"test": {"limit": 5, "used": 3, "reset_at": 0.0}})
    assert bm.check("test", 1) is True
    assert bm.check("test", 2) is True
    assert bm.check("test", 3) is False


def test_budget_spend():
    bm = BudgetManager({"test": {"limit": 3, "used": 0, "reset_at": 0.0}})
    assert bm.spend("test") is True
    assert bm.spend("test") is True
    assert bm.spend("test") is True
    assert bm.spend("test") is False  # over limit


def test_budget_unknown_resource():
    bm = BudgetManager({})
    assert bm.check("unknown") is True
    assert bm.spend("unknown") is True


def test_budget_reset():
    bm = BudgetManager({"test": {"limit": 5, "used": 5, "reset_at": 0.0}})
    assert bm.check("test") is False
    bm.reset_if_needed()
    assert bm.check("test") is True


def test_budget_usage_summary():
    bm = BudgetManager({"a": {"limit": 10, "used": 3, "reset_at": 0.0}})
    s = bm.usage_summary()
    assert abs(s["a"] - 0.3) < 0.01


def test_budget_bilateral_update():
    state = ShadowState(
        c=CarrierState(), beta={"budget.compliance": BilateralValue(0.5, 0.2)},
        nu={}, pi=ProvenanceState(),
    )
    refresh_state(state)
    bm = BudgetManager({"test": {"limit": 5, "used": 3, "reset_at": 0.0}})
    bm.update_bilateral_state(state)
    assert state.beta["budget.compliance"].t > 0.5  # under budget -> truth gain


def test_budget_bilateral_overspend():
    state = ShadowState(
        c=CarrierState(), beta={"budget.compliance": BilateralValue(0.5, 0.2)},
        nu={}, pi=ProvenanceState(),
    )
    refresh_state(state)
    bm = BudgetManager({"test": {"limit": 5, "used": 6, "reset_at": 0.0}})
    bm.update_bilateral_state(state)
    assert state.beta["budget.compliance"].f > 0.2  # over budget -> falsity gain


# ============================================================
# Contract tests
# ============================================================
def test_all_builtins_have_contracts():
    tl = ToolLayer()
    for name in tl.builtins:
        assert name in TOOL_CONTRACTS, f"missing contract for: {name}"


def test_contract_shape():
    c = TOOL_CONTRACTS["web_search"]
    assert isinstance(c, ToolContract)
    assert c.name == "web_search"
    assert "query" in c.inputs
    assert c.cost == "low"
    assert c.budget_resource == "web_search"


def test_browser_contract_is_high_cost():
    c = TOOL_CONTRACTS["browser_action"]
    assert c.cost == "high"
    assert c.required_permissions == "safe_logged"


# ============================================================
# ToolLayer browser/extract interface parity tests
# ============================================================
def test_tool_layer_browser_action_returns_tool_result():
    """ToolLayer.browser_action must return ToolResult, not dict."""
    tl = ToolLayer(timeout=5)
    result = tl.browser_action("unsupported_action")
    assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
    assert result.tool == "browser_action"


def test_tool_layer_extract_text_returns_tool_result():
    """ToolLayer.extract_text must return ToolResult, not dict."""
    tl = ToolLayer(timeout=2)
    result = tl.extract_text("http://nonexistent.invalid")
    assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
    assert result.tool == "extract_text"


def test_tool_layer_all_builtins_are_methods():
    """Every name in builtins must be a callable method on ToolLayer."""
    tl = ToolLayer()
    for name in tl.builtins:
        assert hasattr(tl, name), f"ToolLayer missing method: {name}"
        assert callable(getattr(tl, name)), f"ToolLayer.{name} is not callable"


# ============================================================
# Browser interface test — parse_only mode, no Playwright
# ============================================================
def test_browser_action_returns_dict():
    """browser_action in parse_only mode returns a dict with standard keys."""
    result = browser_action("unsupported_action", mode="parse_only")
    assert isinstance(result, dict)
    assert "ok" in result
    assert "action" in result
    assert "error" in result


def test_browser_pipe_parsing():
    """Verify pipe-delimited action parsing in parse_only mode — no browser bootstrap."""
    result = browser_action("extract_text|http://example.com", mode="parse_only")
    assert isinstance(result, dict)
    assert result["action"] == "extract_text"
    assert result["url"] == "http://example.com"
    assert result["ok"] is True
    assert result["latency_ms"] == 0


# ============================================================
# Extract text interface test (no live network)
# ============================================================
def test_extract_text_returns_dict():
    """extract_text returns a dict, not ToolResult."""
    import requests
    session = requests.Session()
    result = extract_text(session, "http://nonexistent.invalid", timeout=2)
    assert isinstance(result, dict)
    assert "ok" in result
    assert result["ok"] is False  # network will fail


def test_extract_text_no_bs4():
    """Test behavior when bs4 is not available."""
    def fake_ensure(pkg, imp_name=None):
        return False, "not installed"

    import requests
    session = requests.Session()
    result = extract_text(session, "http://example.com", ensure_package=fake_ensure)
    assert result["ok"] is False
    assert "bs4" in result["summary"].lower()


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
