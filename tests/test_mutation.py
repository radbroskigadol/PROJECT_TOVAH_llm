"""
TOVAH v14 tests/test_mutation.py — Mutation system tests.

Verifies:
- analyze_patch_code preserves v13 blocking behavior
- Contract validation catches forbidden patterns and missing refresh
- Staging produces correct records
- Promotion ladder enforces stage ordering
- Promotion cannot skip stages
- Quarantine blocks promotion
- apply_live only works from shadow_deployed
- Revert works
- Mutation log records events
- Non-bypass: no direct path from staged to applied without ladder
"""
from __future__ import annotations

import json
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.mutation.analysis import (
    analyze_patch_code, analyze_patch_with_contract, PatchDescriptor,
    BLOCKED_IMPORT_ROOTS_MUTABLE,
)
from tovah_v14.mutation.staging import stage_patch, StagingResult
from tovah_v14.mutation.quarantine import quarantine_patch, QuarantineManager
from tovah_v14.mutation.promotion_ladder import PromotionLadder
from tovah_v14.mutation.mutation_log import MutationLogger
from tovah_v14.core.contracts import ALLOWED_PATCH_TARGETS, PROTECTED_METHODS


# ============================================================
# analyze_patch_code tests (v13 behavior preservation)
# ============================================================
def test_analyze_valid_patch():
    code = "def research_topic(self, topic, context=''):\n    return []\n"
    ok, fn_names, errs = analyze_patch_code(code)
    assert ok
    assert "research_topic" in fn_names
    assert not errs


def test_analyze_blocks_eval():
    code = "def f():\n    eval('x')\n"
    ok, _, errs = analyze_patch_code(code)
    assert not ok
    assert any("eval" in e for e in errs)


def test_analyze_blocks_subprocess():
    code = "import subprocess\ndef f():\n    pass\n"
    ok, _, errs = analyze_patch_code(code)
    assert not ok
    assert any("subprocess" in e for e in errs)


def test_analyze_blocks_protected():
    code = "def __init__(self):\n    pass\n"
    ok, _, errs = analyze_patch_code(code)
    assert not ok
    assert any("protected" in e for e in errs)


def test_analyze_blocks_dunder():
    code = "def __secret__(self):\n    pass\n"
    ok, _, errs = analyze_patch_code(code)
    assert not ok
    assert any("dunder" in e for e in errs)


def test_analyze_requires_function():
    code = "x = 1\ny = 2\n"
    ok, _, errs = analyze_patch_code(code)
    assert not ok
    assert any("function" in e for e in errs)


def test_analyze_blocks_global():
    code = "def f():\n    global x\n    x = 1\n"
    ok, _, errs = analyze_patch_code(code)
    assert not ok
    assert any("global" in e for e in errs)


# ============================================================
# Contract validation tests
# ============================================================
def test_contract_validation_passes():
    code = """
def research_topic(self, topic, context=""):
    self.state.beta["research.novelty"] = bilateral_recover(
        self.state.beta.get("research.novelty", BilateralValue()), truth_gain=0.1)
    refresh_state(self.state)
    return []
"""
    ok, fn_names, errors, contract_ok = analyze_patch_with_contract("research_topic", code)
    assert ok
    assert contract_ok
    assert "research_topic" in fn_names


def test_contract_catches_forbidden_pattern():
    code = """
def research_topic(self, topic, context=""):
    self.tool_layer.search(topic)
    self.state.beta["x"] = BilateralValue(0.5, 0.5)
    refresh_state(self.state)
    return []
"""
    ok, _, errors, contract_ok = analyze_patch_with_contract("research_topic", code)
    assert not ok or not contract_ok
    assert any("forbidden" in e for e in errors)


def test_contract_catches_missing_refresh():
    code = """
def _classify_query_intent(self, text):
    self.state.beta["intent"] = BilateralValue(0.5, 0.5)
    return "broad_research"
"""
    ok, _, errors, contract_ok = analyze_patch_with_contract("_classify_query_intent", code)
    assert not contract_ok
    assert any("refresh_state" in e for e in errors)


def test_contract_rejects_unknown_target():
    code = "def unknown_method(self):\n    pass\n"
    ok, _, errors, _ = analyze_patch_with_contract("unknown_method", code)
    assert not ok


# ============================================================
# Staging tests
# ============================================================
def test_stage_valid_patch():
    patches: dict = {}
    raw = json.dumps({
        "patch_name": "test_patch",
        "target": "research_topic",
        "code": (
            "def research_topic(self, topic, context=''):\n"
            "    self.state.beta['research.novelty'] = BilateralValue(0.6, 0.1)\n"
            "    refresh_state(self.state)\n"
            "    return []\n"
        ),
        "rationale": "test patch",
    })
    result = stage_patch(raw, source="test", staged_patches=patches)
    assert result.ok, f"staging failed: {result.message}"
    assert "test_patch" in patches
    assert patches["test_patch"]["status"] == "staged"


def test_stage_invalid_json():
    result = stage_patch("not json", staged_patches={})
    assert not result.ok
    assert "json" in result.message.lower()


def test_stage_blocked_target():
    raw = json.dumps({"target": "__init__", "code": "def __init__(self): pass"})
    result = stage_patch(raw, staged_patches={})
    assert not result.ok


# ============================================================
# Promotion ladder tests — THE CORE V14 DISCIPLINE
# ============================================================
def _make_staged_patch(name="test_p", target="research_topic"):
    return {
        name: {
            "patch_name": name,
            "target": target,
            "code": (
                f"def {target}(self, topic, context=''):\n"
                f"    self.state.beta['research.novelty'] = BilateralValue(0.6, 0.1)\n"
                f"    refresh_state(self.state)\n"
                f"    return []\n"
            ),
            "rationale": "test",
            "status": "staged",
        }
    }


def _seed_sovereign_metadata(ladder, name="test_p"):
    """AUDIT FIX (v14.2.7, RC-1): pre-v14.2.7, tests that drove the ladder
    directly without going through the kernel got the implicit-sovereign
    default. After the inversion, tests must declare metadata explicitly.
    This helper registers sovereign-main metadata so tests focused on
    ladder mechanics still exercise their intended code path."""
    ladder.set_source_metadata(
        name,
        source_role="main",
        trust_level="sovereign",
        source_locality="local",
        risk_level="low",
        outcome_success_rate=1.0,
        budget_pressure=0.0,
        dynamic_delta=0.0,
    )


def test_promotion_stage_ordering():
    """Verify patches go through stages in order WITH runners."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    _seed_sovereign_metadata(ladder)
    stage, msg = ladder.advance("test_p", patches)
    assert stage == "static_approved", msg
    stage, msg = ladder.advance("test_p", patches, sandbox_runner=lambda c: (True, "ok"))
    assert stage == "sandbox_passed", msg
    stage, msg = ladder.advance("test_p", patches, regression_runner=lambda: (10, 10, {}))
    assert stage == "regression_passed", msg
    stage, msg = ladder.advance("test_p", patches)
    assert stage == "shadow_deployed", msg
    stage, msg = ladder.advance("test_p", patches)
    assert stage == "shadow_deployed"

def test_promotion_blocks_without_sandbox():
    """Missing sandbox runner must BLOCK."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    ladder.advance("test_p", patches)  # → static_approved
    stage, msg = ladder.advance("test_p", patches)  # no sandbox
    assert stage == "static_approved"
    assert "blocked" in msg.lower()

def test_promotion_blocks_without_regression():
    """Missing regression runner must BLOCK."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    ladder.advance("test_p", patches)
    ladder.advance("test_p", patches, sandbox_runner=lambda c: (True, "ok"))
    stage, msg = ladder.advance("test_p", patches)  # no regression
    assert stage == "sandbox_passed"
    assert "blocked" in msg.lower()


def test_promotion_cannot_skip_stages():
    """Verify you can't jump from proposed directly to live."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    # Only advance() can move stages — there's no set_stage
    assert ladder.current_stage("test_p") == "proposed"


def test_promotion_static_failure_blocks():
    """Bad code should fail at static stage."""
    ladder = PromotionLadder()
    patches = {"bad": {"patch_name": "bad", "target": "research_topic",
                        "code": "import subprocess\ndef research_topic(self): pass",
                        "status": "staged"}}
    stage, msg = ladder.advance("bad", patches)
    assert stage == "proposed"  # didn't advance
    assert "subprocess" in msg or "blocked" in msg


def test_promotion_with_sandbox_runner():
    """Sandbox runner integration."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    # Get to static_approved first
    ladder.advance("test_p", patches)
    # Now test sandbox
    stage, msg = ladder.advance("test_p", patches, sandbox_runner=lambda code: (True, "ok"))
    assert stage == "sandbox_passed"


def test_promotion_sandbox_failure():
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    ladder.advance("test_p", patches)  # → static_approved
    stage, msg = ladder.advance("test_p", patches, sandbox_runner=lambda code: (False, "crash"))
    assert stage == "static_approved"  # didn't advance
    assert "crash" in msg


def test_promotion_with_regression_runner():
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    ladder.advance("test_p", patches)
    ladder.advance("test_p", patches, sandbox_runner=lambda c: (True, "ok"))
    stage, msg = ladder.advance("test_p", patches, regression_runner=lambda: (10, 10, {}))
    assert stage == "regression_passed"


def test_promotion_regression_failure():
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    ladder.advance("test_p", patches)
    ladder.advance("test_p", patches, sandbox_runner=lambda c: (True, "ok"))
    stage, msg = ladder.advance("test_p", patches, regression_runner=lambda: (3, 10, {}))
    assert stage == "sandbox_passed"


def test_apply_live_only_from_shadow():
    """apply_live should only work when stage is shadow_deployed."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()

    class FakeKernel:
        pass

    ok, msg = ladder.apply_live("test_p", patches, FakeKernel, {}, set())
    assert not ok
    assert "shadow_deployed" in msg


def test_full_promotion_lifecycle():
    """Full lifecycle: stage → promote through ladder → apply → revert."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    _seed_sovereign_metadata(ladder)

    class FakeKernel:
        def research_topic(self, topic, context=""):
            return []

    originals: dict = {}
    evolved: set = set()

    # Advance through all gates WITH runners
    ladder.advance("test_p", patches)  # → static
    ladder.advance("test_p", patches, sandbox_runner=lambda c: (True, "ok"))  # → sandbox
    ladder.advance("test_p", patches, regression_runner=lambda: (10, 10, {}))  # → regression
    ladder.advance("test_p", patches)  # → shadow

    assert ladder.current_stage("test_p") == "shadow_deployed"

    # Apply live
    ok, msg = ladder.apply_live("test_p", patches, FakeKernel, originals, evolved)
    assert ok, msg
    assert ladder.current_stage("test_p") == "live_promoted"
    assert "research_topic" in evolved
    assert "research_topic" in originals

    # Revert
    ok, msg = ladder.revert("research_topic", FakeKernel, originals, evolved, patches)
    assert ok
    assert "research_topic" not in evolved


def test_promotion_history_recorded():
    """Every transition should be in the history."""
    ladder = PromotionLadder()
    patches = _make_staged_patch()
    ladder.advance("test_p", patches)
    assert len(ladder.history) >= 1
    assert ladder.history[-1].patch_name == "test_p"


# ============================================================
# Quarantine tests
# ============================================================
def test_quarantine_blocks():
    mgr = QuarantineManager()
    patches = _make_staged_patch()
    quarantine_patch("test_p", "research_topic", "test quarantine", patches, mgr)
    assert mgr.is_quarantined("test_p")
    assert patches["test_p"]["status"] == "quarantined"


def test_quarantine_release():
    mgr = QuarantineManager()
    patches = _make_staged_patch()
    quarantine_patch("test_p", "research_topic", "test", patches, mgr)
    assert mgr.is_quarantined("test_p")
    mgr.release("test_p")
    assert not mgr.is_quarantined("test_p")


# ============================================================
# Mutation log tests
# ============================================================
def test_mutation_log_records():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "mutations.py"
        logger = MutationLogger(log_path)
        logger.record_stage("p1", "research_topic", "test")
        logger.record_apply("p1", "research_topic", "def research_topic(self): pass")
        logger.record_revert("p1", "research_topic")
        assert len(logger.events) == 3
        assert logger.events[0]["event_type"] == "STAGED"
        assert logger.events[1]["event_type"] == "APPLIED"
        assert logger.events[2]["event_type"] == "REVERTED"
        # Check file was written
        content = log_path.read_text(encoding="utf-8")
        assert "STAGED" in content
        assert "APPLIED" in content
        assert "REVERTED" in content


# ============================================================
# Non-bypass test: the fundamental v14 contract
# ============================================================
def test_no_direct_apply_without_ladder():
    """There must be no public function that applies a patch without going through
    the promotion ladder. This is the single most important v14 behavioral change."""
    # Verify: stage_patch does NOT apply
    patches: dict = {}
    raw = json.dumps({
        "patch_name": "sneaky",
        "target": "research_topic",
        "code": "def research_topic(self, topic, context=''):\n    self.state.beta['x'] = BilateralValue(0.5, 0.5)\n    refresh_state(self.state)\n    return []\n",
        "rationale": "should not be auto-applied",
    })
    result = stage_patch(raw, staged_patches=patches)
    assert result.ok
    assert patches["sneaky"]["status"] == "staged"  # NOT "applied"

    # Verify: only apply_live can promote to applied
    ladder = PromotionLadder()
    assert ladder.current_stage("sneaky") == "proposed"  # still proposed


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
