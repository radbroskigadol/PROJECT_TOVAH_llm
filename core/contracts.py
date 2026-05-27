"""
TOVAH v14 core/contracts.py — Method contracts for test-first patching.

Every method in ALLOWED_PATCH_TARGETS has a MethodContract.
Patches are validated against their contract before staging.

This is the foundation of the shift from text-first to contract-first
self-modification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class MethodContract:
    """Formal interface contract for a kernel method.

    Patches must satisfy:
    - required_params present in function signature
    - return_type documented (descriptive, not yet runtime-enforced)
    - must_update_beta: code must contain self.state.beta
    - must_call_refresh: code must call refresh_state
    - forbidden_patterns: strings that must NOT appear in code
    - required_patterns: strings that MUST appear in code
    """
    name: str
    required_params: List[str]
    optional_params: List[str] = field(default_factory=list)
    return_type: str = "Any"
    must_update_beta: bool = True
    must_call_refresh: bool = True
    forbidden_patterns: List[str] = field(default_factory=list)
    required_patterns: List[str] = field(default_factory=list)
    description: str = ""


# --- Patch target sets (v13 compat: exact sets preserved) ---
ALLOWED_PATCH_TARGETS: Set[str] = {
    "research_topic", "assess_patch_json", "_generate_next_goal", "_rank_tool_candidates",
    "_classify_query_intent", "_decompose_goal_into_queries", "_process_natural_instruction",
    "_chat_with_advisor", "run_capability_tests", "send_email_report",
    "_strategic_plan", "_discover_free_services", "_shadowhott_rewrite_method",
    "_autonomous_cycle", "_decide_research_targets", "_lab_growth_cycle",
    "_adapt_research_code", "_discover_tool_opportunities",
}

ALLOWED_INJECT_TARGETS: Set[str] = ALLOWED_PATCH_TARGETS | {
    "browser_action", "extract_text", "_extract_pdf_text_local",
    "_summarize_pdf_text_local", "_tool_use_desire", "_score_local_results",
}

# Extension targets: declared slots for new methods. Being in this set does NOT
# authorize creation — explicit create_new=True approval is always required.
EXTENSION_TARGETS: Set[str] = {
    "_extract_pdf_text_local", "_summarize_pdf_text_local",
    "_tool_use_desire", "_score_local_results",
}

# Unified target surface: every target that can ever be live-applied
ALLOWED_TARGETS_UNIFIED: Set[str] = ALLOWED_INJECT_TARGETS | EXTENSION_TARGETS

PROTECTED_METHODS: Set[str] = {
    "__init__", "run_loop", "apply_staged_patch", "stage_patch", "save_state", "load_state",
    "_check_david_commands", "update_mirror", "_load_shadow_weights", "_save_shadow_weights",
    "_load_credentials", "_save_credentials", "protect_core_goal", "_perform_tool_action",
}


# Common forbidden patterns for patches that misuse kernel internals
_COMMON_FORBIDDEN = [
    "self.tool_layer",       # wrong attribute name
    "self.tools['",          # tools is ToolLayer, not dict
    'self.tools["',
    "self.state.CarrierState",  # wrong: it's self.state.c
    "self.state.carrier_state",
    "from BilateralValue import",   # BV is in scope via exec env
    "from bilateral_or import",
    "from bilateral_recover import",
    "from refresh_state import",
]


# --- Contract Registry ---
CONTRACT_REGISTRY: Dict[str, MethodContract] = {
    "research_topic": MethodContract(
        name="research_topic",
        required_params=["topic"],
        optional_params=["context"],
        return_type="Dict[str, Any]",
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Multi-step research with structured synthesis. Returns synthesis dict.",
    ),
    "assess_patch_json": MethodContract(
        name="assess_patch_json",
        required_params=["raw"],
        return_type="BilateralValue",
        must_update_beta=False,
        must_call_refresh=False,
        forbidden_patterns=_COMMON_FORBIDDEN,
        required_patterns=["BilateralValue("],
        description="Assess patch JSON string. Returns BilateralValue(truth, falsity).",
    ),
    "_generate_next_goal": MethodContract(
        name="_generate_next_goal",
        required_params=[],
        return_type="Optional[Dict[str, Any]]",
        forbidden_patterns=_COMMON_FORBIDDEN + ["return next_goal"],
        description="Generate next goal dict based on subsystem weakness.",
    ),
    "_rank_tool_candidates": MethodContract(
        name="_rank_tool_candidates",
        required_params=["candidates_or_topic"],
        optional_params=["query"],
        return_type="List",
        must_update_beta=False,
        must_call_refresh=False,
        description="Rank tool candidates by intent, state bias, and overlap.",
    ),
    "_classify_query_intent": MethodContract(
        name="_classify_query_intent",
        required_params=["text"],
        return_type="str",
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Classify query intent: url_fetch, github_repo, github_file, paper_lookup, broad_research.",
    ),
    "_decompose_goal_into_queries": MethodContract(
        name="_decompose_goal_into_queries",
        required_params=["goal_text"],
        return_type="List[str]",
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Decompose a goal into 3-6 concrete search queries.",
    ),
    "_process_natural_instruction": MethodContract(
        name="_process_natural_instruction",
        required_params=["instruction"],
        return_type="str",
        description="Process a natural language instruction from David.",
    ),
    "_chat_with_advisor": MethodContract(
        name="_chat_with_advisor",
        required_params=["prompt"],
        return_type="str",
        forbidden_patterns=_COMMON_FORBIDDEN + ["class ShadowScoreCompat"],
        description="Route prompt through advisor APIs. Returns response string.",
    ),
    "run_capability_tests": MethodContract(
        name="run_capability_tests",
        required_params=[],
        return_type="Tuple[int, int, Dict[str, bool]]",
        must_update_beta=False,
        must_call_refresh=False,
        description="Run capability test battery. Returns (passed, total, details).",
    ),
    "send_email_report": MethodContract(
        name="send_email_report",
        required_params=[],
        optional_params=["msg"],
        return_type="None",
        must_update_beta=False,
        must_call_refresh=False,
        description="Send email report (placeholder).",
    ),
    "_strategic_plan": MethodContract(
        name="_strategic_plan",
        required_params=["objective"],
        optional_params=["max_steps"],
        return_type="Optional[StrategicPlan]",
        forbidden_patterns=_COMMON_FORBIDDEN + ["return {", "beliefs="],
        description="Create StrategicPlan for an objective. Returns StrategicPlan or None.",
    ),
    "_discover_free_services": MethodContract(
        name="_discover_free_services",
        required_params=[],
        optional_params=["domain"],
        return_type="List[Dict[str, Any]]",
        description="Discover new free API services.",
    ),
    "_shadowhott_rewrite_method": MethodContract(
        name="_shadowhott_rewrite_method",
        required_params=["method_name"],
        return_type="Tuple[bool, str]",
        description="Rewrite a method to be ShadowHoTT-native via advisor.",
    ),
    "_autonomous_cycle": MethodContract(
        name="_autonomous_cycle",
        required_params=[],
        return_type="None",
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="One full autonomous cycle: goal, research, plan, patch, promote.",
    ),
    "_decide_research_targets": MethodContract(
        name="_decide_research_targets",
        required_params=[],
        return_type="List[Tuple[str, str]]",
        description="Decide what to research this cycle. Returns [(query, context)].",
    ),
    "_lab_growth_cycle": MethodContract(
        name="_lab_growth_cycle",
        required_params=["topic", "results"],
        return_type="List[str]",
        description="Lab growth from research results. Returns status messages.",
    ),
    "_adapt_research_code": MethodContract(
        name="_adapt_research_code",
        required_params=[],
        return_type="List[Dict[str, Any]]",
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Transform research into typed staged proposals with metadata.",
    ),
    "_discover_tool_opportunities": MethodContract(
        name="_discover_tool_opportunities",
        required_params=["topic", "results"],
        return_type="List[Dict[str, str]]",
        description="Discover tool-building opportunities from research results.",
    ),
    # --- Extension targets (Audit S-5) ---
    # These slots accept create_new injections; they live in EXTENSION_TARGETS
    # and historically had no contract entry — preflight only emitted a
    # warning. The contracts below give them light structural requirements
    # so create-new injections are still validated.
    "_extract_pdf_text_local": MethodContract(
        name="_extract_pdf_text_local",
        required_params=["path"],
        return_type="Dict[str, Any]",
        must_update_beta=False,
        must_call_refresh=False,
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Local PDF text extraction. Returns {'ok': bool, 'text': str, ...}.",
    ),
    "_summarize_pdf_text_local": MethodContract(
        name="_summarize_pdf_text_local",
        required_params=["text"],
        optional_params=["max_chars"],
        return_type="Dict[str, Any]",
        must_update_beta=False,
        must_call_refresh=False,
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Local-only PDF text summarization. Returns {'summary': str, ...}.",
    ),
    "_tool_use_desire": MethodContract(
        name="_tool_use_desire",
        required_params=["context"],
        return_type="float",
        must_update_beta=False,
        must_call_refresh=False,
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Compute desire score for using a tool given context. Returns scalar in [0,1].",
    ),
    "_score_local_results": MethodContract(
        name="_score_local_results",
        required_params=["results"],
        optional_params=["query"],
        return_type="List",
        must_update_beta=False,
        must_call_refresh=False,
        forbidden_patterns=_COMMON_FORBIDDEN,
        description="Score local search/research results. Returns ranked list.",
    ),
}


def verify_patch_contract(target: str, code: str) -> tuple[bool, list[str]]:
    """Verify that patch code satisfies the contract for its target method.

    Returns (ok, errors).
    This checks structural requirements only — it does not execute the code.
    """
    contract = CONTRACT_REGISTRY.get(target)
    if contract is None:
        return False, [f"no contract for target: {target}"]

    errors: list[str] = []

    # Check forbidden patterns
    for pat in contract.forbidden_patterns:
        if pat in code:
            errors.append(f"forbidden pattern: {pat}")

    # Check required patterns
    for pat in contract.required_patterns:
        if pat not in code:
            errors.append(f"required pattern missing: {pat}")

    # Check bilateral state discipline
    if contract.must_update_beta and "self.state.beta" not in code:
        errors.append("contract requires self.state.beta updates")
    if contract.must_call_refresh and "refresh_state(" not in code:
        errors.append("contract requires refresh_state() call")

    # Check function signature has required params
    # This is a lightweight check — looks for 'def target(self, param1, param2...'
    import re
    sig_match = re.search(rf"def\s+{re.escape(target)}\s*\(([^)]*)\)", code)
    if sig_match:
        sig = sig_match.group(1)
        sig_params = [p.strip().split(":")[0].split("=")[0].strip()
                      for p in sig.split(",") if p.strip()]
        # Remove 'self'
        sig_params = [p for p in sig_params if p != "self"]
        for rp in contract.required_params:
            if rp not in sig_params and f"*{rp}" not in sig_params and "**" not in sig:
                errors.append(f"required param missing from signature: {rp}")

    return len(errors) == 0, errors
