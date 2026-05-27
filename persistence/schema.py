"""
TOVAH v14 persistence/schema.py — State schema validation.

Defines the canonical kernel state schema and validates loaded state
against it. Catches field drift, type mismatches, and version issues.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from tovah_v14.config.constants import VERSION


# Required top-level fields with expected types
REQUIRED_FIELDS: Dict[str, type] = {
    "state": dict,
    "state_version": str,
    "completed_goals": list,
    "pending_tool_actions": list,
    "staged_patches": dict,
    "patch_history": list,
    "loss_history": list,
    "research_memory": list,
    "improvement_count": (int, float),
    "autonomy_level": (int, float),
    "alpha": (int, float),
    "temperature": (int, float),
}

# Optional fields (present in full saves, may be absent in older versions)
OPTIONAL_FIELDS: Set[str] = {
    "current_goal", "goal_attempts", "shelved_goals", "domain_history",
    "installed_packages", "api_usage", "lab_registry", "crypto_wallet",
    "beneficiary_sol_address", "profile_name", "topic_last_research_time",
    "recent_research_topics", "active_plans", "completed_plans",
    "capabilities", "rewrite_queue", "rewrite_history",
    "memory_episodic", "memory_semantic", "memory_procedural",
    "task_queue", "completed_tasks", "failure_clusters",
    "resource_budgets", "curriculum", "promotion_state",
    "workbench_notes", "tool_contracts", "trace_index", "unresolved",
    "last_research_time", "experience_records", "competence_map",
    "metrics_last_cycle",
    "kernel_ecology",
}

# Required sub-fields within state dict
STATE_SUB_FIELDS: Set[str] = {"c", "beta", "nu", "pi"}


@dataclass
class SchemaValidationResult:
    """Result of state schema validation."""
    ok: bool = True
    version_match: bool = True
    missing_required: List[str] = field(default_factory=list)
    type_mismatches: List[str] = field(default_factory=list)
    missing_state_sub: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def validate_state_schema(state_dict: Dict[str, Any]) -> SchemaValidationResult:
    """Validate a loaded state dict against the canonical schema.

    Does NOT modify the dict. Returns a report.
    """
    result = SchemaValidationResult()

    if not isinstance(state_dict, dict):
        result.ok = False
        result.missing_required.append("state_dict is not a dict")
        return result

    # Version check
    ver = state_dict.get("state_version", "unknown")
    result.version_match = str(ver).startswith("14.")
    if not result.version_match:
        result.warnings.append(f"state_version={ver}, expected 14.x")

    # Required fields
    for fname, expected_type in REQUIRED_FIELDS.items():
        if fname not in state_dict:
            result.missing_required.append(fname)
        elif not isinstance(state_dict[fname], expected_type):
            result.type_mismatches.append(f"{fname}: expected {expected_type.__name__ if isinstance(expected_type, type) else expected_type}, got {type(state_dict[fname]).__name__}")

    # State sub-structure
    so = state_dict.get("state")
    if isinstance(so, dict):
        for sub in STATE_SUB_FIELDS:
            if sub not in so:
                result.missing_state_sub.append(sub)
        # Beta should be a dict
        if not isinstance(so.get("beta"), dict):
            result.type_mismatches.append("state.beta: expected dict")
    elif so is not None:
        result.type_mismatches.append(f"state: expected dict, got {type(so).__name__}")

    result.ok = len(result.missing_required) == 0 and len(result.type_mismatches) == 0
    if result.missing_state_sub:
        result.ok = False

    return result
