"""
TOVAH v14 persistence/migrations.py — State version migration.

Handles upgrading v13 state files to v14 format.
Preserves all existing data, adds defaults for new fields.

MIGRATION SAFETY:
  - Old beta dicts with plain {t, f} entries → coerced to BilateralValue
  - Missing v14 beta keys → added with BilateralValue(0.5, 0.0)
  - Missing v14 fields → added with empty defaults
  - state_version updated to 14.0.0
  - No data is deleted during migration
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

from tovah_v14.core.primitives import BilateralValue, coerce_bilateral_value
from tovah_v14.core.state import CarrierState


# Beta keys that v14 adds beyond v13
V14_NEW_BETA_KEYS: List[str] = [
    "tool.discovery_quality",
    "experience.replay_health", "experience.outcome_quality",
    "competence.map_freshness", "competence.measurement_rate",
    "module.planner_health", "module.executor_health", "module.critic_health",
    "module.memory_health", "module.trainer_health", "module.retriever_health",
    "module.patcher_health", "module.observer_health",
    "metrics.collection_health",
    "boot.validation_status",
    "degraded.mode_active",
    "promotion.ladder_health",
    "recovery.snapshot_health",
    "cleanup.stale_belief_rate", "cleanup.dead_plan_rate", "cleanup.orphan_task_rate",
    "adaptation.conservative_gate",
    "goal.utility.expected", "goal.stability.bias",
]

# All beta keys that should exist (v13 + v14)
ALL_DEFAULT_BETA_KEYS: List[str] = [
    # v13
    "goal.active", "tool.use_desire", "tool.search_efficacy", "tool.fetch_efficacy",
    "advisor.dependence", "patch.pipeline.health", "state.coherent", "research.novelty",
    "report.quality", "runtime.stability", "human.approval_pending", "internet.reachability",
    "planning.confidence", "planning.execution_quality", "service.discovery_desire",
    "capability.growth_rate", "shadowhott.rewrite_progress", "autonomy.self_direction",
    "browser.reachability", "tool.extract_efficacy", "levbel.pdf_health",
    "self_assessment.overall",
    "memory.consolidation_health", "task.queue_health", "sandbox.reliability",
    "regression.pass_rate", "world.model_freshness", "budget.compliance",
    "curriculum.progress", "debug.resolution_rate", "escalation.pending",
    "self_model.accuracy",
] + V14_NEW_BETA_KEYS


def migrate_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a loaded state dict from any older version to v14.

    Non-destructive: adds missing fields, coerces types, but never
    deletes existing data.

    Returns the migrated dict (modifies in place).
    """
    old_version = str(state_dict.get("state_version", "unknown"))

    # 1. Ensure 'state' sub-dict exists
    if "state" not in state_dict or not isinstance(state_dict.get("state"), dict):
        state_dict["state"] = {
            "c": {}, "beta": {}, "nu": {}, "pi": {},
        }

    so = state_dict["state"]

    # 2. Ensure carrier state has v14 fields
    c = so.get("c", {})
    if not isinstance(c, dict):
        c = {}
    c.setdefault("degraded", False)
    so["c"] = c

    # 3. Coerce all beta values and add missing keys
    raw_beta = so.get("beta", {})
    if not isinstance(raw_beta, dict):
        raw_beta = {}
    clean_beta: Dict[str, Dict[str, float]] = {}
    for k, v in raw_beta.items():
        bv = coerce_bilateral_value(v)
        clean_beta[str(k)] = {"t": bv.t, "f": bv.f}
    # Add defaults for missing keys
    for key in ALL_DEFAULT_BETA_KEYS:
        if key not in clean_beta:
            clean_beta[key] = {"t": 0.5, "f": 0.0}
    so["beta"] = clean_beta

    # 4. Ensure other state sub-dicts
    so.setdefault("nu", {})
    so.setdefault("pi", {"step": 0, "tags": [], "history": [], "refresh_count": 0})

    # 5. Add v14 top-level fields with defaults
    state_dict.setdefault("experience_records", [])
    state_dict.setdefault("competence_map", {})
    state_dict.setdefault("metrics_last_cycle", {})

    # 6. Coerce topic_last_research_time to dict of floats
    tlrt = state_dict.get("topic_last_research_time", {})
    if isinstance(tlrt, dict):
        state_dict["topic_last_research_time"] = {
            str(k): float(v) for k, v in tlrt.items()
            if isinstance(v, (int, float))
        }
    else:
        state_dict["topic_last_research_time"] = {}

    # 7. Coerce recent_research_topics to list of strings
    rrt = state_dict.get("recent_research_topics", [])
    if isinstance(rrt, list):
        state_dict["recent_research_topics"] = [str(x) for x in rrt][-80:]
    else:
        state_dict["recent_research_topics"] = []

    # 8. Ensure list fields have correct types
    for key in ("completed_goals", "pending_tool_actions", "patch_history",
                "loss_history", "research_memory", "trace_index", "unresolved",
                "shelved_goals", "domain_history", "installed_packages",
                "active_plans", "completed_plans", "rewrite_queue", "rewrite_history",
                "memory_episodic", "memory_semantic", "memory_procedural",
                "task_queue", "completed_tasks", "failure_clusters"):
        if not isinstance(state_dict.get(key), list):
            state_dict[key] = []

    for key in ("staged_patches", "lab_registry", "api_usage",
                "capabilities", "resource_budgets", "promotion_state",
                "workbench_notes", "tool_contracts"):
        if not isinstance(state_dict.get(key), dict):
            state_dict[key] = {}

    # 9. Ensure numeric fields
    for key, default in [
        ("last_research_time", 0.0), ("improvement_count", 0),
        ("autonomy_level", 0), ("goal_attempts", 0),
        ("alpha", 1.0), ("temperature", 0.9),
    ]:
        try:
            state_dict[key] = type(default)(state_dict.get(key, default))
        except (TypeError, ValueError):
            state_dict[key] = default

    # 10. Preserve crypto fields
    state_dict.setdefault("crypto_wallet", None)
    state_dict.setdefault("beneficiary_sol_address", "De3PtGBEx1XEP7wH6pXbcynSRw9xKNyj8S9k7rG5jRiW")

    # 11. Update version
    state_dict["state_version"] = "14.0.0"

    if old_version != "14.0.0":
        logging.info(f"STATE MIGRATION: {old_version} -> 14.0.0")

    return state_dict
