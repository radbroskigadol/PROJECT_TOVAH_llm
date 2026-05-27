"""
TOVAH v14 modules/manifests.py — Module manifests.

Each module has a manifest declaring:
- role, health key, version
- owned methods
- dependencies on other modules
- interface inputs/outputs
- status

These are SKELETON interfaces for future distribution.
We do NOT fake a completed distributed runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModuleManifest:
    """Describes a kernel module's interface contract."""
    role: str
    health_key: str
    version: str = "14.2.6"
    methods: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    interface_inputs: List[str] = field(default_factory=list)
    interface_outputs: List[str] = field(default_factory=list)
    status: str = "active"


MODULE_MANIFESTS: Dict[str, ModuleManifest] = {
    "planner": ModuleManifest(
        role="planner", health_key="module.planner_health",
        methods=["_strategic_plan", "_generate_next_goal", "_decompose_goal_into_queries"],
        depends_on=["memory_manager", "retriever"],
        interface_inputs=["goal_text", "state_snapshot", "competence_map"],
        interface_outputs=["StrategicPlan", "goal_dict", "query_list"],
    ),
    "executor": ModuleManifest(
        role="executor", health_key="module.executor_health",
        methods=["_perform_tool_action", "_autonomous_cycle"],
        depends_on=["planner", "patcher", "memory_manager"],
        interface_inputs=["action_dict", "plan_step"],
        interface_outputs=["ToolResult", "cycle_outcome"],
    ),
    "critic": ModuleManifest(
        role="critic", health_key="module.critic_health",
        methods=["assess_patch_json", "run_capability_tests", "_self_assess"],
        depends_on=["observer"],
        interface_inputs=["patch_json", "state_snapshot"],
        interface_outputs=["BilateralValue", "test_results", "InvariantReport"],
    ),
    "memory_manager": ModuleManifest(
        role="memory_manager", health_key="module.memory_health",
        methods=["_store_memory", "_consolidate", "_forget"],
        depends_on=["retriever"],
        interface_inputs=["memory_entry", "kind", "query"],
        interface_outputs=["MemoryEntry", "conflict_records"],
    ),
    "trainer": ModuleManifest(
        role="trainer", health_key="module.trainer_health",
        methods=["_train_shadow_step"],
        depends_on=[],
        interface_inputs=["corpus", "budgets"],
        interface_outputs=["loss", "phase", "invariants"],
    ),
    "retriever": ModuleManifest(
        role="retriever", health_key="module.retriever_health",
        methods=["_rank_tool_candidates", "_classify_query_intent", "memory_query"],
        depends_on=[],
        interface_inputs=["query_text", "candidates"],
        interface_outputs=["ranked_list", "intent_class", "memory_results"],
    ),
    "patcher": ModuleManifest(
        role="patcher", health_key="module.patcher_health",
        methods=["stage_patch", "apply_staged_patch", "_adapt_research_code"],
        depends_on=["critic", "observer"],
        interface_inputs=["patch_json", "patch_name"],
        interface_outputs=["staging_result", "apply_result"],
    ),
    "observer": ModuleManifest(
        role="observer", health_key="module.observer_health",
        methods=["_write_report_and_trace", "_self_assess", "build_report"],
        depends_on=[],
        interface_inputs=["state_snapshot", "loss_history"],
        interface_outputs=["InvariantReport", "StateReport", "trace_id"],
    ),
}
