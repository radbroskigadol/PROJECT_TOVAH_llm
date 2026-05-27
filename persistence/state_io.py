"""
TOVAH v14 persistence/state_io.py — State serialization and deserialization.

SEMANTIC PRESERVATION:
  The saved JSON field inventory matches v13 exactly. All v13 state files
  will load without error. New v14 fields use defaults when absent.

MIGRATION SAFETY:
  - Beta values may be plain dicts in old state files → coerced via coerce_bilateral_value
  - Missing fields get defaults
  - Unknown fields are preserved in a _extra dict for forward compatibility

The save_state/load_state functions here are PURE — they take/return data dicts.
The kernel wraps them with its own attribute mapping.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


def save_json(path: Path, obj: Any) -> bool:
    """Atomic JSON write: write to .tmp, then replace.

    Uses os.replace for atomic rename (works on Windows and Unix).
    Returns True on success.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(str(tmp), str(path))
        return True
    except Exception as e:
        logging.error(f"save_json failed for {path}: {e}")
        # Try to clean up tmp
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from file. Returns default if file missing or corrupt."""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"load_json failed for {path}: {e}")
        return default


def save_state_to_file(path: Path, state_dict: Dict[str, Any]) -> bool:
    """Save kernel state dict to JSON file.

    The state_dict should already be serialized to plain dicts/lists/floats.
    This function handles atomic write.
    """
    return save_json(path, state_dict)


def load_state_from_file(path: Path) -> Dict[str, Any]:
    """Load kernel state dict from JSON file.

    Returns empty dict if file missing or corrupt.
    Does NOT coerce bilateral values — that is the caller's responsibility
    (since it requires importing core.primitives, and we want this module
    to remain a thin I/O layer).
    """
    return load_json(path, {})


# ============================================================
# State serialization helpers
# ============================================================

def serialize_state_for_save(
    *,
    # All fields from v13 save_state, in order
    completed_goals: list,
    pending_tool_actions: list,
    staged_patches: dict,
    patch_history: list,
    loss_history: list,
    research_memory: list,
    trace_index: list,
    unresolved: list,
    last_research_time: float,
    improvement_count: int,
    autonomy_level: int,
    current_goal: Any,
    goal_attempts: int,
    shelved_goals: list,
    domain_history: list,
    installed_packages: list,
    state_snapshot: dict,
    alpha: float,
    temperature: float,
    api_usage: dict,
    lab_registry: dict,
    crypto_wallet: Any,
    beneficiary_sol_address: str,
    profile_name: str,
    topic_last_research_time: dict,
    recent_research_topics: list,
    active_plans: list,
    completed_plans: list,
    capabilities: dict,
    rewrite_queue: list,
    rewrite_history: list,
    memory_episodic: list,
    memory_semantic: list,
    memory_procedural: list,
    task_queue: list,
    completed_tasks: list,
    failure_clusters: list,
    resource_budgets: dict,
    curriculum: list,
    promotion_state: dict,
    workbench_notes: dict,
    state_version: str,
    tool_contracts: dict,
    # v14 additions (optional with defaults)
    experience_records: list | None = None,
    competence_map: dict | None = None,
    metrics_last_cycle: dict | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the canonical state dict for persistence.

    This ensures every save has the complete field inventory.
    """
    d: Dict[str, Any] = {
        "completed_goals": completed_goals,
        "pending_tool_actions": pending_tool_actions,
        "staged_patches": staged_patches,
        "patch_history": patch_history,
        "loss_history": loss_history,
        "research_memory": research_memory,
        "trace_index": trace_index,
        "unresolved": unresolved,
        "last_research_time": last_research_time,
        "improvement_count": improvement_count,
        "autonomy_level": autonomy_level,
        "current_goal": current_goal,
        "goal_attempts": goal_attempts,
        "shelved_goals": shelved_goals,
        "domain_history": domain_history,
        "installed_packages": installed_packages,
        "state": state_snapshot,
        "alpha": alpha,
        "temperature": temperature,
        "api_usage": api_usage,
        "lab_registry": lab_registry,
        "crypto_wallet": crypto_wallet,
        "beneficiary_sol_address": beneficiary_sol_address,
        "profile_name": profile_name,
        "topic_last_research_time": topic_last_research_time,
        "recent_research_topics": recent_research_topics,
        "active_plans": active_plans,
        "completed_plans": completed_plans,
        "capabilities": capabilities,
        "rewrite_queue": rewrite_queue,
        "rewrite_history": rewrite_history,
        "memory_episodic": memory_episodic,
        "memory_semantic": memory_semantic,
        "memory_procedural": memory_procedural,
        "task_queue": task_queue,
        "completed_tasks": completed_tasks,
        "failure_clusters": failure_clusters,
        "resource_budgets": resource_budgets,
        "curriculum": curriculum,
        "promotion_state": promotion_state,
        "workbench_notes": workbench_notes,
        "state_version": state_version,
        "tool_contracts": tool_contracts,
    }
    # v14 optional additions
    if experience_records is not None:
        d["experience_records"] = experience_records
    if competence_map is not None:
        d["competence_map"] = competence_map
    if metrics_last_cycle is not None:
        d["metrics_last_cycle"] = metrics_last_cycle
    # Forward compat: preserve any extra keys
    d.update(extra)
    return d


# ============================================================
# Kernel ecology persistence helpers (v16 path)
# ============================================================

def save_kernel_ecology_to_file(path: Path, ecology_dict: Dict[str, Any]) -> bool:
    """Save separate kernel-ecology state without disturbing the v13 core file."""
    return save_json(path, ecology_dict)


def load_kernel_ecology_from_file(path: Path) -> Dict[str, Any]:
    """Load kernel-ecology state. Returns empty dict if missing/corrupt."""
    return load_json(path, {})


def serialize_kernel_ecology_state(
    *,
    boot_mode: str,
    packet_log: list,
    child_kernel_registry: dict,
    hub_state: dict | None = None,
    subkernel_states: dict | None = None,
    goal_lineage: dict | None = None,
    module_proposals: list | None = None,
    resource_requests: list | None = None,
    tool_requests: list | None = None,
    tool_access_decisions: list | None = None,
    worker_budget_decisions: list | None = None,
    module_policy_decisions: list | None = None,
    memory_sync_requests: list | None = None,
    promotion_requests: list | None = None,
    promotion_gate_log: list | None = None,
    module_registry_state: dict | None = None,
    message_bus_state: dict | None = None,
    promotion_ladder_state: dict | None = None,
    delegation_state: dict | None = None,
    branch_provenance: dict | None = None,
    branch_checkpoints: list | None = None,
    cluster_registry_state: dict | None = None,
    cluster_trust_state: dict | None = None,
    node_identity_state: dict | None = None,
    distributed_queue_state: dict | None = None,
    worker_budget_state: dict | None = None,
) -> Dict[str, Any]:
    return {
        "boot_mode": boot_mode,
        "packet_log": list(packet_log),
        "child_kernel_registry": dict(child_kernel_registry),
        "hub_state": dict(hub_state or {}),
        "subkernel_states": dict(subkernel_states or {}),
        "goal_lineage": dict(goal_lineage or {}),
        "module_proposals": list(module_proposals or []),
        "resource_requests": list(resource_requests or []),
        "tool_requests": list(tool_requests or []),
        "tool_access_decisions": list(tool_access_decisions or []),
        "worker_budget_decisions": list(worker_budget_decisions or []),
        "module_policy_decisions": list(module_policy_decisions or []),
        "memory_sync_requests": list(memory_sync_requests or []),
        "promotion_requests": list(promotion_requests or []),
        "promotion_gate_log": list(promotion_gate_log or []),
        "module_registry_state": dict(module_registry_state or {}),
        "message_bus_state": dict(message_bus_state or {}),
        "promotion_ladder_state": dict(promotion_ladder_state or {}),
        "delegation_state": dict(delegation_state or {}),
        "branch_provenance": dict(branch_provenance or {}),
        "branch_checkpoints": list(branch_checkpoints or []),
        "cluster_registry_state": dict(cluster_registry_state or {}),
        "cluster_trust_state": dict(cluster_trust_state or {}),
        "node_identity_state": dict(node_identity_state or {}),
        "distributed_queue_state": dict(distributed_queue_state or {}),
        "worker_budget_state": dict(worker_budget_state or {}),
    }
