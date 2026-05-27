"""
TOVAH v14 kernel/preflight.py — Startup preflight and command registry.
Preflight with severity: errors block, warnings inform.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
from tovah_v14.config.paths import STATE_FILE, SNAPSHOT_DIR, COMMAND_FILE, KERNEL_ECOLOGY_FILE, BRANCH_CHECKPOINT_DIR, MEMORY_PROVENANCE_FILE, CLUSTER_REGISTRY_FILE, CLUSTER_TRUST_FILE, NODE_IDENTITY_FILE
from tovah_v14.config.constants import VERSION

@dataclass
class CommandEntry:
    name: str
    handler: str
    status: str
    approval_required: bool = False
    autonomous_eligible: bool = False
    notes: str = ""

COMMAND_REGISTRY: Dict[str, CommandEntry] = {}

def _reg(name, handler, status="active", approval=False, auto=False, notes=""):
    COMMAND_REGISTRY[name] = CommandEntry(name, handler, status, approval, auto, notes)

_reg("STATUS", "run_capability_tests+get_self_summary", notes="bounded regression")
_reg("PATCHES", "staged_patches listing")
_reg("REJECT_PATCH:<n>", "staged_patches status update")
_reg("GOAL:<text>", "current_goal setter", auto=True)
_reg("CLEAR_GOAL", "current_goal=None")
_reg("CANCEL_GOAL", "current_goal=None", status="alias")
_reg("REMOVE_GOAL", "current_goal=None", status="alias")
_reg("COMPLETE_GOAL", "completed_goals append")
_reg("RESEARCH:<topic>", "research_topic()", auto=True,
     notes="multi-step structured synthesis with contradiction detection")
_reg("TOOL:<tool>|<arg>|<arg2>", "_perform_tool_action")
_reg("RUN_LAB:<topic>", "research_topic()+lab output")
_reg("LAB_STATUS", "lab_registry listing")
_reg("PROMOTE_TOOL:<n>", "LAB_STAGED->LAB_ACTIVE", approval=True)
_reg("REJECT_TOOL:<n>|<reason>", "LAB_STAGED->LAB_REJECTED")
_reg("EXPORT_MATH:<title>", "write markdown to LAB_MATH")
_reg("REQUEST_ACCOUNT:<body>", "_append_need()")
_reg("CREDS:<svc>|<user>|<pass>", "_save_credentials()", approval=True)
_reg("STAGE_PATCH:<json>", "stage_patch() via authoritative preflight", auto=True)
_reg("APPLY_PATCH:<n>", "apply_staged_patch() via ladder", approval=True,
     notes="bounded regression; blocks honestly if runners absent")
_reg("INGEST_LEVBEL", "DEFERRED", status="deferred", notes="PDF ingestion not migrated")
_reg("TRACE", "_write_report_and_trace()")
_reg("PAUSE", "state.c.paused=True")
_reg("RESUME", "state.c.paused=False")
_reg("DEVNOTE:<text>", "MODEL_NOTES_FILE append")
_reg("RUN_CODE:<code>", "exec()", approval=True)
_reg("ADD_PATCH_TARGET:<n>", "ALLOWED_PATCH_TARGETS.add()")
_reg("ADD_INJECT_TARGET:<n>", "ALLOWED_INJECT_TARGETS.add()")
_reg("REMOVE_BLOCK:<n>", "BLOCKED_*.discard()")
_reg("LIST_BLOCKS", "BLOCKED_* listing")
_reg("AUTO_PROMOTE", "promotion_ladder.advance() for all staged",
     notes="blocks if runners absent")
_reg("UNPROTECT:<n>", "PROTECTED_METHODS.discard()", approval=True)
_reg("REVERT_PATCH:<target>", "promotion_ladder.revert()")
_reg("REMOVE_LAST_PATCH", "revert most recent")
_reg("LIST_PATCHED", "_evolved_method_names listing")
_reg("PATCH_REJECTS", "_patch_reject_log listing")
_reg("INSTALL:<pkg>", "_pip_install()", approval=True)
_reg("ROLLBACK_MODEL", "_rollback_model()")
_reg("SNAPSHOT", "_save_model_snapshot()")
_reg("TRAINING_PHASE", "loss + phase report")
_reg("INJECT_METHOD:<target>\\n<code>", "direct_inject_method() -> stage+promote",
     approval=True, notes="no direct binding; create-new requires explicit flag")
_reg("INJECT_TOOL:<code>", "inject_tool_via_advisor()", approval=True)
_reg("PLAN:<objective>", "PlanManager.add()")
_reg("LIST_PLANS", "plan_manager.active listing")
_reg("LIST_SERVICES", "_free_services listing")
_reg("DISCOVER_SERVICES:<domain>", "_discover_free_services()",
     notes="deterministic ranking + advisor-enhanced")
_reg("ACTIVATE_SERVICE:<n>", "_free_services status update")
_reg("LIST_CAPABILITIES", "_capabilities listing")
_reg("REWRITE_METHOD:<n>", "_rewrite_queue append")
_reg("REWRITE_STATUS", "_rewrite_queue + history listing")
_reg("MEMORY_STATUS", "memory_store.counts()")
_reg("MEMORY_QUERY:<kind>|<query>", "memory_query()")
_reg("TASK_STATUS", "task_queue listing")
_reg("TASK_CREATE:<goal>", "task_queue.create()")
_reg("REGRESSION", "run_capability_tests()", notes="bounded tier")
_reg("WORLD_STATE", "readout_state()")
_reg("SELF_MODEL", "update_self_model()", notes="richer: competence/budgets/module-health/blocked-growth")
_reg("BUDGETS", "budget_manager.budgets listing")
_reg("CURRICULUM", "_curriculum listing")
_reg("ESCALATIONS", "_escalation_log listing")
_reg("WORKBENCH_NOTE:<topic>|<content>", "_workbench_notes setter")
_reg("WORKBENCH_SEARCH:<query>", "_workbench_notes search")
_reg("FAILURE_CLUSTERS", "cluster_failures()")
_reg("SANDBOX_EXEC:<code>", "restricted exec")
_reg("OFFLINE_GROWTH", "train_shadow_step() manual")
_reg("PROMOTE_LADDER:<patch>", "promotion_ladder.advance()")
_reg("PDF_STATUS", "PdfReader availability")
_reg("HUB_STATUS", "get_kernel_ecology_summary()", notes="kernel ecology status")
_reg("HUB_REVERT", "hub snapshot revert", approval=True, notes="kernel ecology status")
_reg("SPAWN_SUBKERNEL[:specialization|mission]", "_spawn_subkernel()", approval=True, notes="governed subkernel spawn")
_reg("LIST_SUBKERNELS", "subkernel registry listing")
_reg("KERNEL_PACKET_LOG", "kernel packet log tail")
_reg("LIST_BRANCH_CHECKPOINTS", "list_branch_checkpoints()", notes="persisted ecology branch checkpoints")
_reg("SAVE_BRANCH_CHECKPOINT", "checkpoint_branch_ecology()", approval=True, notes="persist ecology branch checkpoint")
_reg("MEMORY_PROVENANCE", "branch_provenance.summary()", notes="branch-memory provenance graph")
_reg("MODULE_REGISTRY", "module_registry.summary()", notes="governed module registry")
_reg("MODULE_BUS", "message_bus.summary()", notes="module bus routes + recent traffic")
_reg("PROMOTION_REQUESTS", "promotion request packet log")
_reg("MODULE_POLICY", "module promotion policy decisions")
_reg("PROMOTION_GATES", "trust-weighted promotion gate decisions")
_reg("WORKER_POLICIES", "worker role policy profiles")
_reg("WORKER_BUDGETS", "worker-role budget and lease gate decisions")
_reg("TOOL_ACCESS_DECISIONS", "tool_access_decisions listing", notes="worker-role tool access decisions")
_reg("EXPORT_CORPUS:<dir>", "export pretraining corpus from current state", notes="optional |since_cycle suffix; e.g. EXPORT_CORPUS:/tmp/corpus|0")
_reg("TRAIN_FROM_CORPUS[:<dir>[|<epochs>[|<batch>[|<save>]]]]",
     "pretrain shadow model from JSONL corpus shards",
     approval=True,
     notes="defaults: dir=tovah_corpus/stream, epochs=1, batch=8. Updates live shadow model.")
_reg("CLUSTER_STATUS", "cluster_registry.summary()", notes="local/distributed-ready node registry summary")
_reg("NODE_TRUST", "cluster_trust.summary()", notes="cluster trust ledger summary")
_reg("NODE_IDENTITY", "node_identity.summary()", notes="local/distributed node identity summary")
_reg("CLUSTER_DELEGATIONS", "delegation_manager.summary()", notes="cluster-aware delegation lease summary")
_reg("DISTRIBUTED_QUEUE", "distributed_queue.summary()", notes="cluster-aware delegation queue summary")

@dataclass
class PreflightResult:
    ok: bool = True
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

def run_preflight(kernel: Any) -> PreflightResult:
    result = PreflightResult()
    for name in kernel.tools.builtins:
        exists = hasattr(kernel.tools, name) and callable(getattr(kernel.tools, name))
        result.checks[f"tool.{name}"] = exists
        if not exists:
            result.errors.append(f"ToolLayer missing: {name}")
    has_handler = hasattr(kernel, "_check_david_commands") and callable(kernel._check_david_commands)
    result.checks["command_handler"] = has_handler
    if not has_handler:
        result.errors.append("_check_david_commands missing")
    valid_version = kernel._state_version == VERSION or kernel._state_version.startswith("14.")
    result.checks["state_version"] = valid_version
    if not valid_version:
        result.warnings.append(f"state version mismatch: {kernel._state_version}")
    import os
    result.checks["state_file_dir"] = STATE_FILE.parent.exists()
    result.checks["snapshot_dir"] = SNAPSHOT_DIR.exists()
    result.checks["command_file_dir"] = COMMAND_FILE.parent.exists()
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    result.checks["thread_cap_set"] = threads != "unset"
    result.checks["shadow_model"] = kernel.shadow_model is not None
    result.checks["shadow_optimizer"] = kernel.shadow_optimizer is not None
    from tovah_v14.modules.manifests import MODULE_MANIFESTS
    for role, manifest in MODULE_MANIFESTS.items():
        for method_name in manifest.methods:
            exists = hasattr(kernel, method_name) and callable(getattr(kernel, method_name, None))
            result.checks[f"manifest.{role}.{method_name}"] = exists
            if not exists:
                result.errors.append(f"manifest method missing: {role}.{method_name}")
    result.checks["budget_manager"] = isinstance(kernel.budget_manager.budgets, dict) and len(kernel.budget_manager.budgets) > 0
    result.checks["memory_store"] = kernel.memory_store is not None
    result.checks["task_queue"] = kernel.task_queue is not None
    result.checks["experience_store"] = kernel.experience_store is not None
    # Kernel ecology scaffold (non-blocking while integration is staged)
    try:
        from tovah_v14.kernel.packet import KernelPacket
        from tovah_v14.kernel.kernel_roles import KernelRole
        from tovah_v14.kernel.kernel_policy import ownership_summary
        from tovah_v14.kernel.hub_kernel import HubKernel
        from tovah_v14.kernel.subkernel import Subkernel
        from tovah_v14.cluster.registry import ClusterRegistry
        from tovah_v14.cluster.trust import ClusterTrustLedger
        from tovah_v14.selfmodel.node_identity import NodeIdentity
        result.checks["ecology.packet_schema"] = callable(getattr(KernelPacket, "to_dict", None))
        result.checks["ecology.kernel_roles"] = len(list(KernelRole)) >= 3
        result.checks["ecology.kernel_policy"] = "main" in ownership_summary()
        result.checks["ecology.hub_kernel"] = HubKernel is not None
        result.checks["ecology.subkernel"] = Subkernel is not None
        result.checks["ecology.kernel_summary"] = hasattr(kernel, "get_kernel_ecology_summary")
        result.checks["ecology.packet_log"] = hasattr(kernel, "kernel_packet_log") and isinstance(kernel.kernel_packet_log, list)
        result.checks["ecology.boot_mode"] = getattr(kernel, "boot_mode", "") in {"main_only", "main_with_hub", "distributed_ready"}
        result.checks["ecology.module_registry"] = hasattr(kernel, "module_registry") and hasattr(kernel.module_registry, "summary")
        result.checks["ecology.message_bus"] = hasattr(kernel, "message_bus") and hasattr(kernel.message_bus, "summary")
        result.checks["ecology.provenance_graph"] = hasattr(kernel, "branch_provenance") and hasattr(kernel.branch_provenance, "summary")
        result.checks["ecology.checkpoint_method"] = hasattr(kernel, "checkpoint_branch_ecology")
        result.checks["ecology.cluster_registry"] = hasattr(kernel, "cluster_registry") and isinstance(kernel.cluster_registry, ClusterRegistry)
        result.checks["ecology.cluster_trust"] = hasattr(kernel, "cluster_trust") and isinstance(kernel.cluster_trust, ClusterTrustLedger)
        result.checks["ecology.node_identity"] = hasattr(kernel, "node_identity") and isinstance(kernel.node_identity, NodeIdentity)
        result.checks["ecology.distributed_queue"] = hasattr(kernel, "distributed_queue") and hasattr(kernel.distributed_queue, "summary")
        result.checks["ecology.ecology_file_dir"] = KERNEL_ECOLOGY_FILE.parent.exists()
        result.checks["ecology.branch_checkpoint_dir"] = BRANCH_CHECKPOINT_DIR.exists()
        result.checks["ecology.provenance_file_dir"] = MEMORY_PROVENANCE_FILE.parent.exists()
        result.checks["ecology.cluster_registry_file_dir"] = CLUSTER_REGISTRY_FILE.parent.exists()
        result.checks["ecology.cluster_trust_file_dir"] = CLUSTER_TRUST_FILE.parent.exists()
        result.checks["ecology.node_identity_file_dir"] = NODE_IDENTITY_FILE.parent.exists()
    except Exception as e:
        result.checks["ecology.packet_schema"] = False
        result.checks["ecology.kernel_roles"] = False
        result.checks["ecology.kernel_policy"] = False
        result.checks["ecology.hub_kernel"] = False
        result.checks["ecology.subkernel"] = False
        result.checks["ecology.kernel_summary"] = False
        result.checks["ecology.packet_log"] = False
        result.checks["ecology.boot_mode"] = False
        result.checks["ecology.module_registry"] = False
        result.checks["ecology.message_bus"] = False
        result.warnings.append(f"kernel ecology scaffold unavailable: {e}")
    for cmd in ("HUB_STATUS", "HUB_REVERT", "LIST_SUBKERNELS", "KERNEL_PACKET_LOG", "MODULE_REGISTRY", "MODULE_BUS", "PROMOTION_REQUESTS", "MODULE_POLICY", "PROMOTION_GATES", "WORKER_POLICIES", "WORKER_BUDGETS", "TOOL_ACCESS_DECISIONS", "MEMORY_PROVENANCE", "SAVE_BRANCH_CHECKPOINT", "LIST_BRANCH_CHECKPOINTS", "CLUSTER_STATUS", "NODE_TRUST", "DISTRIBUTED_QUEUE", "CLUSTER_DELEGATIONS", "NODE_IDENTITY"):
        result.checks[f"command.{cmd}"] = cmd in COMMAND_REGISTRY
    result.checks["command.SPAWN_SUBKERNEL"] = any(name.startswith("SPAWN_SUBKERNEL") for name in COMMAND_REGISTRY)
    # Patch-target drift (BLOCKING)
    from tovah_v14.core.contracts import ALLOWED_PATCH_TARGETS, CONTRACT_REGISTRY, EXTENSION_TARGETS
    for target in ALLOWED_PATCH_TARGETS:
        if target in EXTENSION_TARGETS:
            continue
        exists = hasattr(kernel, target) and callable(getattr(kernel, target, None))
        result.checks[f"patch_target.{target}"] = exists
        if not exists:
            result.errors.append(f"BLOCKING: orphaned patch target: {target}")
    for ct in CONTRACT_REGISTRY:
        if ct in EXTENSION_TARGETS:
            continue
        exists = hasattr(kernel, ct) and callable(getattr(kernel, ct, None))
        result.checks[f"contract.{ct}"] = exists
        if not exists:
            result.errors.append(f"BLOCKING: contract target missing: {ct}")
    result.ok = len(result.errors) == 0
    passed = sum(1 for v in result.checks.values() if v)
    total = len(result.checks)
    level = logging.INFO if result.ok else logging.WARNING
    logging.log(level, f"PREFLIGHT: {passed}/{total}, {len(result.warnings)} warn, {len(result.errors)} err")
    return result
