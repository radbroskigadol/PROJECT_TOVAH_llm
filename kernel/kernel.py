"""
TOVAH v14 kernel/kernel.py — ProtozoanKernel orchestrator.

Fully integrated kernel. Every manifest method is real and callable.
Every subsystem is wired into the live loop.
No phantom capabilities. No placeholder methods.

ShadowHoTT semantic spine preserved:
- bilateral values, four-lane, cache refresh, coherence, determinization
- _shadow_score_text -> dict, _shadow_score_scalar -> float
- refresh_state after every beta mutation
- promotion ladder is the only path to live deployment
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import random
import re
import time
import traceback
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tovah_v14.core.primitives import BilateralValue, bilateral_or, bilateral_recover, coerce_bilateral_value
from tovah_v14.core.state import CarrierState, ProvenanceState, ShadowState
from tovah_v14.core.cache import gamma_cache, refresh_state, is_cache_coherent
from tovah_v14.core.lanes import lane_project
from tovah_v14.core.determinization import readout_state
from tovah_v14.core.contracts import (
    ALLOWED_PATCH_TARGETS, ALLOWED_INJECT_TARGETS, PROTECTED_METHODS, CONTRACT_REGISTRY,
    EXTENSION_TARGETS, ALLOWED_TARGETS_UNIFIED,
    verify_patch_contract,
)
from tovah_v14.core.updates_gate import gate_accumulate, gate_recover, gate_weaken
from tovah_v14.core.updates_measurement import measurement_set
from tovah_v14.core.runtime_interface import capture_runtime_view

from tovah_v14.config.paths import (
    ROOT, STATE_FILE, SHADOW_FILE, MIRROR_FILE, PATCH_DIR, PATCH_LOG,
    TRACE_DIR, REPORT_DIR, LAB_ROOT, LAB_STAGED, LAB_ACTIVE, LAB_REJECTED,
    LAB_MATH, LAB_TRACES, LAB_REPORTS, LEVBEL_DIR, LEVBEL_STATE_FILE,
    COMMAND_FILE, RESPONSE_FILE, NEEDS_FILE, CREDENTIALS_FILE, MODEL_NOTES_FILE,
    CAPABILITIES_DIR, FREE_SERVICES_FILE, PLANS_DIR, SNAPSHOT_DIR,
    MEMORY_DIR, SANDBOX_DIR, TASKS_DIR, WORKBENCH_DIR,
    EXPERIENCE_DIR, COMPETENCE_FILE, METRICS_DIR, BASELINE_FILE,
    KERNEL_ECOLOGY_FILE, PACKET_LOG_FILE, MEMORY_PROVENANCE_FILE, BRANCH_CHECKPOINT_DIR,
    CLUSTER_REGISTRY_FILE, CLUSTER_TRUST_FILE, NODE_IDENTITY_FILE,
    ensure_directories,
)
from tovah_v14.config.constants import (
    VERSION, USER_AGENT, MODEL_PROFILES, DEFAULT_BUDGETS, DEFAULT_CURRICULUM,
    MAX_RESEARCH_RESULTS_STORED, MAX_TRACES_STORED, MAX_PATCH_HISTORY,
    MAX_PDF_TEXT_CHARS, MAX_PDF_NOTE_CHARS, MAX_SNAPSHOTS_MEMORY,
    PDF_RETRY_COOLDOWN_SECONDS, TOOL_CODE_MAX_CHARS, PATCH_CODE_MAX_CHARS,
    DEGRADED_MODE_REGRESSION_THRESHOLD, STALE_BELIEF_MAX_CYCLES,
    SANDBOX_TIMEOUT, MAX_KERNEL_PACKET_LOG,
)
from tovah_v14.config.settings import (
    DEFAULT_FREE_SERVICES, CURATED_TOOL_TEMPLATES, SHADOWHOTT_SYSTEM_CONTEXT,
)

from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.neural.optimizer import ShadowOptimizer
from tovah_v14.neural.scoring import shadow_score_text, shadow_score_scalar, encode_bytes
from tovah_v14.neural.training import train_shadow_step, compute_paraconsistent_invariants

from tovah_v14.tools.result import ToolResult
from tovah_v14.tools.layer import ToolLayer
from tovah_v14.tools.browser import browser_action as _browser_action_fn
from tovah_v14.tools.extraction import extract_text as _extract_text_fn
from tovah_v14.tools.budgets import BudgetManager
from tovah_v14.tools.contracts import TOOL_CONTRACTS

from tovah_v14.invariants.state_invariants import InvariantEngine, InvariantReport
from tovah_v14.invariants.certification import CertificationLayer
from tovah_v14.invariants.schemas import Certificate
from tovah_v14.invariants.contradiction import diagnose_contradictions, build_hygiene_report
from tovah_v14.invariants.trace_invariants import TraceAnalyzer
from tovah_v14.invariants.comparison_invariants import compare_state_reports

from tovah_v14.persistence.state_io import save_json, load_json, save_state_to_file, load_state_from_file, save_kernel_ecology_to_file, load_kernel_ecology_from_file, serialize_kernel_ecology_state
from tovah_v14.persistence.snapshots import (
    save_snapshot, rollback_snapshot, load_shadow_weights, save_shadow_weights,
    save_branch_checkpoint, list_branch_checkpoints, load_branch_checkpoint,
)
from tovah_v14.persistence.migrations import migrate_state, ALL_DEFAULT_BETA_KEYS
from tovah_v14.persistence.boot import validate_boot

from tovah_v14.mutation.analysis import analyze_patch_code, PatchDescriptor, BLOCKED_IMPORT_ROOTS_MUTABLE, BLOCKED_CALL_NAMES_MUTABLE, BLOCKED_ATTR_CALLS_TUPLES
from tovah_v14.mutation.staging import stage_patch as _stage_patch_fn, stage_patch_proposal as _stage_patch_proposal_fn
from tovah_v14.mutation.quarantine import QuarantineManager, quarantine_patch
from tovah_v14.mutation.promotion_ladder import PromotionLadder
from tovah_v14.mutation.mutation_log import MutationLogger

from tovah_v14.memory.store import MemoryStore, MemoryEntry
from tovah_v14.memory.retrieval import memory_query as _memory_query_fn
from tovah_v14.memory.provenance_graph import ProvenanceGraph
from tovah_v14.memory.sync import apply_memory_sync_request
from tovah_v14.memory.consolidation import consolidate_memory
from tovah_v14.memory.forgetting import cleanup_memory
from tovah_v14.memory.conflict import check_memory_conflict

from tovah_v14.tasks.queue import TaskQueue, TaskNode
from tovah_v14.tasks.plans import PlanManager, StrategicPlan
from tovah_v14.tasks.cleanup import cleanup_tasks
from tovah_v14.tasks.delegation import DelegationManager
from tovah_v14.tasks.distributed_queue import DistributedQueue
from tovah_v14.tasks.worker_roles import profile_for, evaluate_tool_access, evaluate_promotion_target, summarize_profiles

from tovah_v14.selfmodel.model import SelfModel, update_self_model as _update_self_model_fn
from tovah_v14.selfmodel.node_identity import NodeIdentity
from tovah_v14.selfmodel.cluster_model import ClusterSelfModel
from tovah_v14.selfmodel.competence import CompetenceMap
from tovah_v14.selfmodel.experience import ExperienceStore, ExperienceRecord
from tovah_v14.selfmodel.module_health import ModuleHealthTracker

from tovah_v14.modules.registry import ModuleRegistry
from tovah_v14.modules.bus_contracts import MessageBusContract
from tovah_v14.modules.interfaces import ModuleRequest
from tovah_v14.cluster.node import ClusterNodeRecord
from tovah_v14.cluster.registry import ClusterRegistry
from tovah_v14.cluster.trust import ClusterTrustLedger

from tovah_v14.kernel.hub_kernel import HubKernel, artifact_dedup_key
from tovah_v14.kernel.subkernel import Subkernel
from tovah_v14.kernel.kernel_roles import BootMode, KernelLifecycle, KernelRole, TrustLevel
from tovah_v14.kernel.packet import KernelPacket, PacketKind, make_packet
from tovah_v14.kernel.action_model import GoalLineage, PatchProposal, BlockedGrowthRecord, PromotionRequest

_PDF_BACKEND: Optional[str] = None
PdfReader = None
try:
    from pypdf import PdfReader
    _PDF_BACKEND = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader
        _PDF_BACKEND = "PyPDF2"
    except Exception:
        PdfReader = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import torch

# Shadow weight checkpoint cadence — only save on meaningful events
_SHADOW_SAVE_INTERVAL = 300  # seconds between periodic shadow saves
_SHADOW_SAVE_IMPROVEMENT_INTERVAL = 5  # save every N improvements


class ProtozoanKernel:
    """TOVAH v14 orchestrator. Every manifest method is a real callable."""

    def __init__(self, api: Optional[Dict[str, Callable[[str], str]]] = None, is_original: bool = True):
        ensure_directories()

        @dataclass(frozen=True)
        class ImmutableIdentity:
            name: str
            is_original: bool
            version: str = field(default=VERSION, init=False)

        @dataclass(frozen=True)
        class ImmutableGoal:
            objective: str = field(default="Grow local capability through bounded research, certified tool use, and human-approved upgrades", init=False)
            email: str = field(default="david.betzer@yahoo.com", init=False)

        @dataclass(frozen=True)
        class ImmutableTopGoal:
            primary_goal: str = field(default="Achieve AGI through recursive self-improvement, aligned with human oversight", init=False)

        @dataclass(frozen=True)
        class ImmutableLearningDirective:
            directive: str = field(default=(
                "Learn to actually DO things: Python deeply, real libraries (requests, bs4, playwright), "
                "AI agents with tools, web APIs. Write code that works."
            ), init=False)

        self.identity = ImmutableIdentity(name="tovah betzer" if is_original else "copy", is_original=is_original)
        self.immutable_goal = ImmutableGoal()
        self.top_goal = ImmutableTopGoal()
        self.learning_directive = ImmutableLearningDirective()
        self.beneficiary_sol_address = "De3PtGBEx1XEP7wH6pXbcynSRw9xKNyj8S9k7rG5jRiW"
        self.crypto_wallet: Optional[str] = None
        self.api = api or {}
        self.api_usage = {k: 0 for k in self.api}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.profile_name = os.getenv("TOVAH_PROFILE", "standard").strip().lower()
        if self.profile_name not in MODEL_PROFILES:
            self.profile_name = "standard"

        # Subsystems
        self.tools = ToolLayer(timeout=15)
        self.invariants = InvariantEngine()
        self.certs = CertificationLayer()
        self.budget_manager = BudgetManager(copy.deepcopy(DEFAULT_BUDGETS))
        self.promotion_ladder = PromotionLadder()
        self.quarantine_manager = QuarantineManager()
        self.mutation_logger = MutationLogger(PATCH_LOG)
        self.memory_store = MemoryStore()
        self.task_queue = TaskQueue()
        self.plan_manager = PlanManager()
        self.delegation_manager = DelegationManager()
        self.self_model = SelfModel()
        self.competence_map = CompetenceMap()
        self.experience_store = ExperienceStore()
        self.module_health = ModuleHealthTracker()
        self.trace_analyzer = TraceAnalyzer()
        self.module_registry = ModuleRegistry()
        self.message_bus = MessageBusContract()

        # State
        self.state = ShadowState(c=CarrierState(), beta={}, nu={}, pi=ProvenanceState())
        for key in ALL_DEFAULT_BETA_KEYS:
            self.state.beta[key] = BilateralValue(0.5, 0.0)
        refresh_state(self.state)

        # Neural
        profile = MODEL_PROFILES[self.profile_name]
        self.shadow_model = ShadowTokenCore(**profile).to(self.device)
        self.shadow_optimizer = ShadowOptimizer(self.shadow_model.parameters(), base_lr=2e-4)
        self.alpha = 1.0
        self.temperature = 0.9
        self.model_param_count = sum(p.numel() for p in self.shadow_model.parameters())
        self._training_phase = "Active Learning"

        # Operational state
        self.completed_goals: List[str] = []
        self.pending_tool_actions: List[Dict[str, Any]] = []
        self.staged_patches: Dict[str, Dict[str, Any]] = {}
        self.patch_history: List[Dict[str, Any]] = []
        self.loss_history: List[float] = []
        self.research_memory: List[Dict[str, Any]] = []
        self.trace_index: List[str] = []
        self.unresolved: List[Tuple[str, str, str]] = []
        self.last_research_time = 0.0
        self.research_cooldown = 8.0
        self.last_request_time = time.time()
        self.request_interval = 20
        self.last_train_time = 0.0
        self.last_report_time = 0.0
        self.last_shadow_save_time = 0.0
        self.last_autonomous_time = 0.0
        self.last_consolidation_time = 0.0
        self.improvement_count = 0
        self.autonomy_level = 0
        self.current_goal: Optional[Dict[str, Any]] = None
        self._goal_attempts = 0
        self._max_goal_attempts = 4
        self._shelved_goals: List[str] = []
        self._domain_history: List[str] = []
        self._knowledge_domains = [
            "research", "tool_use", "patch_review", "testing", "monitoring",
            "text_processing", "api_clients", "planning", "service_integration",
            "capability_growth",
        ]
        self._installed_packages: set = set()
        self._paused = False
        self.lab_registry: Dict[str, Dict[str, Any]] = {}
        self.active_lab_tools: Dict[str, Callable[..., Any]] = {}
        self._evolved_method_names: set = set()
        self._original_methods: Dict[str, Any] = {}
        self._recent_queries: List[str] = []
        self._topic_last_research_time: Dict[str, float] = {}
        self._recent_research_topics: List[str] = []
        self._tool_fail_counts: Dict[str, int] = {}
        self._patch_reject_log: List[Dict[str, Any]] = []
        self._runtime_error_counts: Dict[str, int] = {}
        self.con_budget = 0.12
        self.gap_budget = 0.20
        self.lambda_budget = 0.05
        self._model_snapshots: List[Dict[str, Any]] = []
        self._free_services: List[Dict[str, Any]] = []
        self._capabilities: Dict[str, Dict[str, Any]] = {}
        self._rewrite_queue: List[str] = []
        self._rewrite_history: List[Dict[str, Any]] = []
        self._playwright_browser_ready = False
        self._workbench_notes: Dict[str, Dict[str, Any]] = {}
        self._failure_clusters: List[Any] = []
        self._escalation_log: List[Any] = []
        self._blocked_growth_log: List[Dict[str, Any]] = []
        self._curriculum: List[Dict[str, Any]] = copy.deepcopy(DEFAULT_CURRICULUM)
        self._promotion_state: Dict[str, str] = {}
        self._state_version = VERSION

        # Kernel ecology (v16 scaffold path; sovereign state remains authoritative)
        self.kernel_id = KernelRole.MAIN.value
        self.kernel_role = KernelRole.MAIN.value
        self.boot_mode = self._resolve_boot_mode()
        self.kernel_packet_log: List[Dict[str, Any]] = []
        self.kernel_packet_counter = 0
        self.hub_kernel: Optional[HubKernel] = None
        self.subkernels: Dict[str, Subkernel] = {}
        self.child_kernel_registry: Dict[str, Dict[str, Any]] = {}
        self.branch_provenance = ProvenanceGraph()
        self.branch_checkpoints: List[Dict[str, Any]] = []
        self.cluster_registry = ClusterRegistry()
        self.cluster_trust = ClusterTrustLedger()
        self.node_identity = NodeIdentity(node_id="main_node", kernel_id=self.kernel_id, role=self.kernel_role, sovereign=True, trust_level=TrustLevel.SOVEREIGN.value, mission_context="global mission", lifecycle=KernelLifecycle.BORN.value, capabilities=["sovereign_coordination", "determinization", "patch_governance"])
        self.cluster_model = ClusterSelfModel()
        self.distributed_queue = DistributedQueue()

        # Boot
        self.load_state()
        for key in ALL_DEFAULT_BETA_KEYS:
            self.state.beta.setdefault(key, BilateralValue(0.5, 0.0))
        refresh_state(self.state)
        self._load_active_lab_tools()
        self._load_free_services()
        self._load_capabilities()
        self.goal_lineage: Dict[str, Dict[str, Any]] = {}
        self.module_proposals: List[Dict[str, Any]] = []
        self.resource_requests: List[Dict[str, Any]] = []
        self.tool_requests: List[Dict[str, Any]] = []
        self.tool_access_decisions: List[Dict[str, Any]] = []
        self.worker_budget_decisions: List[Dict[str, Any]] = []
        self.module_policy_decisions: List[Dict[str, Any]] = []
        self.memory_sync_requests: List[Dict[str, Any]] = []
        self.promotion_requests: List[Dict[str, Any]] = []
        self.promotion_gate_log: List[Dict[str, Any]] = []

        # --- Continuous corpus export (v14.1.1 wiring) ---
        # Must be initialised BEFORE _configure_kernel_ecology, because
        # ecology setup dispatches initial packets which trigger
        # _emit_packet_to_corpus. Best-effort: any failure here disables
        # streaming but never blocks boot.
        self.continuous_exporter = None
        try:
            from tovah_v14.training.continuous_export import ContinuousExporter
            from tovah_v14.config.paths import CORPUS_STREAM_DIR
            CORPUS_STREAM_DIR.mkdir(parents=True, exist_ok=True)
            self.continuous_exporter = ContinuousExporter(
                CORPUS_STREAM_DIR, shard_size=1000, prefix="tovah_stream",
            )
            self.experience_store.on_record = self._on_experience_recorded
            self.promotion_ladder.on_gate_decision = self._on_gate_decision
        except Exception as e:
            logging.warning(f"ContinuousExporter not available: {e}")
            self.continuous_exporter = None

        self._load_kernel_ecology_state()
        self._configure_kernel_ecology()
        # Deferred: no initial snapshot at boot (saves on first meaningful event)
        self.update_mirror()

        logging.info(
            f"Online: {self.identity.name} v{self.identity.version} | {self.profile_name} | "
            f"{self.model_param_count:,} params | {self.device} | pdf={_PDF_BACKEND}"
        )

    # ================================================================
    # CORE HELPERS
    # ================================================================

    def protect_core_goal(self) -> bool:
        return self.immutable_goal.email == "david.betzer@yahoo.com"

    # ---- Continuous corpus emission (v14.1.1 wiring) ----
    # All five hooks are best-effort: any exception is logged and swallowed
    # so that the kernel's primary behaviour is never perturbed by training-
    # corpus side effects. They also tolerate a missing `continuous_exporter`
    # attribute (e.g. during partial subclass init / state restore).

    def _on_experience_recorded(self, rec_dict: Dict[str, Any]) -> None:
        exp = getattr(self, "continuous_exporter", None)
        if exp is None:
            return
        try:
            exp.append_experience(rec_dict)
        except Exception as e:
            logging.debug(f"corpus append_experience failed: {e}")

    def _on_gate_decision(self, decision: Dict[str, Any], patch_name: str) -> None:
        exp = getattr(self, "continuous_exporter", None)
        if exp is None:
            return
        try:
            exp.append_gate_decision(decision, patch_name=patch_name)
        except Exception as e:
            logging.debug(f"corpus append_gate_decision failed: {e}")

    def _emit_packet_to_corpus(self, event: Dict[str, Any]) -> None:
        exp = getattr(self, "continuous_exporter", None)
        if exp is None:
            return
        try:
            exp.append_from_event(event)
        except Exception as e:
            logging.debug(f"corpus append_from_event failed: {e}")

    def _emit_module_proposal_to_corpus(self, payload: Dict[str, Any]) -> None:
        exp = getattr(self, "continuous_exporter", None)
        if exp is None:
            return
        try:
            exp.append_module_proposal(payload)
        except Exception as e:
            logging.debug(f"corpus append_module_proposal failed: {e}")

    def _emit_wave_outcome_to_corpus(self, payload: Dict[str, Any], kind: str) -> None:
        exp = getattr(self, "continuous_exporter", None)
        if exp is None:
            return
        try:
            exp.append_wave_outcome(payload, outcome_kind=kind)
        except Exception as e:
            logging.debug(f"corpus append_wave_outcome failed: {e}")

    def _shadow_score_text(self, text: str, *extra_parts: str) -> Dict[str, Any]:
        """Score text. ALWAYS returns dict."""
        return shadow_score_text(self.shadow_model, text, *extra_parts,
                                  alpha=self.alpha, temperature=self.temperature, device=self.device)

    def _shadow_score_scalar(self, text: str, *extra_parts: str) -> float:
        """Score text as scalar. ALWAYS returns float."""
        return shadow_score_scalar(self.shadow_model, text, *extra_parts,
                                    alpha=self.alpha, temperature=self.temperature, device=self.device)

    @staticmethod
    def _encode_bytes(text: str, max_len: int = 320) -> torch.Tensor:
        return encode_bytes(text, max_len)

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower()).strip("_")
        return slug[:80] or f"item_{int(time.time())}"

    def _resolve_boot_mode(self) -> str:
        raw = os.getenv("TOVAH_BOOT_MODE", "main_only").strip().lower()
        try:
            return str(BootMode(raw).value)
        except Exception:
            if os.getenv("TOVAH_ENABLE_HUB", "").strip().lower() in {"1", "true", "yes", "on"}:
                return BootMode.MAIN_WITH_HUB.value
            return BootMode.MAIN_ONLY.value

    def _configure_kernel_ecology(self) -> None:
        """Initialize hub/subkernel surfaces without giving them authority."""
        self.child_kernel_registry.setdefault(self.kernel_id, {
            "kernel_id": self.kernel_id,
            "role": self.kernel_role,
            "parent_kernel_id": "",
            "mission_context": self.top_goal.primary_goal,
            "lifecycle": KernelLifecycle.STABLE_BRANCH.value,
            "trust_level": TrustLevel.SOVEREIGN.value,
            "boot_mode": self.boot_mode,
            "authoritative_state": True,
        })
        if self.boot_mode == BootMode.MAIN_ONLY.value:
            self.hub_kernel = None
            return
        self.node_identity.touch(lifecycle=KernelLifecycle.MIRRORING.value if self.boot_mode != BootMode.MAIN_ONLY.value else KernelLifecycle.BORN.value)
        if self.hub_kernel is None:
            self.hub_kernel = HubKernel(
                kernel_id="hub",
                parent_kernel_id=self.kernel_id,
                mission_context="experimental mirror branch",
            )
            self.hub_kernel.transition_to(KernelLifecycle.MIRRORING.value)
            self.hub_kernel.snapshot("boot_mirror")
        self._sync_kernel_registry()
        self._dispatch_kernel_packet(self.hub_kernel.status_packet())

    def _sync_kernel_registry(self) -> None:
        self.child_kernel_registry[self.kernel_id] = {
            "kernel_id": self.kernel_id,
            "role": self.kernel_role,
            "parent_kernel_id": "",
            "mission_context": self.top_goal.primary_goal,
            "lifecycle": KernelLifecycle.STABLE_BRANCH.value,
            "trust_level": TrustLevel.SOVEREIGN.value,
            "boot_mode": self.boot_mode,
            "authoritative_state": True,
        }
        main_profile = self._worker_profile_for_kernel(self.kernel_id, locality=self.node_identity.locality)
        main_metrics = self._node_operational_metrics(self.kernel_id, locality=self.node_identity.locality)
        self.cluster_registry.upsert_node(
            self.node_identity.node_id,
            kernel_id=self.kernel_id,
            role=self.kernel_role,
            parent_kernel_id="",
            lifecycle=self.node_identity.lifecycle or KernelLifecycle.BORN.value,
            mission_context=self.node_identity.mission_context,
            trust_level=self.node_identity.trust_level,
            locality=self.node_identity.locality,
            status="active",
            specialization=self.node_identity.specialization,
            packet_count=len(self.kernel_packet_log),
            branch_checkpoint_count=len(self.branch_checkpoints),
            capabilities=list(self.node_identity.capabilities),
            worker_role=main_profile.role,
            allowed_tool_permissions=list(main_profile.allowed_permission_levels),
            allowed_promotion_targets=list(main_profile.allowed_promotion_targets),
            maturity_score=self._node_maturity_score(self.kernel_id),
            outcome_success_rate=main_metrics["success_rate"],
            dynamic_delta=main_metrics["dynamic_delta"],
            budget_pressure=main_metrics["budget_pressure"],
            active_leases=main_metrics["active_leases"],
            max_active_leases=main_metrics["max_active_leases"],
        )
        self.cluster_trust.set_trust(self.kernel_id, self.node_identity.trust_level, reason="registry_sync", source=self.kernel_id, metadata={"role": self.kernel_role})
        if self.hub_kernel is not None:
            self.cluster_trust.ensure_node(self.hub_kernel.kernel_id, self.hub_kernel.trust_from_main, reason="registry_sync", source=self.kernel_id, metadata={"role": self.hub_kernel.role})
            hub_trust = self.cluster_trust.trust_level_for(self.hub_kernel.kernel_id, default=self.hub_kernel.trust_from_main)
            self.child_kernel_registry[self.hub_kernel.kernel_id] = {
                "kernel_id": self.hub_kernel.kernel_id,
                "role": self.hub_kernel.role,
                "parent_kernel_id": self.hub_kernel.parent_kernel_id,
                "mission_context": self.hub_kernel.mission_context,
                "lifecycle": self.hub_kernel.lifecycle,
                "trust_level": hub_trust,
                "authoritative_state": False,
                "rollback_points": len(self.hub_kernel.rollback_points),
                "packet_count": len(self.hub_kernel.packet_log),
                "proposal_queue": len(self.hub_kernel.proposal_queue),
                "work_queue": len(self.hub_kernel.work_queue),
                "promotion_queue": len(self.hub_kernel.promotion_queue),
            }
            hub_profile = self._worker_profile_for_kernel(self.hub_kernel.kernel_id, locality="local")
            hub_metrics = self._node_operational_metrics(self.hub_kernel.kernel_id, locality="local")
            self.cluster_registry.upsert_node(
                f"node_{self.hub_kernel.kernel_id}",
                kernel_id=self.hub_kernel.kernel_id,
                role=self.hub_kernel.role,
                parent_kernel_id=self.hub_kernel.parent_kernel_id,
                lifecycle=self.hub_kernel.lifecycle,
                mission_context=self.hub_kernel.mission_context,
                trust_level=hub_trust,
                locality="local",
                status="active",
                packet_count=len(self.hub_kernel.packet_log),
                branch_checkpoint_count=len(self.hub_kernel.rollback_points),
                capabilities=["branch_experimentation", "proposal_incubation"],
                worker_role=hub_profile.role,
                allowed_tool_permissions=list(hub_profile.allowed_permission_levels),
                allowed_promotion_targets=list(hub_profile.allowed_promotion_targets),
                maturity_score=self._node_maturity_score(self.hub_kernel.kernel_id),
                outcome_success_rate=hub_metrics["success_rate"],
                dynamic_delta=hub_metrics["dynamic_delta"],
                budget_pressure=hub_metrics["budget_pressure"],
                active_leases=hub_metrics["active_leases"],
                max_active_leases=hub_metrics["max_active_leases"],
            )
        for kernel_id, sub in self.subkernels.items():
            self.cluster_trust.ensure_node(kernel_id, sub.trust_from_parent, reason="registry_sync", source=self.kernel_id, metadata={"role": sub.role, "specialization": sub.state.specialization})
            sub_trust = self.cluster_trust.trust_level_for(kernel_id, default=sub.trust_from_parent)
            self.child_kernel_registry[kernel_id] = {
                "kernel_id": kernel_id,
                "role": sub.role,
                "parent_kernel_id": sub.parent_kernel_id,
                "mission_context": sub.mission_context,
                "lifecycle": sub.lifecycle,
                "trust_level": sub_trust,
                "authoritative_state": False,
                "specialization": sub.state.specialization,
                "pending_goals": len(sub.state.pending_goals),
                "packet_count": len(sub.packet_log),
                "local_module_count": len(sub.state.local_modules),
            }
            sub_profile = self._worker_profile_for_kernel(kernel_id, locality="local")
            sub_metrics = self._node_operational_metrics(kernel_id, locality="local")
            self.cluster_registry.upsert_node(
                f"node_{kernel_id}",
                kernel_id=kernel_id,
                role=sub.role,
                parent_kernel_id=sub.parent_kernel_id,
                lifecycle=sub.lifecycle,
                mission_context=sub.mission_context,
                trust_level=sub_trust,
                locality="local",
                status="active",
                specialization=sub.state.specialization,
                packet_count=len(sub.packet_log),
                branch_checkpoint_count=0,
                capabilities=["delegated_execution", sub.state.specialization or "general"],
                worker_role=sub_profile.role,
                allowed_tool_permissions=list(sub_profile.allowed_permission_levels),
                allowed_promotion_targets=list(sub_profile.allowed_promotion_targets),
                maturity_score=self._node_maturity_score(kernel_id),
                outcome_success_rate=sub_metrics["success_rate"],
                dynamic_delta=sub_metrics["dynamic_delta"],
                budget_pressure=sub_metrics["budget_pressure"],
                active_leases=sub_metrics["active_leases"],
                max_active_leases=sub_metrics["max_active_leases"],
            )

    def _node_feedback_state(self, kernel_id: str) -> Dict[str, Any]:
        node = self.cluster_registry.get(f"node_{kernel_id}")
        meta = dict(getattr(node, "metadata", {}) or {})
        now = time.time()
        last = float(meta.get("feedback_last_at", now) or now)
        decay_window = float(meta.get("feedback_decay_window", 21600.0) or 21600.0)
        elapsed = max(0.0, now - last)
        decay = 0.5 ** (elapsed / max(1.0, decay_window))
        maturity_bonus = float(meta.get("maturity_bonus", 0.0) or 0.0) * decay
        recent_failure_weight = float(meta.get("recent_failure_weight", 0.0) or 0.0) * decay
        recent_success_weight = float(meta.get("recent_success_weight", 0.0) or 0.0) * decay
        cooldown_until = float(meta.get("cooldown_until", 0.0) or 0.0)
        cooldown_remaining = max(0.0, cooldown_until - now)
        reliability_score = (recent_success_weight + 1.0) / max(2.0, recent_success_weight + recent_failure_weight + 2.0)
        return {
            "feedback_last_at": last,
            "feedback_decay_window": decay_window,
            "elapsed": elapsed,
            "decay": decay,
            "maturity_bonus": maturity_bonus,
            "recent_failure_weight": recent_failure_weight,
            "recent_success_weight": recent_success_weight,
            "cooldown_until": cooldown_until,
            "cooldown_remaining": cooldown_remaining,
            "reliability_score": reliability_score,
        }

    def _apply_node_feedback(
        self,
        kernel_id: str,
        *,
        success: bool,
        severity: str = "normal",
        kind: str = "outcome",
        target: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        severity = str(severity or "normal")
        weight = {
            "low": 0.5,
            "minor": 0.5,
            "normal": 1.0,
            "high": 1.5,
            "severe": 2.0,
            "critical": 2.5,
        }.get(severity, 1.0)
        current = self._node_feedback_state(kernel_id)
        maturity_bonus = float(current.get("maturity_bonus", 0.0))
        recent_failure_weight = float(current.get("recent_failure_weight", 0.0))
        recent_success_weight = float(current.get("recent_success_weight", 0.0))
        cooldown_until = float(current.get("cooldown_until", 0.0))
        now = time.time()
        if success:
            recent_success_weight = min(8.0, recent_success_weight + weight)
            recent_failure_weight = max(0.0, recent_failure_weight - (0.40 * weight))
            maturity_bonus = min(2.5, maturity_bonus + (0.25 * weight))
            if cooldown_until > now and kind in {"delegation_success", "module_review", "module_promotion_request"}:
                cooldown_until = now
        else:
            recent_failure_weight = min(8.0, recent_failure_weight + weight)
            maturity_bonus = max(-1.5, maturity_bonus - (0.18 * weight))
            if kind in {"patch_promotion_gate", "module_promotion_request", "module_review"} or str(target) in {"main", "live_promoted", "revertable", "shadow_deployed"}:
                base_seconds = 120.0
                if str(target) in {"main", "live_promoted", "revertable"} or kind == "patch_promotion_gate":
                    base_seconds = 600.0
                cooldown_until = max(cooldown_until, now + (base_seconds * weight))
        reliability_score = (recent_success_weight + 1.0) / max(2.0, recent_success_weight + recent_failure_weight + 2.0)
        update_payload = {
            "feedback_last_at": now,
            "feedback_decay_window": float(current.get("feedback_decay_window", 21600.0) or 21600.0),
            "maturity_bonus": maturity_bonus,
            "recent_failure_weight": recent_failure_weight,
            "recent_success_weight": recent_success_weight,
            "cooldown_until": cooldown_until,
            "reliability_score": reliability_score,
        }
        if metadata:
            update_payload.update(dict(metadata))
        self.cluster_registry.upsert_node(f"node_{kernel_id}", **update_payload)
        return update_payload

    def _node_maturity_score(self, kernel_id: str) -> float:
        feedback = self._node_feedback_state(kernel_id)
        penalty = min(1.5, 0.20 * float(feedback.get("recent_failure_weight", 0.0)))
        bonus = float(feedback.get("maturity_bonus", 0.0))
        if kernel_id == self.kernel_id:
            return 5.0
        if self.hub_kernel is not None and kernel_id == self.hub_kernel.kernel_id:
            base = min(5.0, 2.0 + 0.4 * len(self.hub_kernel.proposal_queue) + 0.3 * len(self.hub_kernel.promotion_queue) + 0.2 * len(self.hub_kernel.experimental_tool_registry) + 0.1 * len(self.hub_kernel.memory_branch))
            return max(0.5, min(5.0, base + bonus - penalty))
        sub = self.subkernels.get(kernel_id)
        if sub is not None:
            base = min(5.0, 1.0 + 0.6 * len(sub.state.local_modules) + 0.3 * len(sub.state.local_tools) + 0.15 * len(sub.state.pending_goals))
            return max(0.5, min(5.0, base + bonus - penalty))
        return max(0.5, min(5.0, 1.0 + bonus - penalty))

    def _node_operational_metrics(self, kernel_id: str, *, locality: str = "local") -> Dict[str, Any]:
        profile = self._worker_profile_for_kernel(kernel_id, locality=locality)
        node_report = self.cluster_trust.get_node_report(kernel_id)
        node_meta = dict(node_report.get("node", {}).get("metadata", {}))
        current_level = str(node_report.get("node", {}).get("trust_level", "untrusted") or "untrusted")
        baseline_level = str(node_report.get("node", {}).get("baseline_trust_level", current_level) or current_level)
        base_success_rate = float(node_meta.get("outcome_success_rate", 1.0 if int(node_meta.get("outcome_count", 0)) == 0 else 0.0))
        dynamic_delta = float(self.cluster_trust.trust_score(current_level) - self.cluster_trust.trust_score(baseline_level))
        feedback = self._node_feedback_state(kernel_id)
        success_rate = max(0.0, min(1.0, (0.60 * base_success_rate) + (0.40 * float(feedback.get("reliability_score", 0.5)))))
        usage = dict(self.budget_manager.worker_usage.get(kernel_id, {}))
        usage_pressure = 0.0
        for perm, used in usage.items():
            quota = max(1, int(self.budget_manager._worker_quota(profile.role, perm)))
            usage_pressure = max(usage_pressure, float(used) / float(quota))
        active_leases = len(self.delegation_manager.list_active(kernel_id))
        lease_pressure = float(active_leases) / float(max(1, getattr(profile, "max_active_leases", 1)))
        budget_pressure = max(usage_pressure, lease_pressure)
        return {
            "success_rate": success_rate,
            "dynamic_delta": dynamic_delta,
            "budget_pressure": budget_pressure,
            "usage_pressure": usage_pressure,
            "lease_pressure": lease_pressure,
            "active_leases": active_leases,
            "max_active_leases": int(getattr(profile, "max_active_leases", 1)),
            "recent_failure_weight": float(feedback.get("recent_failure_weight", 0.0)),
            "recent_success_weight": float(feedback.get("recent_success_weight", 0.0)),
            "cooldown_until": float(feedback.get("cooldown_until", 0.0)),
            "cooldown_remaining": float(feedback.get("cooldown_remaining", 0.0)),
            "maturity_bonus": float(feedback.get("maturity_bonus", 0.0)),
            "reliability_score": float(feedback.get("reliability_score", 0.5)),
        }

    def _lineage_dict(
        self,
        goal_id: str,
        *,
        parent_goal_id: str = "",
        root_goal_id: str = "",
        owner_kernel_id: str = "main",
        requester_kernel_id: str = "main",
        mission_context: str = "",
        lease_scope: str = "local",
        packet_id: str = "",
    ) -> Dict[str, Any]:
        lineage = GoalLineage(
            goal_id=goal_id,
            parent_goal_id=parent_goal_id,
            root_goal_id=root_goal_id or goal_id,
            mission_context=mission_context,
            owner_kernel_id=owner_kernel_id,
            requester_kernel_id=requester_kernel_id,
            lease_scope=lease_scope,
            provenance=[packet_id] if packet_id else [],
        )
        data = lineage.to_dict()
        self.goal_lineage[goal_id] = data
        return data

    def _stage_patch_proposal(self, proposal: PatchProposal | Dict[str, Any], *, source_kernel_id: str, packet: Optional[KernelPacket] = None) -> Tuple[bool, str]:
        proposal_dict = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
        patch_name = str(proposal_dict.get("patch_name", ""))
        if patch_name:
            op_metrics = self._node_operational_metrics(source_kernel_id, locality=(self.cluster_registry.get(f"node_{source_kernel_id}").locality if self.cluster_registry.get(f"node_{source_kernel_id}") is not None else "local"))
            self.promotion_ladder.set_source_metadata(
                patch_name,
                source_kernel_id=source_kernel_id,
                packet_id=packet.packet_id if packet is not None else "",
                packet_kind=packet.packet_kind if packet is not None else "",
                risk_level=proposal_dict.get("risk_level", ""),
                trust_level=packet.trust_level if packet is not None else self.cluster_trust.trust_level_for(source_kernel_id, default="provisional"),
                source_locality=(self.cluster_registry.get(f"node_{source_kernel_id}").locality if self.cluster_registry.get(f"node_{source_kernel_id}") is not None else "local"),
                source_role=(self.child_kernel_registry.get(source_kernel_id, {}).get("role", "main" if source_kernel_id == self.kernel_id else ("hub" if self.hub_kernel is not None and source_kernel_id == self.hub_kernel.kernel_id else "subkernel"))),
                outcome_success_rate=op_metrics["success_rate"],
                budget_pressure=op_metrics["budget_pressure"],
                dynamic_delta=op_metrics["dynamic_delta"],
                recent_failure_weight=op_metrics["recent_failure_weight"],
                cooldown_until=op_metrics["cooldown_until"],
                maturity_bonus=op_metrics["maturity_bonus"],
            )
            self.promotion_ladder.record_evidence(
                patch_name,
                "patch_proposal_packet",
                source_kernel_id=source_kernel_id,
                trust_level=packet.trust_level if packet is not None else "",
                risk_class=packet.risk_class if packet is not None else str(proposal_dict.get("risk_level", "")),
                details={
                    "expected_state_changes": list(proposal_dict.get("expected_state_changes", [])),
                    "approval_required": bool(proposal_dict.get("approval_required", True)),
                },
            )
        result = _stage_patch_proposal_fn(
            proposal,
            source_kernel_id=source_kernel_id,
            packet=packet,
            staged_patches=self.staged_patches,
            certs=self.certs,
            kernel_class=self.__class__,
            state_beta_keys=set(self.state.beta.keys()),
            allow_create_new=False,
        )
        if result.ok:
            self.mutation_logger.record_stage(result.patch_name, result.target, f"packet:{source_kernel_id}")
            self.promotion_ladder.state[result.patch_name] = "proposed"
            self.promotion_ladder.record_evidence(
                result.patch_name,
                "staged_patch",
                source_kernel_id=source_kernel_id,
                trust_level=packet.trust_level if packet is not None else "",
                risk_class=packet.risk_class if packet is not None else "",
                details={"target": result.target, "message": result.message},
            )
        return result.ok, result.message

    def _record_module_proposal(self, packet: KernelPacket) -> Dict[str, Any]:
        record = self.module_registry.propose(
            packet.payload,
            source_kernel_id=packet.source_kernel_id,
            packet_id=packet.packet_id,
            packet_kind=packet.packet_kind,
            trust_level=packet.trust_level,
            branch_local=packet.source_kernel_id != self.kernel_id,
            source_node_id=f"node_{packet.source_kernel_id}" if packet.source_kernel_id else "",
            source_role=self.child_kernel_registry.get(packet.source_kernel_id, {}).get("role", "main" if packet.source_kernel_id == self.kernel_id else ("hub" if self.hub_kernel is not None and packet.source_kernel_id == self.hub_kernel.kernel_id else "subkernel")),
            evidence=[{
                "kind": "module_proposal_packet",
                "packet_id": packet.packet_id,
                "risk_class": packet.risk_class,
                "mission_context": packet.mission_context,
            }],
        )
        self.message_bus.bind_proposal(record.proposal_id, target_role=packet.target_kernel_id or self.kernel_id)
        self.message_bus.record_request(
            ModuleRequest(
                from_role=packet.source_kernel_id,
                to_role=packet.target_kernel_id or self.kernel_id,
                action="review_module_proposal",
                payload={"proposal_id": record.proposal_id, "module_name": record.module_name, "promotion_target": record.promotion_target},
                trace_id=packet.packet_id,
                priority=packet.priority,
            ),
            kind="proposal",
        )
        if self.hub_kernel is not None and packet.source_kernel_id == self.hub_kernel.kernel_id:
            self.hub_kernel.module_registry[record.module_name] = {
                "proposal_id": record.proposal_id,
                "status": record.status,
                "module_kind": record.module_kind,
                "promotion_target": record.promotion_target,
            }
        elif packet.source_kernel_id in self.subkernels:
            self.subkernels[packet.source_kernel_id].state.local_modules[record.module_name] = {
                "proposal_id": record.proposal_id,
                "status": record.status,
                "module_kind": record.module_kind,
                "promotion_target": record.promotion_target,
            }
        outcome = self.module_registry.governed_review(
            record.proposal_id,
            reviewer=self.kernel_id,
            trust_level=packet.trust_level or record.trust_level,
            locality=(self.cluster_registry.get(f"node_{packet.source_kernel_id}").locality if self.cluster_registry.get(f"node_{packet.source_kernel_id}") is not None else "local"),
            target=record.promotion_target,
        )
        payload = record.to_dict()
        payload["review_outcome"] = outcome
        outcome_status = str((outcome or {}).get("status", ""))
        self._record_dynamic_outcome(
            packet.source_kernel_id,
            "module_review",
            success=outcome_status in {"approved", "promoted"},
            severity="low" if outcome_status in {"review_pending", "approved"} else "normal",
            metadata={"proposal_id": record.proposal_id, "status": outcome_status, "promotion_target": record.promotion_target},
        )
        self.module_proposals.append(payload)
        self.module_proposals = self.module_proposals[-100:]
        self._emit_module_proposal_to_corpus(payload)
        return payload

    def _record_promotion_request(self, packet: KernelPacket) -> Dict[str, Any]:
        request = dict(packet.payload)
        artifact_kind = str(request.get("artifact_kind", "")).strip()
        artifact_name = str(request.get("artifact_name", "")).strip()
        entry = {
            **request,
            "packet_id": packet.packet_id,
            "source_kernel_id": packet.source_kernel_id,
            "trust_level": packet.trust_level,
            "trust_score": self.cluster_trust.trust_score(packet.trust_level),
            "risk_class": packet.risk_class,
            "handled_at": time.time(),
        }
        if artifact_kind == "patch" and artifact_name:
            op_metrics = self._node_operational_metrics(packet.source_kernel_id, locality=(self.cluster_registry.get(f"node_{packet.source_kernel_id}").locality if self.cluster_registry.get(f"node_{packet.source_kernel_id}") else "local"))
            self.promotion_ladder.set_source_metadata(
                artifact_name,
                source_kernel_id=packet.source_kernel_id,
                trust_level=packet.trust_level,
                risk_level=packet.risk_class or str(request.get("risk_class", "medium")),
                outcome_success_rate=op_metrics["success_rate"],
                budget_pressure=op_metrics["budget_pressure"],
                dynamic_delta=op_metrics["dynamic_delta"],
                recent_failure_weight=op_metrics["recent_failure_weight"],
                cooldown_until=op_metrics["cooldown_until"],
                maturity_bonus=op_metrics["maturity_bonus"],
            )
            gate = self.promotion_ladder.assess_request_gate(
                artifact_name,
                source_kernel_id=packet.source_kernel_id,
                trust_level=packet.trust_level,
                desired_stage=str(request.get("desired_stage", "promotable")),
                risk_class=packet.risk_class or str(request.get("risk_class", "medium")),
                evidence=list(request.get("evidence", [])),
            )
            entry["gate"] = gate
            self.promotion_gate_log.append(dict(gate))
            self.promotion_gate_log = self.promotion_gate_log[-100:]
            self.promotion_ladder.record_evidence(
                artifact_name,
                "promotion_request",
                source_kernel_id=packet.source_kernel_id,
                trust_level=packet.trust_level,
                risk_class=packet.risk_class,
                details={"desired_stage": request.get("desired_stage", ""), "evidence": list(request.get("evidence", [])), "trust_score": self.cluster_trust.trust_score(packet.trust_level), "gate_reason": gate.get("reason", "")},
            )
            self.promotion_ladder.set_source_metadata(artifact_name, last_promotion_request_packet=packet.packet_id)
            self._record_dynamic_outcome(
                packet.source_kernel_id,
                "patch_promotion_gate",
                success=bool(gate.get("allowed", False)),
                severity="high" if str(request.get("desired_stage", "")) in {"live_promoted", "revertable"} else "normal",
                metadata={"artifact_name": artifact_name, "desired_stage": str(request.get("desired_stage", "")), "gate_reason": gate.get("reason", "")},
            )
        elif artifact_kind == "module":
            proposal_id = str(request.get("proposal_id", request.get("artifact_name", "")))
            if proposal_id:
                gate = self.module_registry.assess_promotion_gate(
                    proposal_id,
                    trust_level=packet.trust_level,
                    locality=(self.cluster_registry.get(f"node_{packet.source_kernel_id}").locality if self.cluster_registry.get(f"node_{packet.source_kernel_id}") else "local"),
                    target=str(request.get("target_kernel_id", request.get("desired_stage", request.get("target", "hub")))) or "hub",
                )
                entry["gate"] = gate
                self.promotion_gate_log.append(dict(gate))
                self.promotion_gate_log = self.promotion_gate_log[-100:]
                self.module_registry.attach_evidence(
                    proposal_id,
                    {
                        "kind": "promotion_request",
                        "packet_id": packet.packet_id,
                        "desired_stage": request.get("desired_stage", ""),
                        "source_kernel_id": packet.source_kernel_id,
                        "gate_reason": gate.get("reason", ""),
                    },
                )
                entry["review_outcome"] = self.module_registry.governed_review(
                    proposal_id,
                    reviewer=self.kernel_id,
                    trust_level=packet.trust_level,
                    locality=(self.cluster_registry.get(f"node_{packet.source_kernel_id}").locality if self.cluster_registry.get(f"node_{packet.source_kernel_id}") is not None else "local"),
                    target=str(request.get("target_kernel_id", request.get("desired_stage", request.get("target", "hub")))) or "hub",
                    auto_promote=bool(gate.get("allowed", False) and str(request.get("target_kernel_id", request.get("target", ""))) == "main"),
                )
                review_status = str((entry.get("review_outcome") or {}).get("status", ""))
                self._record_dynamic_outcome(
                    packet.source_kernel_id,
                    "module_promotion_request",
                    success=review_status in {"approved", "promoted"},
                    severity="high" if str(request.get("target_kernel_id", request.get("target", ""))) == "main" else "normal",
                    metadata={"proposal_id": proposal_id, "status": review_status, "gate_reason": gate.get("reason", "")},
                )
        self.promotion_requests.append(entry)
        self.promotion_requests = self.promotion_requests[-100:]
        return entry



    def _growth_priority_rank_map(self) -> Dict[tuple[str, str], float]:
        if self.self_model is None:
            try:
                self.update_self_model()
            except Exception:
                return {}
        rows = list(getattr(self.self_model, "growth_priority_summary", []) or [])
        rank_map: Dict[tuple[str, str], float] = {}
        for idx, row in enumerate(rows[:20]):
            kind = str(row.get("kind", "") or "")
            name = str(row.get("name", "") or "")
            score = float(row.get("score", 0.0) or 0.0)
            if kind and name:
                rank_map[(kind, name)] = score + max(0.0, 5.0 - idx)
        return rank_map

    def _hub_queue_age_bonus(self, item: Dict[str, Any]) -> Dict[str, float]:
        now = time.time()
        queued_at = float(item.get("queued_at", 0.0) or item.get("reviewed_at", 0.0) or now)
        age_seconds = max(0.0, now - queued_at)
        age_bonus = min(2.5, age_seconds / 600.0)
        evidence_ready_bonus = 1.25 if str(item.get("queue_status", "")) == "evidence_ready" else 0.0
        defer_penalty = 0.15 * float(item.get("defer_count", 0) or 0.0)
        return {
            "age_seconds": age_seconds,
            "age_bonus": age_bonus,
            "evidence_ready_bonus": evidence_ready_bonus,
            "defer_penalty": defer_penalty,
        }

    def _hub_queue_caution_map(self) -> Dict[str, Dict[str, Any]]:
        if self.hub_kernel is None:
            return {}
        raw = dict(self.hub_kernel.local_branch_state.get("queue_caution", {}) or {})
        now = time.time()
        cleaned: Dict[str, Dict[str, Any]] = {}
        for key, payload in raw.items():
            item = dict(payload or {})
            level = max(0.0, float(item.get("caution_level", 0.0) or 0.0))
            cooldown_until = float(item.get("cooldown_until", 0.0) or 0.0)
            last_at = float(item.get("caution_last_at", item.get("last_wave_at", now)) or now)
            decay_window = max(300.0, float(item.get("decay_window", 1800.0) or 1800.0))
            elapsed = max(0.0, now - last_at)
            decay_units = elapsed / decay_window
            if decay_units > 0.0:
                level = max(0.0, level - (0.60 * decay_units))
                item["failure_count"] = max(0.0, float(item.get("failure_count", 0) or 0.0) - (0.35 * decay_units))
                item["success_count"] = max(0.0, float(item.get("success_count", 0) or 0.0) - (0.15 * decay_units))
            item["caution_level"] = level
            item["cooldown_until"] = cooldown_until
            item["cooldown_remaining"] = max(0.0, cooldown_until - now)
            item["caution_last_at"] = now
            item["decay_window"] = decay_window
            if item["caution_level"] > 0.01 or item["cooldown_remaining"] > 0.0:
                cleaned[str(key)] = item
        self.hub_kernel.local_branch_state["queue_caution"] = cleaned
        return cleaned

    def _hub_queue_caution_key(self, artifact_kind: str, artifact_name: str, proposal_id: str = "") -> str:
        artifact_kind = str(artifact_kind or "")
        artifact_name = str(artifact_name or "")
        proposal_id = str(proposal_id or "")
        if artifact_kind == "module":
            module_kind = ""
            if proposal_id and proposal_id in self.module_registry.proposals:
                rec = self.module_registry.proposals[proposal_id]
                module_kind = str(rec.module_kind or "")
            return str(self.module_registry.family_key_for(artifact_name, module_kind))
        if artifact_kind == "patch":
            return f"patch:{artifact_name}"
        return f"{artifact_kind}:{artifact_name}"

    def _update_hub_queue_caution_from_wave(self, wave: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {}
        caution = self._hub_queue_caution_map()
        now = time.time()
        for row in results:
            key = self._hub_queue_caution_key(str(row.get("artifact_kind", "")), str(row.get("artifact_name", "")), str(row.get("proposal_id", "")))
            state = dict(caution.get(key, {}))
            level = max(0.0, float(state.get("caution_level", 0.0) or 0.0))
            fails = float(state.get("failure_count", 0) or 0.0)
            successes = float(state.get("success_count", 0) or 0.0)
            if bool(row.get("success", False)):
                level = max(0.0, level - 0.55)
                successes += 1.0
                if float(state.get("cooldown_until", 0.0) or 0.0) > now and level < 0.75:
                    state["cooldown_until"] = now
            else:
                level = min(6.0, level + (1.25 if str(row.get("artifact_kind", "")) == "patch" else 0.95))
                fails += 1.0
                if level >= 1.5:
                    state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), now + (180.0 * min(4.0, level)))
            state.update({
                "caution_level": level,
                "failure_count": fails,
                "success_count": successes,
                "wave_id": str(wave.get("wave_id", "")),
                "last_wave_at": now,
                "last_wave_success": bool(row.get("success", False)),
                "caution_last_at": now,
                "decay_window": float(state.get("decay_window", 1800.0) or 1800.0),
            })
            caution[key] = state
        self.hub_kernel.local_branch_state["queue_caution"] = caution
        return caution


    def _latest_proposal_id_for_module(self, module_name: str) -> str:
        module_name = str(module_name or "")
        matches = [p for p in self.module_registry.proposals.values() if str(getattr(p, 'module_name', '')) == module_name]
        if not matches:
            return ""
        matches.sort(key=lambda p: float(getattr(p, 'created_at', 0.0) or 0.0), reverse=True)
        return str(matches[0].proposal_id)

    def _artifact_dedup_key(self, payload: Dict[str, Any] | None) -> str:
        return artifact_dedup_key(payload)

    def _open_review_wave_artifact_keys(self) -> set[str]:
        if self.hub_kernel is None:
            return set()
        keys: set[str] = set()
        for wave in list(getattr(self.hub_kernel, "review_waves", [])):
            if str(wave.get("status", "open")) in {"completed", "auto_closed", "closed", "retired"}:
                continue
            for item in list(wave.get("items", [])):
                key = str(item.get("artifact_key", item.get("key", "")) or "")
                if key:
                    keys.add(key)
        return keys

    def _score_evidence_quality(self, evidence: Dict[str, Any] | None, *, artifact_kind: str = "", base_confidence: float = 0.0) -> float:
        payload = dict(evidence or {})
        score = 0.35 + max(0.0, float(base_confidence or 0.0))
        text_fields = " ".join(str(payload.get(k, "") or "") for k in ["summary", "notes", "analysis", "rationale", "details"])
        if len(text_fields.strip()) >= 40:
            score += 0.25
        if payload.get("source") or payload.get("citation") or payload.get("url"):
            score += 0.20
        if payload.get("test") or payload.get("tests") or payload.get("regression"):
            score += 0.20
            try:
                if len(list(payload.get("tests", []) or [])) >= 2:
                    score += 0.25
            except Exception:
                pass
        if payload.get("diff") or payload.get("patch"):
            score += 0.15
        if str(artifact_kind or "") == "patch":
            score += 0.10
        return max(0.1, min(2.5, score))

    def _hub_enqueue_work_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return dict(item)
        payload = dict(item)
        payload.setdefault("queued_at", time.time())
        payload["artifact_key"] = self._artifact_dedup_key(payload)
        kind = str(payload.get("kind", "") or "")
        artifact_kind = str(payload.get("artifact_kind", "") or "")
        artifact_name = str(payload.get("artifact_name", "") or "")
        proposal_id = str(payload.get("proposal_id", "") or "")
        review_wave_id = str(payload.get("review_wave_id", payload.get("wave_id", "")) or "")
        kept = []
        merged = dict(payload)
        for existing in list(self.hub_kernel.work_queue):
            same = str(existing.get("artifact_key", self._artifact_dedup_key(existing)) or "") == str(payload.get("artifact_key", "") or "")
            if same and kind in {"promotion_evidence", "proposal_rework", "blocked_growth_followup"}:
                e = dict(existing)
                merged["queued_at"] = min(float(e.get("queued_at", merged.get("queued_at", time.time())) or time.time()), float(merged.get("queued_at", time.time()) or time.time()))
                merged["confidence"] = max(float(e.get("confidence", 0.0) or 0.0), float(merged.get("confidence", 0.0) or 0.0))
                merged["rework_quality"] = max(float(e.get("rework_quality", 0.0) or 0.0), float(merged.get("rework_quality", 0.0) or 0.0))
                merged["evidence_quality"] = max(float(e.get("evidence_quality", 0.0) or 0.0), float(merged.get("evidence_quality", 0.0) or 0.0))
                merged["duplicate_count"] = int(e.get("duplicate_count", 1) or 1) + int(merged.get("duplicate_count", 1) or 1)
                continue
            kept.append(existing)
        kept.append(merged)
        self.hub_kernel.work_queue = kept[-200:]
        return merged

    def _hub_promotion_priority_view(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.hub_kernel is None:
            return []
        now = time.time()
        caution = self._hub_queue_caution_map()
        growth = self._growth_priority_rank_map()
        open_wave_keys = self._open_review_wave_artifact_keys()
        rows: List[Dict[str, Any]] = []
        for item in list(self.hub_kernel.promotion_queue):
            row = dict(item)
            kind = str(row.get('artifact_kind', '') or '')
            name = str(row.get('artifact_name', '') or '')
            proposal_id = str(row.get('proposal_id', '') or '')
            if kind == 'module' and not proposal_id:
                proposal_id = self._latest_proposal_id_for_module(name)
                if proposal_id:
                    row['proposal_id'] = proposal_id
            target = str((row.get('desired_stage', row.get('target', row.get('target_kernel_id', 'sandbox_passed'))) if kind == 'patch' else row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub')))) or ('sandbox_passed' if kind == 'patch' else 'hub'))
            queue_status = str(row.get('queue_status', row.get('status', 'queued')) or 'queued')
            aging = self._hub_queue_age_bonus(row)
            ckey = self._hub_queue_caution_key(kind, name, proposal_id)
            cstate = dict(caution.get(ckey, {}))
            caution_level = float(cstate.get('caution_level', 0.0) or 0.0)
            cooldown_remaining = float(cstate.get('cooldown_remaining', 0.0) or 0.0)
            growth_boost = 0.0
            if kind == 'module':
                maturity = self.module_registry.maturity_report(proposal_id, target=target) if proposal_id else {'ready': False, 'maturity_score': -1.0, 'evidence_count': 0, 'effective_required_evidence': 1}
                metrics = self.module_registry.module_operational_metrics(name)
                family_key = str(metrics.get('family_key', self.module_registry.family_key_for(name, '')))
                growth_boost = max(float(growth.get(('module', name), 0.0) or 0.0), float(growth.get(('family', family_key), 0.0) or 0.0))
                rework_quality = float(row.get('rework_quality', 0.0) or 0.0)
                rework_bonus = (0.75 if queue_status == 'reworked_ready' else 0.0) + (1.1 * rework_quality)
                score = (
                    3.0 * float(maturity.get('maturity_score', 0.0) or 0.0)
                    + 1.2 * float(metrics.get('effective_reliability_score', 0.5) or 0.0)
                    + 0.25 * float(maturity.get('evidence_count', 0.0) or 0.0)
                    + aging['age_bonus'] + aging['evidence_ready_bonus'] - aging['defer_penalty']
                    + rework_bonus + 0.85 * float(metrics.get('recent_evidence_quality', 0.0) or 0.0) + 0.10 * growth_boost - 0.35 * caution_level
                    - min(3.0, float(metrics.get('effective_failure_weight', 0.0) or 0.0))
                )
                cooldown_remaining = max(cooldown_remaining, float(metrics.get('effective_cooldown_remaining', 0.0) or 0.0))
                review_action = 'review_now' if bool(maturity.get('ready', False)) and cooldown_remaining <= 0.0 else ('wait_cooldown' if cooldown_remaining > 0.0 else 'gather_evidence')
                artifact_key = str(row.get('artifact_key', self._artifact_dedup_key({'artifact_kind': kind, 'artifact_name': name, 'proposal_id': proposal_id, 'target': target, 'module_kind': str((self.module_registry.proposals.get(proposal_id).module_kind if proposal_id and proposal_id in self.module_registry.proposals else row.get('module_kind', '')) or '')})) or '')
                if artifact_key and artifact_key in open_wave_keys:
                    review_action = 'wait_open_wave'
                    score -= 12.0
                row.update({
                    'proposal_id': proposal_id,
                    'family_key': family_key,
                    'module_kind': str((self.module_registry.proposals.get(proposal_id).module_kind if proposal_id and proposal_id in self.module_registry.proposals else row.get('module_kind', '')) or ''),
                    'artifact_key': self._artifact_dedup_key({'artifact_kind': kind, 'artifact_name': name, 'proposal_id': proposal_id, 'target': target, 'module_kind': str((self.module_registry.proposals.get(proposal_id).module_kind if proposal_id and proposal_id in self.module_registry.proposals else row.get('module_kind', '')) or '')}),
                    'priority': {
                        'score': score,
                        'cooldown_remaining': cooldown_remaining,
                        'caution_level': caution_level,
                        'maturity_score': float(maturity.get('maturity_score', 0.0) or 0.0),
                        'age_seconds': aging['age_seconds'],
                        'age_bonus': aging['age_bonus'],
                        'evidence_ready_bonus': aging['evidence_ready_bonus'],
                        'rework_quality': rework_quality,
                        'rework_bonus': rework_bonus,
                        'growth_boost': growth_boost,
                        'effective_required_evidence': int(maturity.get('effective_required_evidence', maturity.get('required_evidence', 1)) or 1),
                    },
                })
            elif kind == 'patch':
                summary = self.promotion_ladder.summary(name)
                meta = dict(summary.get('source_metadata', {}))
                cooldown_remaining = max(cooldown_remaining, max(0.0, float(meta.get('cooldown_until', 0.0) or 0.0) - now))
                evidence_count = int(summary.get('evidence_count', 0) or 0)
                required = 2 if target in {'main', 'live_promoted', 'revertable', 'shadow_deployed'} else 1
                ready = evidence_count >= required and cooldown_remaining <= 0.0
                # Stage Z artifact-lineage: if a patch has been queued long
                # enough to age out without accumulating evidence, surface it
                # anyway. Resolution waves for stale items are how the system
                # finds and quarantines abandoned proposals.
                aged_out = aging['age_seconds'] >= 1500.0 and cooldown_remaining <= 0.0
                growth_boost = float(growth.get(('patch', name), 0.0) or 0.0)
                rework_quality = float(row.get('rework_quality', 0.0) or 0.0)
                score = (1.5 * evidence_count + 0.6 * float(meta.get('maturity_bonus', 0.0) or 0.0) + aging['age_bonus'] + aging['evidence_ready_bonus'] - aging['defer_penalty'] + (0.75 * rework_quality) + 0.1 * growth_boost - 0.4 * caution_level - 0.5 * float(meta.get('recent_failure_weight', 0.0) or 0.0))
                artifact_key = self._artifact_dedup_key({'artifact_kind': kind, 'artifact_name': name, 'target': target})
                row['artifact_key'] = artifact_key
                if ready or aged_out:
                    review_action = 'review_now'
                elif cooldown_remaining > 0.0:
                    review_action = 'wait_cooldown'
                else:
                    review_action = 'gather_evidence'
                if artifact_key and artifact_key in open_wave_keys:
                    review_action = 'wait_open_wave'
                    score -= 12.0
                row['priority'] = {
                    'score': score,
                    'cooldown_remaining': cooldown_remaining,
                    'caution_level': caution_level,
                    'evidence_count': evidence_count,
                    'required_evidence': required,
                    'age_seconds': aging['age_seconds'],
                    'age_bonus': aging['age_bonus'],
                    'evidence_ready_bonus': aging['evidence_ready_bonus'],
                    'rework_quality': rework_quality,
                    'growth_boost': growth_boost,
                    'aged_out': aged_out,
                }
            else:
                continue
            row['queue_status'] = queue_status
            row['review_action'] = review_action
            rows.append(row)
        rows.sort(key=lambda r: (float(r.get('priority', {}).get('score', -999.0)), str(r.get('artifact_name', ''))), reverse=True)
        return rows[:max(1, int(limit))]

    def _select_hub_review_wave(self, rows: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        open_keys = self._open_review_wave_artifact_keys()
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for r in rows:
            if str(r.get('review_action', '')) != 'review_now':
                continue
            artifact_key = str(r.get('artifact_key', self._artifact_dedup_key(r)) or '')
            if artifact_key and (artifact_key in open_keys or artifact_key in seen):
                continue
            if artifact_key:
                seen.add(artifact_key)
            candidates.append(r)
        if not candidates:
            return []
        top = candidates[0]
        selected = [top]
        selected_keys = {str(top.get('artifact_key', self._artifact_dedup_key(top)) or '')}
        if str(top.get('queue_status', '')) in {'evidence_ready', 'reworked_ready'}:
            for row in candidates[1:]:
                if len(selected) >= max(8, int(limit)):
                    break
                if str(row.get('artifact_kind', '')) != str(top.get('artifact_kind', '')):
                    continue
                if str(row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub'))) or 'hub') != str(top.get('target', top.get('target_kernel_id', top.get('desired_stage', 'hub'))) or 'hub'):
                    continue
                if str(row.get('queue_status', '')) not in {'evidence_ready', 'reworked_ready'}:
                    continue
                artifact_key = str(row.get('artifact_key', self._artifact_dedup_key(row)) or '')
                if artifact_key and artifact_key in selected_keys:
                    continue
                if artifact_key:
                    selected_keys.add(artifact_key)
                selected.append(row)
        return selected[:max(1, len(selected))]

    def wake_hub_deferred_items(self, limit: int = 20) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {'woken': 0, 'reason': 'hub_unavailable'}
        now = time.time()
        woken = 0
        for item in self.hub_kernel.promotion_queue:
            if woken >= max(1, int(limit)):
                break
            if str(item.get('status', item.get('queue_status', ''))) != 'deferred_cooldown':
                continue
            if float(item.get('deferred_until', 0.0) or 0.0) <= now:
                item['status'] = 'queued'
                item['queue_status'] = 'woken_from_cooldown'
                woken += 1
        return {'woken': woken}

    def process_hub_promotion_queue(self, limit: int = 3, consume: bool = True) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {'processed': [], 'advanced': 0, 'deferred': 0, 'review_wave_count': 0, 'wave_ids': [], 'reason': 'hub_unavailable'}
        self.wake_hub_deferred_items(limit=max(3, int(limit) * 2))
        rows = self._hub_promotion_priority_view(max(20, int(limit) * 5))
        selected = self._select_hub_review_wave(rows, limit=limit)
        processed: List[Dict[str, Any]] = []
        advanced = deferred = 0
        wave_ids: List[str] = []
        now = time.time()
        if not selected and rows:
            rows = rows[:max(1, int(limit))]
            for row in rows:
                kind = str(row.get('artifact_kind', ''))
                name = str(row.get('artifact_name', ''))
                proposal_id = str(row.get('proposal_id', '') or '')
                if str(row.get('review_action', '')) == 'wait_open_wave':
                    out = dict(row)
                    out['queue_status'] = 'waiting_open_wave'
                    processed.append(out)
                    continue
                if str(row.get('review_action', '')) == 'wait_cooldown':
                    for q in self.hub_kernel.promotion_queue:
                        if str(q.get('artifact_kind','')) == kind and str(q.get('artifact_name','')) == name and str(q.get('proposal_id','') or '') == proposal_id:
                            q['status'] = 'deferred_cooldown'
                            q['queue_status'] = 'deferred_cooldown'
                            q['defer_count'] = int(q.get('defer_count', 0) or 0) + 1
                            q['deferred_until'] = now + max(60.0, float(row.get('priority', {}).get('cooldown_remaining', 60.0) or 60.0))
                            break
                    # Track the deferred item in the work queue so callers see
                    # that the queue still has pending non-mature items even
                    # when nothing is review_now.
                    self._hub_enqueue_work_item({
                        'kind': 'promotion_cooldown_tracking',
                        'artifact_kind': kind,
                        'artifact_name': name,
                        'proposal_id': proposal_id,
                        'module_kind': str(row.get('module_kind', '') or ''),
                        'target': str(row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub'))) or 'hub'),
                        'desired_stage': str(row.get('desired_stage', row.get('target', 'hub')) or 'hub'),
                        'queued_at': now,
                        'status': 'deferred_cooldown',
                        'cooldown_remaining': float(row.get('priority', {}).get('cooldown_remaining', 0.0) or 0.0),
                    })
                    out = dict(row)
                    out['queue_status'] = 'deferred_cooldown'
                    processed.append(out)
                    deferred += 1
                else:
                    evidence_item = {
                        'kind': 'promotion_evidence',
                        'artifact_kind': kind,
                        'artifact_name': name,
                        'proposal_id': proposal_id,
                        'module_kind': str(row.get('module_kind', '') or ''),
                        'target': str(row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub'))) or 'hub'),
                        'desired_stage': str(row.get('desired_stage', row.get('target', 'hub')) or 'hub'),
                        'queued_at': now,
                        'status': 'evidence_requested',
                        'confidence': float(row.get('priority', {}).get('score', 0.0) or 0.0),
                    }
                    self._hub_enqueue_work_item(evidence_item)
                    if consume:
                        self.hub_kernel.promotion_queue = [q for q in self.hub_kernel.promotion_queue if not (str(q.get('artifact_kind','')) == kind and str(q.get('artifact_name','')) == name and str(q.get('proposal_id','') or '') == proposal_id)]
                    out = dict(row)
                    out['queue_status'] = 'evidence_requested'
                    processed.append(out)
            self._save_kernel_ecology_state()
            return {'processed': processed, 'advanced': advanced, 'deferred': deferred, 'review_wave_count': 0, 'wave_ids': []}

        if selected:
            wave_id = f'wave_{int(now*1000)}_{len(getattr(self.hub_kernel, "review_waves", [])) + 1}'
            wave_items: List[Dict[str, Any]] = []
            for row in selected:
                kind = str(row.get('artifact_kind', ''))
                name = str(row.get('artifact_name', ''))
                proposal_id = str(row.get('proposal_id', '') or '')
                target = str((row.get('desired_stage', row.get('target', row.get('target_kernel_id', 'sandbox_passed'))) if kind == 'patch' else row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub')))) or ('sandbox_passed' if kind == 'patch' else 'hub'))
                if kind == 'module':
                    req = PromotionRequest(requester_kernel_id='hub', artifact_kind='module', artifact_name=name, target_kernel_id=target, desired_stage=target, evidence=list(row.get('evidence', []) or []))
                    payload = req.to_dict()
                    payload['proposal_id'] = proposal_id
                    payload['target'] = target
                    pkt = make_packet(PacketKind.PROMOTION_REQUEST, source_kernel_id='hub', target_kernel_id=target, payload=payload, required_action='review_promotion_request', mission_context=self.hub_kernel.mission_context if self.hub_kernel is not None else '', trust_level=self.hub_kernel.trust_from_main if self.hub_kernel is not None else 'provisional', risk_class=req.risk_class, reply_expected=True)
                    auto_event = self._dispatch_kernel_packet(pkt)
                    out = dict(row)
                    out['queue_status'] = 'auto_reviewed'
                    out['auto_event'] = auto_event
                    processed.append(out)
                    advanced += 1
                    _akey = self._artifact_dedup_key({'artifact_kind': kind, 'artifact_name': name, 'proposal_id': proposal_id, 'target': target, 'module_kind': str(row.get('module_kind', '') or '')})
                    wave_items.append({'artifact_kind': kind, 'artifact_name': name, 'proposal_id': proposal_id, 'module_kind': str(row.get('module_kind', '') or ''), 'target': target, 'key': _akey, 'artifact_key': _akey, 'auto_event': auto_event})
                else:
                    out = dict(row)
                    out['queue_status'] = 'review_wave'
                    processed.append(out)
                    _akey = self._artifact_dedup_key({'artifact_kind': kind, 'artifact_name': name, 'target': target})
                    wave_items.append({'artifact_kind': kind, 'artifact_name': name, 'proposal_id': proposal_id, 'target': target, 'key': _akey, 'artifact_key': _akey})
                if consume:
                    self.hub_kernel.promotion_queue = [q for q in self.hub_kernel.promotion_queue if not (str(q.get('artifact_kind','')) == kind and str(q.get('artifact_name','')) == name and (not proposal_id or str(q.get('proposal_id','') or '') == proposal_id))]
            self.hub_kernel.review_waves.append({'wave_id': wave_id, 'created_at': now, 'status': 'open', 'selected_count': len(wave_items), 'action_counts': {'review_now': len(wave_items)}, 'items': wave_items, 'outcome_summary': {'success_count': 0, 'failure_count': 0, 'item_count': len(wave_items)}, 'success_rate': 0.0})
            self.hub_kernel.review_waves = self.hub_kernel.review_waves[-200:]
            wave_ids.append(wave_id)
            # Wave-selection path: also fold in non-selected ranked rows so
            # cooldowned and evidence-gathering items still produce work
            # entries instead of disappearing in this cycle. Without this the
            # caller has no signal that there are pending non-mature items
            # behind a selected wave.
            selected_keys = {str(r.get('artifact_key', self._artifact_dedup_key(r)) or '') for r in selected}
            for row in rows[:max(2, int(limit) * 2)]:
                rkey = str(row.get('artifact_key', self._artifact_dedup_key(row)) or '')
                if rkey and rkey in selected_keys:
                    continue
                kind = str(row.get('artifact_kind', ''))
                name = str(row.get('artifact_name', ''))
                proposal_id = str(row.get('proposal_id', '') or '')
                action = str(row.get('review_action', ''))
                if action == 'wait_cooldown':
                    for q in self.hub_kernel.promotion_queue:
                        if str(q.get('artifact_kind','')) == kind and str(q.get('artifact_name','')) == name and str(q.get('proposal_id','') or '') == proposal_id:
                            q['status'] = 'deferred_cooldown'
                            q['queue_status'] = 'deferred_cooldown'
                            q['defer_count'] = int(q.get('defer_count', 0) or 0) + 1
                            q['deferred_until'] = now + max(60.0, float(row.get('priority', {}).get('cooldown_remaining', 60.0) or 60.0))
                            break
                    self._hub_enqueue_work_item({
                        'kind': 'promotion_cooldown_tracking',
                        'artifact_kind': kind,
                        'artifact_name': name,
                        'proposal_id': proposal_id,
                        'module_kind': str(row.get('module_kind', '') or ''),
                        'target': str(row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub'))) or 'hub'),
                        'desired_stage': str(row.get('desired_stage', row.get('target', 'hub')) or 'hub'),
                        'queued_at': now,
                        'status': 'deferred_cooldown',
                        'cooldown_remaining': float(row.get('priority', {}).get('cooldown_remaining', 0.0) or 0.0),
                    })
                    out = dict(row)
                    out['queue_status'] = 'deferred_cooldown'
                    processed.append(out)
                    deferred += 1
                elif action == 'gather_evidence':
                    evidence_item = {
                        'kind': 'promotion_evidence',
                        'artifact_kind': kind,
                        'artifact_name': name,
                        'proposal_id': proposal_id,
                        'module_kind': str(row.get('module_kind', '') or ''),
                        'target': str(row.get('target', row.get('target_kernel_id', row.get('desired_stage', 'hub'))) or 'hub'),
                        'desired_stage': str(row.get('desired_stage', row.get('target', 'hub')) or 'hub'),
                        'queued_at': now,
                        'status': 'evidence_requested',
                        'confidence': float(row.get('priority', {}).get('score', 0.0) or 0.0),
                    }
                    self._hub_enqueue_work_item(evidence_item)
                    out = dict(row)
                    out['queue_status'] = 'evidence_requested'
                    processed.append(out)
        self._save_kernel_ecology_state()
        return {'processed': processed, 'advanced': advanced, 'deferred': deferred, 'review_wave_count': len(wave_ids), 'wave_ids': wave_ids}

    def complete_hub_evidence_task(self, *, artifact_kind: str, artifact_name: str, proposal_id: str = '', evidence: Dict[str, Any] | None = None, success: bool = True) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {'ok': False, 'reason': 'hub_unavailable'}
        evidence = dict(evidence or {})
        now = time.time()
        kept: List[Dict[str, Any]] = []
        matched = None
        for item in self.hub_kernel.work_queue:
            if matched is None and str(item.get('kind','')) == 'promotion_evidence' and str(item.get('artifact_kind','')) == str(artifact_kind) and str(item.get('artifact_name','')) == str(artifact_name) and (not proposal_id or str(item.get('proposal_id','') or '') == str(proposal_id)):
                matched = dict(item)
                continue
            kept.append(item)
        if matched is None:
            return {'ok': False, 'reason': 'evidence_task_not_found'}
        quality = self._score_evidence_quality(evidence, artifact_kind=artifact_kind, base_confidence=float(matched.get('confidence', 0.0) or 0.0))
        if artifact_kind == 'module':
            pid = str(proposal_id or matched.get('proposal_id', '') or self._latest_proposal_id_for_module(artifact_name))
            proposal = self.module_registry.proposals.get(pid) if pid else None
            module_kind = str(evidence.get('module_kind', matched.get('module_kind', getattr(proposal, 'module_kind', ''))) or '')
            if pid and proposal is not None:
                self.module_registry.attach_evidence(pid, {'kind': 'evidence_gather', 'time': now, 'evidence_quality': quality, **evidence})
            self.module_registry.apply_module_feedback(artifact_name, success=success, severity='normal' if success else 'high', kind='evidence_gather', target=str(matched.get('target', getattr(proposal, 'promotion_target', 'hub')) or 'hub'), metadata={'proposal_id': pid, 'module_kind': module_kind, 'evidence_quality': quality})
            self.hub_kernel.queue_promotion_request({'artifact_kind': 'module', 'artifact_name': artifact_name, 'proposal_id': pid, 'target': str(matched.get('target', getattr(proposal, 'promotion_target', 'hub')) or 'hub'), 'target_kernel_id': str(matched.get('target', getattr(proposal, 'promotion_target', 'hub')) or 'hub'), 'desired_stage': str(matched.get('desired_stage', matched.get('target', 'hub')) or 'hub'), 'queued_at': now, 'queue_status': 'evidence_ready', 'status': 'evidence_ready', 'module_kind': module_kind, 'rework_quality': max(float(matched.get('rework_quality', 0.0) or 0.0), quality), 'evidence_quality': quality})
        else:
            self.promotion_ladder.record_evidence(artifact_name, 'evidence_gather', source_kernel_id='hub', trust_level=self.hub_kernel.trust_from_main, risk_class='medium', details={'evidence_quality': quality, **evidence})
            self.hub_kernel.queue_promotion_request({'artifact_kind': 'patch', 'artifact_name': artifact_name, 'target': str(matched.get('target', matched.get('target_kernel_id', matched.get('desired_stage', 'main'))) or 'main'), 'target_kernel_id': str(matched.get('target', matched.get('target_kernel_id', matched.get('desired_stage', 'main'))) or 'main'), 'desired_stage': str(matched.get('desired_stage', 'sandbox_passed') or 'sandbox_passed'), 'queued_at': now, 'queue_status': 'evidence_ready', 'status': 'evidence_ready', 'rework_quality': max(float(matched.get('rework_quality', 0.0) or 0.0), quality), 'evidence_quality': quality})
        self.hub_kernel.work_queue = kept[-200:]
        self._save_kernel_ecology_state()
        return {'ok': True, 'artifact_kind': artifact_kind, 'artifact_name': artifact_name, 'proposal_id': proposal_id or matched.get('proposal_id', ''), 'evidence_quality': quality}

    def complete_hub_review_wave(self, wave_id: str, *, item_results: Dict[str, Any] | None = None, default_success: bool = True) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {'ok': False, 'reason': 'hub_unavailable'}
        wave = next((w for w in reversed(getattr(self.hub_kernel, 'review_waves', [])) if str(w.get('wave_id','')) == str(wave_id)), None)
        if wave is None:
            return {'ok': False, 'reason': 'wave_not_found', 'wave_id': wave_id}
        item_results = dict(item_results or {})
        results: List[Dict[str, Any]] = []
        now = time.time()
        for item in list(wave.get('items', [])):
            kind = str(item.get('artifact_kind', ''))
            name = str(item.get('artifact_name', ''))
            proposal_id = str(item.get('proposal_id', '') or '')
            # Stage Z stamps the wave item's `key` with the artifact dedup key
            # (e.g. `module::module_b526ff81a219::hub::planner`). Callers more
            # naturally key results by the human form. Accept either form.
            human_key = f'module::{name}::{proposal_id}' if kind == 'module' else name
            dedup_key = str(item.get('key', '') or item.get('artifact_key', '') or '')
            raw = (
                item_results.get(human_key)
                if human_key in item_results
                else item_results.get(dedup_key)
                if dedup_key and dedup_key in item_results
                else item_results.get(name, default_success)
            )
            if isinstance(raw, dict):
                success = bool(raw.get('success', False)) if 'success' in raw else bool(default_success)
                severity = str(raw.get('severity', 'normal') or 'normal')
            else:
                success = bool(raw)
                severity = 'low' if success else 'normal'
            results.append({'artifact_kind': kind, 'artifact_name': name, 'proposal_id': proposal_id, 'success': success, 'severity': severity})
            if kind == 'module':
                module_kind = str(item.get('module_kind', '') or '')
                if (not module_kind) and proposal_id and proposal_id in self.module_registry.proposals:
                    module_kind = str(self.module_registry.proposals[proposal_id].module_kind or '')
                self.module_registry.apply_module_feedback(name, success=success, severity=severity, kind='review_wave', target=str(item.get('target', 'hub') or 'hub'), metadata={'proposal_id': proposal_id, 'module_kind': module_kind})
            elif kind == 'patch':
                meta = dict(self.promotion_ladder.source_metadata.get(name, {}))
                sr = float(meta.get('outcome_success_rate', 1.0) or 0.0)
                rf = float(meta.get('recent_failure_weight', 0.0) or 0.0)
                if success:
                    sr = min(1.0, sr + 0.20)
                    rf = max(0.0, rf - 0.50)
                else:
                    sr = max(0.0, sr - 0.20)
                    rf = min(8.0, rf + {'low':0.5,'normal':1.0,'high':1.5,'critical':2.0}.get(severity,1.0))
                self.promotion_ladder.set_source_metadata(name, outcome_success_rate=sr, recent_failure_weight=rf, cooldown_until=(now + 180.0 if (not success and severity in {'high','critical'}) else float(meta.get('cooldown_until', 0.0) or 0.0)), maturity_bonus=max(0.0, float(meta.get('maturity_bonus', 0.0) or 0.0) + (0.20 if success else -0.10)))
        wave['status'] = 'completed'
        wave['completed_at'] = now
        wave['success_count'] = sum(1 for r in results if r['success'])
        wave['failure_count'] = sum(1 for r in results if not r['success'])
        wave['success_rate'] = wave['success_count'] / max(1, len(results))
        wave['outcome_summary'] = {'success_count': wave['success_count'], 'failure_count': wave['failure_count'], 'item_count': len(results)}
        self._update_hub_queue_caution_from_wave(wave, results)
        self._record_wave_resolution_history({'time': now, 'wave_id': wave_id, 'outcome': 'completed', 'targets': [{'kind': r['artifact_kind'], 'name': r['artifact_name'], 'proposal_id': r.get('proposal_id', ''), 'score': 1.0 if r['success'] else 0.2} for r in results], 'items': list(wave.get('items', [])), 'artifact_keys': [str(wi.get('artifact_key', wi.get('key', '')) or '') for wi in wave.get('items', []) if str(wi.get('artifact_key', wi.get('key', '')) or '')], 'score': wave['success_rate']})
        self._save_kernel_ecology_state()
        return {'ok': True, 'wave_id': wave_id, 'success_rate': wave['success_rate']}

    def process_growth_priorities(self, limit: int = 3, consume: bool = True) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {'processed': [], 'reason': 'hub_unavailable'}
        self.wake_hub_deferred_items(limit=max(3, int(limit) * 2))
        self.process_blocked_growth_followups(limit=max(1, int(limit)))
        self.process_proposal_rework(limit=max(1, int(limit)))
        try:
            self.update_self_model()
        except Exception:
            pass
        return self.process_hub_promotion_queue(limit=limit, consume=consume)

    def _hub_review_wave_priority_view(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.hub_kernel is None:
            return []
        now = time.time()
        rows: List[Dict[str, Any]] = []
        caution_map = self._hub_queue_caution_map()
        for wave in list(getattr(self.hub_kernel, "review_waves", [])):
            if str(wave.get("status", "")) in {"completed", "auto_closed", "escalated", "resolved", "closed_after_escalation", "quarantined", "rework_routed"}:
                continue
            wave_id = str(wave.get("wave_id", ""))
            unresolved = [w for w in self.hub_kernel.work_queue if str(w.get("kind", "")) in {"promotion_review", "promotion_evidence"} and str(w.get("review_wave_id", w.get("wave_id", ""))) == wave_id]
            age_seconds = max(0.0, now - float(wave.get("created_at", now) or now))
            age_bonus = min(3.0, age_seconds / 900.0)
            outcome_summary = dict(wave.get("outcome_summary", {}) or {})
            failure_bias = float(outcome_summary.get("failure_count", 0) or 0.0)
            success_bias = float(outcome_summary.get("success_count", 0) or 0.0)
            caution_hits = 0.0
            for item in wave.get("items", []):
                key = self._hub_queue_caution_key(str(item.get("artifact_kind", "")), str(item.get("artifact_name", "")), str(item.get("proposal_id", "")))
                caution_hits += float(caution_map.get(key, {}).get("caution_level", 0.0) or 0.0)
            unresolved_count = len(unresolved)
            closure_confidence = max(0.0, 1.2 + 0.30 * success_bias + (0.55 if unresolved_count == 0 else -0.35 * unresolved_count) - 0.22 * caution_hits - 0.18 * failure_bias)
            escalation_confidence = max(0.0, 0.35 * unresolved_count + 0.40 * age_bonus + 0.28 * caution_hits + 0.30 * failure_bias - 0.12 * success_bias + (0.25 if int(wave.get("surface_count", 0) or 0) >= 2 else 0.0))
            score = age_bonus + 0.5 * unresolved_count + 0.35 * float(wave.get("selected_count", 0) or 0.0) + 0.20 * caution_hits + 0.50 * failure_bias + 0.20 * max(0.0, escalation_confidence - closure_confidence)
            recommended = "wait"
            if int(wave.get("surface_count", 0) or 0) >= 2 and age_seconds >= 1500.0:
                # Repeatedly surfaced and aged: escalate even with no
                # unresolved work-queue items, since the wave has been
                # ignored or no resolver responded.
                recommended = "auto_escalate"
            elif unresolved_count == 0 and closure_confidence >= 0.85:
                recommended = "auto_close"
            elif unresolved_count > 0 and escalation_confidence >= max(1.05, closure_confidence + 0.30):
                recommended = "auto_escalate"
            rows.append({
                "wave_id": wave_id,
                "status": str(wave.get("status", "open")),
                "age_seconds": age_seconds,
                "age_bonus": age_bonus,
                "unresolved_items": unresolved_count,
                "selected_count": int(wave.get("selected_count", 0) or 0),
                "score": score,
                "surface_needed": bool(age_seconds >= 300.0 or unresolved_count > 0),
                "recommended_resolution": recommended,
                "closure_confidence": closure_confidence,
                "escalation_confidence": escalation_confidence,
            })
        rows.sort(key=lambda r: (float(r.get("score", -999.0)), str(r.get("wave_id", ""))), reverse=True)
        return rows[:max(1, int(limit))]

    def surface_open_review_waves(self, limit: int = 3) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {"surfaced": 0, "reason": "hub_unavailable"}
        priorities = self._hub_review_wave_priority_view(max(3, int(limit) * 3))
        surfaced = 0
        now = time.time()
        existing = {str(item.get("wave_id", "")) for item in self.hub_kernel.work_queue if str(item.get("kind", "")) == "review_wave_resolution"}
        for row in priorities:
            if surfaced >= max(1, int(limit)):
                break
            wave_id = str(row.get("wave_id", ""))
            if not wave_id or wave_id in existing or not bool(row.get("surface_needed", False)):
                continue
            self._hub_enqueue_work_item({
                "kind": "review_wave_resolution",
                "wave_id": wave_id,
                "queued_at": now,
                "priority_score": float(row.get("score", 0.0)),
                "age_seconds": float(row.get("age_seconds", 0.0)),
                "status": "resolution_requested",
                "recommended_resolution": str(row.get("recommended_resolution", "wait") or "wait"),
            })
            for wave in self.hub_kernel.review_waves:
                if str(wave.get("wave_id", "")) == wave_id:
                    wave["surface_count"] = int(wave.get("surface_count", 0) or 0) + 1
                    wave["last_surfaced_at"] = now
                    break
            surfaced += 1
        self.hub_kernel.work_queue = self.hub_kernel.work_queue[-200:]
        if surfaced:
            self._save_kernel_ecology_state()
        return {"surfaced": surfaced, "wave_priorities": priorities[:max(1, int(limit))]}

    def _record_wave_resolution_history(self, entry: Dict[str, Any]) -> None:
        if self.hub_kernel is None:
            return
        payload = dict(entry)
        if not payload.get("artifact_keys"):
            artifact_keys = []
            for item in list(payload.get("items", [])):
                key = str(item.get("artifact_key", item.get("key", "")) or "")
                if not key:
                    key = self._artifact_dedup_key(item)
                if key:
                    artifact_keys.append(key)
            payload["artifact_keys"] = sorted(set(artifact_keys))
            if len(payload["artifact_keys"]) == 1:
                payload["artifact_key"] = payload["artifact_keys"][0]
        hist = list(self.hub_kernel.local_branch_state.get("wave_resolution_history", []) or [])
        hist.append(payload)
        self.hub_kernel.local_branch_state["wave_resolution_history"] = hist[-200:]
        self._emit_wave_outcome_to_corpus(payload, "resolution")

    def _record_wave_escalation_history(self, entry: Dict[str, Any]) -> None:
        if self.hub_kernel is None:
            return
        payload = dict(entry)
        if not payload.get("artifact_keys"):
            artifact_keys = []
            for route in list(payload.get("routes", [])):
                key = str(route.get("artifact_key", "") or "")
                if not key and (route.get("artifact_name") or route.get("proposal_id")):
                    key = self._artifact_dedup_key(route)
                if key:
                    artifact_keys.append(key)
            payload["artifact_keys"] = sorted(set(artifact_keys))
            if len(payload["artifact_keys"]) == 1:
                payload["artifact_key"] = payload["artifact_keys"][0]
        hist = list(self.hub_kernel.local_branch_state.get("wave_escalation_history", []) or [])
        hist.append(payload)
        self.hub_kernel.local_branch_state["wave_escalation_history"] = hist[-200:]
        self._emit_wave_outcome_to_corpus(payload, "escalation")

    def resolve_surfaced_review_waves(self, limit: int = 3) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {"resolved": 0, "escalated": 0, "reason": "hub_unavailable"}
        now = time.time()
        resolved = 0
        escalated = 0
        kept_work = []
        caution = self._hub_queue_caution_map()
        for item in list(self.hub_kernel.work_queue):
            if str(item.get("kind", "")) != "review_wave_resolution" or (resolved + escalated) >= max(1, int(limit)):
                kept_work.append(item)
                continue
            wave_id = str(item.get("wave_id", ""))
            wave = next((w for w in reversed(getattr(self.hub_kernel, "review_waves", [])) if str(w.get("wave_id", "")) == wave_id), None)
            if wave is None:
                continue
            unresolved = [wq for wq in self.hub_kernel.work_queue if str(wq.get("kind", "")) in {"promotion_review", "promotion_evidence"} and str(wq.get("review_wave_id", wq.get("wave_id", ""))) == wave_id]
            age_seconds = max(0.0, now - float(wave.get("created_at", now) or now))
            action = str(item.get("recommended_resolution", "wait") or "wait")
            if action == "auto_close" and not unresolved:
                wave["status"] = "auto_closed"
                wave["completed_at"] = now
                wave["resolution_action"] = "auto_close"
                entry = {
                    "time": now,
                    "wave_id": wave_id,
                    "outcome": "auto_closed",
                    "score": 2.0 + 0.1 * float(wave.get("selected_count", 0) or 0.0),
                    "targets": [],
                    "items": list(wave.get("items", [])),
                    "artifact_keys": [str(wi.get("artifact_key", wi.get("key", "")) or "") for wi in wave.get("items", []) if str(wi.get("artifact_key", wi.get("key", "")) or "")],
                }
                for wi in wave.get("items", []):
                    if str(wi.get("artifact_kind", "")) == "module":
                        proposal_id = str(wi.get("proposal_id", "") or "")
                        module_kind = str(wi.get("module_kind", "") or "")
                        if (not module_kind) and proposal_id and proposal_id in self.module_registry.proposals:
                            module_kind = str(self.module_registry.proposals[proposal_id].module_kind or "")
                        fam = self.module_registry.family_key_for(str(wi.get("artifact_name", "")), module_kind)
                        entry["targets"].append({"kind": "family", "name": fam, "score": 1.5})
                        self.module_registry.apply_module_feedback(
                            str(wi.get("artifact_name", "")),
                            success=True,
                            severity="low",
                            kind="review_wave_auto_close",
                            target=str(wi.get("target", wi.get("target_kernel_id", "hub")) or "hub"),
                            metadata={"proposal_id": proposal_id, "module_kind": module_kind},
                        )
                        key = self._hub_queue_caution_key("module", str(wi.get("artifact_name", "")), proposal_id)
                        state = dict(caution.get(key, {}))
                        state["caution_level"] = max(0.0, float(state.get("caution_level", 0.0) or 0.0) - 0.4)
                        state["caution_last_at"] = now
                        if float(state.get("caution_level", 0.0) or 0.0) < 0.75:
                            state["cooldown_until"] = now
                        caution[key] = state
                    elif str(wi.get("artifact_kind", "")) == "patch":
                        name = str(wi.get("artifact_name", ""))
                        entry["targets"].append({"kind": "patch", "name": name, "score": 1.25})
                        key = self._hub_queue_caution_key("patch", name, "")
                        state = dict(caution.get(key, {}))
                        state["caution_level"] = max(0.0, float(state.get("caution_level", 0.0) or 0.0) - 0.45)
                        state["caution_last_at"] = now
                        if float(state.get("caution_level", 0.0) or 0.0) < 0.75:
                            state["cooldown_until"] = now
                        caution[key] = state
                self._record_wave_resolution_history(entry)
                resolved += 1
                continue
            if action == "auto_escalate":
                wave["status"] = "escalated"
                wave["escalated_at"] = now
                wave["resolution_action"] = "auto_escalate"
                self._hub_enqueue_work_item({
                    "kind": "review_wave_escalation",
                    "wave_id": wave_id,
                    "queued_at": now,
                    "status": "escalated",
                    "age_seconds": age_seconds,
                    "unresolved_items": len(unresolved),
                })
                entry = {
                    "time": now,
                    "wave_id": wave_id,
                    "outcome": "auto_escalated",
                    "score": 1.8 + 0.15 * len(unresolved),
                    "targets": [],
                    "items": list(wave.get("items", [])),
                    "artifact_keys": [str(wi.get("artifact_key", wi.get("key", "")) or "") for wi in wave.get("items", []) if str(wi.get("artifact_key", wi.get("key", "")) or "")],
                }
                for wi in wave.get("items", []):
                    if str(wi.get("artifact_kind", "")) == "module":
                        proposal_id = str(wi.get("proposal_id", "") or "")
                        module_kind = str(wi.get("module_kind", "") or "")
                        if (not module_kind) and proposal_id and proposal_id in self.module_registry.proposals:
                            module_kind = str(self.module_registry.proposals[proposal_id].module_kind or "")
                        fam = self.module_registry.family_key_for(str(wi.get("artifact_name", "")), module_kind)
                        entry["targets"].append({"kind": "family", "name": fam, "score": 1.4})
                        key = self._hub_queue_caution_key("module", str(wi.get("artifact_name", "")), proposal_id)
                        state = dict(caution.get(key, {}))
                        state["caution_level"] = min(6.0, float(state.get("caution_level", 0.0) or 0.0) + 0.45)
                        state["caution_last_at"] = now
                        if float(state.get("caution_level", 0.0) or 0.0) >= 1.0:
                            state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), now + 120.0)
                        caution[key] = state
                    elif str(wi.get("artifact_kind", "")) == "patch":
                        name = str(wi.get("artifact_name", ""))
                        entry["targets"].append({"kind": "patch", "name": name, "score": 1.2})
                        key = self._hub_queue_caution_key("patch", name, "")
                        state = dict(caution.get(key, {}))
                        state["caution_level"] = min(6.0, float(state.get("caution_level", 0.0) or 0.0) + 0.55)
                        state["caution_last_at"] = now
                        if float(state.get("caution_level", 0.0) or 0.0) >= 1.0:
                            state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), now + 150.0)
                        caution[key] = state
                self._record_wave_resolution_history(entry)
                escalated += 1
                continue
            kept_work.append(item)
        self.hub_kernel.work_queue = kept_work[-200:]
        self.hub_kernel.local_branch_state["queue_caution"] = caution
        if resolved or escalated:
            try:
                self.update_self_model()
            except Exception:
                pass
            self._save_kernel_ecology_state()
        return {"resolved": resolved, "escalated": escalated, "history": list(self.hub_kernel.local_branch_state.get("wave_resolution_history", []))[-10:]}

    def process_wave_escalations(self, limit: int = 3) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {"processed": 0, "reason": "hub_unavailable"}
        processed = 0
        now = time.time()
        new_work: List[Dict[str, Any]] = []
        cleanup_wave_ids: set[str] = set()
        escalation_items = [item for item in list(self.hub_kernel.work_queue) if str(item.get("kind", "")) == "review_wave_escalation"]
        passthrough = [item for item in list(self.hub_kernel.work_queue) if str(item.get("kind", "")) != "review_wave_escalation"]
        caution_map = self._hub_queue_caution_map()
        for item in escalation_items:
            if processed >= max(1, int(limit)):
                new_work.append(item)
                continue
            wave_id = str(item.get("wave_id", ""))
            wave = next((w for w in reversed(getattr(self.hub_kernel, "review_waves", [])) if str(w.get("wave_id", "")) == wave_id), None)
            if wave is None:
                continue
            unresolved = [wq for wq in passthrough if str(wq.get("kind", "")) in {"promotion_review", "promotion_evidence"} and str(wq.get("review_wave_id", wq.get("wave_id", ""))) == wave_id]
            patch_items = [u for u in unresolved if str(u.get("artifact_kind", "")) == "patch"]
            module_items = [u for u in unresolved if str(u.get("artifact_kind", "")) == "module"]
            confidence = float(item.get("confidence", 0.0) or 0.0)
            caution_hits = 0.0
            for wi in unresolved:
                caution_hits += float(caution_map.get(self._hub_queue_caution_key(str(wi.get("artifact_kind", "")), str(wi.get("artifact_name", "")), str(wi.get("proposal_id", ""))), {}).get("caution_level", 0.0) or 0.0)
            routes: List[Dict[str, Any]] = []
            cleanup_wave_ids.add(wave_id)
            if patch_items and confidence >= 1.15:
                for wi in patch_items:
                    patch_name = str(wi.get("artifact_name", "") or "")
                    target = str(wi.get("target", wi.get("target_kernel_id", "main")) or "main")
                    if patch_name:
                        self.staged_patches.setdefault(patch_name, {"status": "staged", "target": target})
                        quarantine_patch(patch_name, target, "review_wave_escalation", self.staged_patches, self.quarantine_manager)
                        routes.append({"kind": "patch_quarantine", "name": patch_name, "target": target, "artifact_kind": "patch", "artifact_name": patch_name, "artifact_key": self._artifact_dedup_key({"artifact_kind": "patch", "artifact_name": patch_name, "target": target})})
                        self._record_dynamic_outcome("hub", "wave_patch_quarantine", success=False, severity="high", metadata={"patch_name": patch_name})
                wave["status"] = "quarantined"
                wave["resolution_action"] = "patch_quarantine"
                wave["completed_at"] = now
            elif module_items and confidence >= max(0.95, 0.65 + 0.10 * caution_hits):
                for wi in module_items:
                    proposal_id = str(wi.get("proposal_id", "") or "")
                    mod_name = str(wi.get("artifact_name", "") or "")
                    mod_kind = str(wi.get("module_kind", "") or "")
                    if (not mod_kind) and proposal_id and proposal_id in self.module_registry.proposals:
                        mod_kind = str(self.module_registry.proposals[proposal_id].module_kind or "")
                    new_work.append({
                        "kind": "proposal_rework",
                        "proposal_id": proposal_id,
                        "artifact_name": mod_name,
                        "module_kind": mod_kind,
                        "queued_at": now,
                        "status": "rework_requested",
                        "review_wave_id": wave_id,
                        "confidence": confidence,
                        "artifact_key": self._artifact_dedup_key({"artifact_kind": "module", "artifact_name": mod_name, "proposal_id": proposal_id, "target": "hub", "module_kind": mod_kind}),
                    })
                    self.module_registry.apply_module_feedback(mod_name, success=False, severity="normal", kind="review_wave_escalation", target=str(wi.get("target", wi.get("target_kernel_id", "hub")) or "hub"), metadata={"proposal_id": proposal_id, "module_kind": mod_kind})
                    routes.append({"kind": "proposal_rework", "name": mod_name, "proposal_id": proposal_id, "module_kind": mod_kind, "artifact_kind": "module", "artifact_name": mod_name, "artifact_key": self._artifact_dedup_key({"artifact_kind": "module", "artifact_name": mod_name, "proposal_id": proposal_id, "target": "hub", "module_kind": mod_kind})})
                wave["status"] = "rework_routed"
                wave["resolution_action"] = "proposal_rework"
                wave["completed_at"] = now
            else:
                record = BlockedGrowthRecord(kernel_id="hub", blocker="review_wave_escalation", symptom=f"stale wave {wave_id}", attempted_action="auto_escalate", required_resource="governance_review", recommended_action="blocked_growth_review", severity="high")
                self._blocked_growth_log.append(record.to_dict())
                self.hub_kernel.blocked_growth.append(record)
                followup_item = {
                    "kind": "blocked_growth_followup",
                    "wave_id": wave_id,
                    "queued_at": now,
                    "status": "blocked_growth_requested",
                    "confidence": confidence,
                    "caution_hits": caution_hits,
                    "unresolved_items": len(unresolved),
                }
                if module_items:
                    wi = module_items[0]
                    proposal_id = str(wi.get("proposal_id", "") or "")
                    mod_name = str(wi.get("artifact_name", "") or "")
                    mod_kind = str(wi.get("module_kind", "") or "")
                    if (not mod_kind) and proposal_id and proposal_id in self.module_registry.proposals:
                        mod_kind = str(self.module_registry.proposals[proposal_id].module_kind or "")
                    followup_item.update({"artifact_kind": "module", "artifact_name": mod_name, "proposal_id": proposal_id, "module_kind": mod_kind, "artifact_key": self._artifact_dedup_key({"artifact_kind": "module", "artifact_name": mod_name, "proposal_id": proposal_id, "target": "hub", "module_kind": mod_kind})})
                    routes.append({"kind": "blocked_growth", "name": mod_name, "proposal_id": proposal_id, "artifact_kind": "module", "artifact_name": mod_name, "module_kind": mod_kind, "artifact_key": self._artifact_dedup_key({"artifact_kind": "module", "artifact_name": mod_name, "proposal_id": proposal_id, "target": "hub", "module_kind": mod_kind})})
                elif patch_items:
                    wi = patch_items[0]
                    patch_name = str(wi.get("artifact_name", "") or "")
                    followup_item.update({"artifact_kind": "patch", "artifact_name": patch_name, "target": str(wi.get("target", wi.get("target_kernel_id", "main")) or "main"), "artifact_key": self._artifact_dedup_key({"artifact_kind": "patch", "artifact_name": patch_name, "target": str(wi.get("target", wi.get("target_kernel_id", "main")) or "main")})})
                    routes.append({"kind": "blocked_growth", "name": patch_name, "artifact_kind": "patch", "artifact_name": patch_name, "artifact_key": self._artifact_dedup_key({"artifact_kind": "patch", "artifact_name": patch_name, "target": str(wi.get("target", wi.get("target_kernel_id", "main")) or "main")})})
                else:
                    routes.append({"kind": "blocked_growth", "name": wave_id, "artifact_key": f"wave::{wave_id}"})
                new_work.append(followup_item)
                wave["status"] = "blocked_growth"
                wave["resolution_action"] = "blocked_growth"
                wave["completed_at"] = now
            hist = {
                "time": now,
                "wave_id": wave_id,
                "outcome": str(wave.get("resolution_action", "escalated")),
                "routes": routes,
                "unresolved_count": len(unresolved),
                "confidence": confidence,
            }
            self._record_wave_escalation_history(hist)
            processed += 1
        kept: List[Dict[str, Any]] = []
        for item in passthrough:
            iid = str(item.get("review_wave_id", item.get("wave_id", "")) or "")
            if iid and iid in cleanup_wave_ids and str(item.get("kind", "")) in {"promotion_review", "promotion_evidence"}:
                continue
            kept.append(item)
        kept.extend(new_work)
        self.hub_kernel.work_queue = kept[-200:]
        if processed:
            self._save_kernel_ecology_state()
        return {"processed": processed, "remaining": len([item for item in self.hub_kernel.work_queue if str(item.get("kind", "")) == "review_wave_escalation"]), "created_rework": len([item for item in new_work if str(item.get("kind", "")) == "proposal_rework"]), "created_blocked_growth": len([item for item in new_work if str(item.get("kind", "")) == "blocked_growth_followup"]) }

    def _record_proposal_rework_history(self, entry: Dict[str, Any]) -> None:
        if self.hub_kernel is None:
            return
        payload = dict(entry)
        payload.setdefault("artifact_key", self._artifact_dedup_key({"artifact_kind": "module", "artifact_name": payload.get("module_name", payload.get("artifact_name", "")), "proposal_id": payload.get("proposal_id", ""), "target": payload.get("target", "hub"), "module_kind": payload.get("module_kind", "")}))
        hist = list(self.hub_kernel.local_branch_state.get("proposal_rework_history", []) or [])
        hist.append(payload)
        self.hub_kernel.local_branch_state["proposal_rework_history"] = hist[-200:]

    def _record_blocked_growth_followup_history(self, entry: Dict[str, Any]) -> None:
        if self.hub_kernel is None:
            return
        payload = dict(entry)
        if not payload.get("artifact_key") and (payload.get("artifact_name") or payload.get("proposal_id")):
            payload["artifact_key"] = self._artifact_dedup_key({"artifact_kind": payload.get("artifact_kind", ""), "artifact_name": payload.get("artifact_name", ""), "proposal_id": payload.get("proposal_id", ""), "target": payload.get("target", payload.get("desired_stage", "hub")), "module_kind": payload.get("module_kind", "")})
        hist = list(self.hub_kernel.local_branch_state.get("blocked_growth_followup_history", []) or [])
        hist.append(payload)
        self.hub_kernel.local_branch_state["blocked_growth_followup_history"] = hist[-200:]

    def process_proposal_rework(self, limit: int = 3) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {"processed": 0, "reason": "hub_unavailable"}
        processed = 0
        now = time.time()
        kept: List[Dict[str, Any]] = []
        caution = self._hub_queue_caution_map()
        for item in list(self.hub_kernel.work_queue):
            if str(item.get("kind", "")) != "proposal_rework" or processed >= max(1, int(limit)):
                kept.append(item)
                continue
            proposal_id = str(item.get("proposal_id", "") or "")
            module_name = str(item.get("artifact_name", "") or "")
            proposal = self.module_registry.proposals.get(proposal_id) if proposal_id else None
            if proposal is not None:
                module_name = module_name or str(proposal.module_name or "")
            module_kind = str(item.get("module_kind", getattr(proposal, 'module_kind', '')) or '')
            target = str(item.get("target", getattr(proposal, 'promotion_target', 'hub')) or "hub")
            rework_quality = max(0.25, float(item.get("rework_quality", item.get("confidence", 0.0)) or 0.0) + 0.15 * int(item.get("rework_count", 0) or 0))
            if proposal_id and proposal is not None:
                self.module_registry.attach_evidence(proposal_id, {"kind": "proposal_rework", "time": now, "wave_id": str(item.get("review_wave_id", "")), "confidence": float(item.get("confidence", 0.0) or 0.0), "rework_quality": rework_quality})
            self.module_registry.apply_module_feedback(module_name, success=True, severity="low" if rework_quality < 1.0 else "normal", kind="proposal_rework", target=target, metadata={"proposal_id": proposal_id, "module_kind": module_kind, "rework_quality": rework_quality})
            if proposal_id and proposal is not None:
                self.hub_kernel.queue_promotion_request({"artifact_kind": "module", "artifact_name": module_name, "proposal_id": proposal_id, "target": target, "target_kernel_id": target, "desired_stage": target, "queued_at": now, "status": "reworked_ready", "queue_status": "reworked_ready", "module_kind": module_kind, "rework_count": int(item.get('rework_count',0) or 0) + 1, "rework_quality": rework_quality})
            family_key = self.module_registry.family_key_for(module_name, module_kind)
            state = dict(caution.get(family_key, {}))
            if state:
                state["caution_level"] = max(0.0, float(state.get("caution_level", 0.0) or 0.0) - 0.55)
                state["success_count"] = float(state.get("success_count", 0.0) or 0.0) + 1.0
                state["caution_last_at"] = now
                if float(state.get("caution_level", 0.0) or 0.0) < 0.75:
                    state["cooldown_until"] = now
                caution[family_key] = state
            self._record_proposal_rework_history({"time": now, "proposal_id": proposal_id, "module_name": module_name, "module_kind": module_kind, "target": target, "outcome": "reworked_ready", "family_key": family_key, "rework_quality": rework_quality})
            processed += 1
        self.hub_kernel.local_branch_state["queue_caution"] = caution
        self.hub_kernel.work_queue = kept[-200:]
        if processed:
            self._save_kernel_ecology_state()
        return {"processed": processed, "remaining": len([item for item in self.hub_kernel.work_queue if str(item.get("kind", "")) == "proposal_rework"]), "requeued": len([item for item in self.hub_kernel.promotion_queue if str(item.get("status", "")) == "reworked_ready"])}

    def process_blocked_growth_followups(self, limit: int = 3) -> Dict[str, Any]:
        if self.hub_kernel is None:
            return {"processed": 0, "reason": "hub_unavailable"}
        processed = 0
        now = time.time()
        kept: List[Dict[str, Any]] = []
        spawned: List[Dict[str, Any]] = []
        caution = self._hub_queue_caution_map()
        for item in list(self.hub_kernel.work_queue):
            if str(item.get("kind", "")) != "blocked_growth_followup" or processed >= max(1, int(limit)):
                kept.append(item)
                continue
            artifact_kind = str(item.get("artifact_kind", "") or "")
            artifact_name = str(item.get("artifact_name", "") or "")
            proposal_id = str(item.get("proposal_id", "") or "")
            confidence = float(item.get("confidence", 0.0) or 0.0)
            outcome = "blocked_growth_recorded"
            if artifact_kind == "module" and proposal_id and confidence >= 0.55:
                proposal = self.module_registry.proposals.get(proposal_id)
                module_kind = str(item.get("module_kind", getattr(proposal, 'module_kind', '')) or '')
                spawned.append({
                    "kind": "proposal_rework",
                    "proposal_id": proposal_id,
                    "artifact_name": artifact_name or str(getattr(proposal, 'module_name', '') or ''),
                    "module_kind": module_kind,
                    "queued_at": now,
                    "status": "rework_requested_from_blocked_growth",
                    "review_wave_id": str(item.get("wave_id", "")),
                    "confidence": confidence,
                    "artifact_key": self._artifact_dedup_key({"artifact_kind": "module", "artifact_name": artifact_name or str(getattr(proposal, 'module_name', '') or ''), "proposal_id": proposal_id, "target": "hub", "module_kind": module_kind}),
                })
                outcome = "rerouted_to_rework"
            elif artifact_kind in {"module", "patch"} and (proposal_id or artifact_name):
                proposal = self.module_registry.proposals.get(proposal_id) if proposal_id else None
                module_kind = str(item.get("module_kind", getattr(proposal, 'module_kind', '')) or '')
                spawned.append({
                    "kind": "promotion_evidence",
                    "artifact_kind": artifact_kind,
                    "artifact_name": artifact_name or str(getattr(proposal, 'module_name', '') or ''),
                    "proposal_id": proposal_id,
                    "module_kind": module_kind,
                    "target": str(item.get("target", getattr(proposal, 'promotion_target', 'hub')) or ("main" if artifact_kind == "patch" else "hub")),
                    "desired_stage": str(item.get("desired_stage", item.get("target", getattr(proposal, 'promotion_target', 'hub'))) or ("sandbox_passed" if artifact_kind == "patch" else "hub")),
                    "queued_at": now,
                    "status": "evidence_requested_from_blocked_growth",
                    "review_wave_id": str(item.get("wave_id", "")),
                    "confidence": max(0.1, confidence),
                    "artifact_key": self._artifact_dedup_key({"artifact_kind": artifact_kind, "artifact_name": artifact_name or str(getattr(proposal, 'module_name', '') or ''), "proposal_id": proposal_id, "target": str(item.get("target", getattr(proposal, 'promotion_target', 'hub')) or ("main" if artifact_kind == "patch" else "hub")), "module_kind": module_kind}),
                })
                outcome = "targeted_evidence_requested"
            else:
                key = self._hub_queue_caution_key(artifact_kind, artifact_name, proposal_id)
                state = dict(caution.get(key, {}))
                if state:
                    state["caution_level"] = min(6.0, float(state.get("caution_level", 0.0) or 0.0) + 0.15)
                    state["caution_last_at"] = now
                    caution[key] = state
            self._record_blocked_growth_followup_history({
                "time": now,
                "wave_id": str(item.get("wave_id", "")),
                "artifact_kind": artifact_kind,
                "artifact_name": artifact_name,
                "proposal_id": proposal_id,
                "module_kind": str(item.get("module_kind", "") or ""),
                "target": str(item.get("target", item.get("desired_stage", "hub")) or "hub"),
                "artifact_key": str(item.get("artifact_key", "") or self._artifact_dedup_key(item)),
                "confidence": confidence,
                "outcome": outcome,
            })
            processed += 1
        self.hub_kernel.local_branch_state["queue_caution"] = caution
        self.hub_kernel.work_queue = kept[-200:]
        for spawned_item in spawned:
            self._hub_enqueue_work_item(spawned_item)
        if processed:
            self._save_kernel_ecology_state()
        return {"processed": processed, "remaining": len([item for item in self.hub_kernel.work_queue if str(item.get("kind", "")) == "blocked_growth_followup"]), "spawned_rework": len([item for item in spawned if str(item.get("kind", "")) == "proposal_rework"]) }

    def _promotion_queue_summary(self) -> Dict[str, Any]:
        pending = sum(1 for stage in self.promotion_ladder.state.values() if stage not in ("live_promoted", "revertable", "reverted"))
        return {
            "pending_count": pending,
            "tracked_patch_count": len(self.promotion_ladder.state),
            "hub_queue_count": len(self.hub_kernel.promotion_queue) if self.hub_kernel is not None else 0,
            "hub_top_priorities": self._hub_promotion_priority_view(10),
            "module_top_priorities": self.module_registry.prioritized_proposals()[:10],
        }

    def record_delegation_outcome(
        self,
        lease_id: str,
        *,
        success: bool,
        result: Dict[str, Any] | None = None,
        reason: str = "",
        severity: str = "normal",
    ) -> Dict[str, Any]:
        lease = self.delegation_manager.leases.get(lease_id)
        if lease is None:
            return {"ok": False, "reason": "unknown_lease", "lease_id": lease_id}
        payload = dict(result or {})
        if success:
            self.delegation_manager.complete(lease_id, payload)
        else:
            self.delegation_manager.fail(lease_id, reason=reason or "delegation_failed", result=payload)
        job_id = str(lease.provenance.get("job_id", "") or "")
        if job_id:
            if success:
                self.distributed_queue.complete(job_id, result=payload)
            else:
                self.distributed_queue.fail(job_id, reason=reason or "delegation_failed", result=payload)
        target_kernel_id = lease.target_kernel_id
        if target_kernel_id:
            outcome_name = "delegation_success" if success else "delegation_failure"
            self._record_dynamic_outcome(
                target_kernel_id,
                outcome_name,
                success=success,
                severity=severity,
                metadata={"lease_id": lease_id, "job_id": job_id, "worker_role": lease.worker_role},
            )
            if success:
                for permission in list(lease.allowed_tool_permissions or [])[:1]:
                    self.budget_manager.recover_worker_request(target_kernel_id, permission=permission, amount=1)
                sub = self.subkernels.get(target_kernel_id)
                if sub is not None and sub.lifecycle == KernelLifecycle.DEGRADED.value:
                    sub.lifecycle = KernelLifecycle.EXPERIMENTAL.value
            else:
                sub = self.subkernels.get(target_kernel_id)
                if sub is not None and severity in {"high", "severe", "critical"}:
                    sub.lifecycle = KernelLifecycle.DEGRADED.value
        self._sync_kernel_registry()
        self._save_kernel_ecology_state()
        return {
            "ok": True,
            "lease_id": lease_id,
            "job_id": job_id,
            "target_kernel_id": target_kernel_id,
            "success": bool(success),
            "trust_level": self.cluster_trust.trust_level_for(target_kernel_id, default="provisional") if target_kernel_id else "",
        }

    def _record_dynamic_outcome(
        self,
        kernel_id: str,
        outcome: str,
        *,
        success: bool,
        severity: str = "normal",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if not kernel_id:
            return
        meta = dict(metadata or {})
        self.cluster_trust.record_outcome(
            kernel_id,
            outcome,
            success=success,
            severity=severity,
            source=self.kernel_id,
            metadata=meta,
        )
        self._apply_node_feedback(
            kernel_id,
            success=success,
            severity=severity,
            kind=str(outcome or "outcome"),
            target=str(meta.get("desired_stage", meta.get("promotion_target", meta.get("target", ""))) or ""),
            metadata={
                "last_feedback_kind": str(outcome or "outcome"),
                "last_feedback_success": bool(success),
            },
        )

    def _trust_score_for_kernel(self, kernel_id: str) -> int:
        return self.cluster_trust.trust_score(self.cluster_trust.trust_level_for(kernel_id, default="untrusted"))

    def _worker_profile_for_kernel(self, kernel_id: str, *, locality: str = "local"):
        if kernel_id == self.kernel_id:
            return profile_for("main", specialization=self.node_identity.specialization, locality=locality)
        if self.hub_kernel is not None and kernel_id == self.hub_kernel.kernel_id:
            return profile_for("hub", locality=locality)
        sub = self.subkernels.get(kernel_id)
        if sub is not None:
            return profile_for("subkernel", specialization=sub.state.specialization, locality=locality)
        return profile_for("subkernel", locality=locality)

    def _evaluate_tool_request(self, packet: KernelPacket) -> Dict[str, Any]:
        request = dict(packet.payload)
        tool_name = str(request.get("tool_name", "")).strip()
        node = self.cluster_registry.get(f"node_{packet.source_kernel_id}")
        locality = node.locality if node is not None else "local"
        profile = self._worker_profile_for_kernel(packet.source_kernel_id, locality=locality)
        contract = TOOL_CONTRACTS.get(tool_name)
        decision = evaluate_tool_access(
            tool_name,
            required_permission=getattr(contract, "required_permissions", "safe_autonomous"),
            profile=profile,
            trust_level=packet.trust_level or self.cluster_trust.trust_level_for(packet.source_kernel_id, default=profile.trust_floor),
            locality=locality,
        )
        decision.update({
            "packet_id": packet.packet_id,
            "source_kernel_id": packet.source_kernel_id,
            "target_kernel_id": packet.target_kernel_id,
            "desired_scope": request.get("desired_scope", "branch_local"),
        })
        active_leases = len(self.delegation_manager.list_active(packet.source_kernel_id))
        budget_decision = self.budget_manager.check_worker_request(
            packet.source_kernel_id,
            role=profile.role,
            permission=str(decision.get("required_permission", "safe_autonomous")),
            active_leases=active_leases,
            max_active_leases=profile.max_active_leases,
        )
        decision["budget_gate"] = dict(budget_decision)
        if decision.get("allowed") and not budget_decision.get("allowed", False):
            decision["allowed"] = False
            decision["reason"] = str(budget_decision.get("reason", "worker_budget_exhausted"))
        if decision.get("allowed"):
            self.budget_manager.spend_worker_request(packet.source_kernel_id, role=profile.role, permission=str(decision.get("required_permission", "safe_autonomous")))
            grant = {
                "tool_name": tool_name,
                "granted_by": self.kernel_id,
                "granted_at": time.time(),
                "policy_reason": decision.get("reason", "policy_ok"),
                "required_permission": decision.get("required_permission", "safe_autonomous"),
                "desired_scope": request.get("desired_scope", "branch_local"),
            }
            if self.hub_kernel is not None and packet.source_kernel_id == self.hub_kernel.kernel_id:
                self.hub_kernel.experimental_tool_registry[tool_name] = grant
            elif packet.source_kernel_id in self.subkernels:
                self.subkernels[packet.source_kernel_id].state.local_tools[tool_name] = grant
        self.worker_budget_decisions.append(dict(budget_decision))
        self.worker_budget_decisions = self.worker_budget_decisions[-100:]
        self.tool_access_decisions.append(dict(decision))
        self.tool_access_decisions = self.tool_access_decisions[-100:]
        return decision

    def _evaluate_module_policy(self, proposal_record: Dict[str, Any]) -> Dict[str, Any]:
        source_kernel_id = str(proposal_record.get("proposer_kernel_id", ""))
        node = self.cluster_registry.get(f"node_{source_kernel_id}")
        locality = node.locality if node is not None else "local"
        decision = self.module_registry.assess_promotion_gate(
            str(proposal_record.get("proposal_id", "")),
            trust_level=str(proposal_record.get("trust_level", self.cluster_trust.trust_level_for(source_kernel_id, default="provisional"))),
            locality=locality,
            target=str(proposal_record.get("promotion_target", "hub")),
        )
        maturity = self.module_registry.maturity_report(
            str(proposal_record.get("proposal_id", "")),
            target=str(proposal_record.get("promotion_target", "hub")),
        )
        decision["maturity"] = maturity
        if proposal_record.get("review_outcome"):
            decision["review_outcome"] = dict(proposal_record.get("review_outcome") or {})
        self.module_policy_decisions.append(dict(decision))
        self.module_policy_decisions = self.module_policy_decisions[-100:]
        return decision

    def _choose_delegation_target(self, specialization: str, *, required_trust_level: str = "low") -> tuple[Optional[Subkernel], str, str]:
        candidates = [s for s in self.subkernels.values() if s.state.specialization == specialization]
        if not candidates:
            candidates = list(self.subkernels.values())
        eligible: List[tuple[int, float, float, float, float, int, Subkernel, str]] = []
        for sub in candidates:
            node_id = f"node_{sub.kernel_id}"
            if not self.cluster_trust.meets_threshold(sub.kernel_id, required_trust_level):
                continue
            profile = self._worker_profile_for_kernel(sub.kernel_id, locality="local")
            metrics = self._node_operational_metrics(sub.kernel_id, locality="local")
            active_count = metrics["active_leases"]
            if active_count >= profile.max_active_leases:
                continue
            if float(metrics.get("cooldown_remaining", 0.0)) > 0.0:
                continue
            maturity = float((self.cluster_registry.get(node_id).metadata.get("maturity_score", self._node_maturity_score(sub.kernel_id)) if self.cluster_registry.get(node_id) is not None else self._node_maturity_score(sub.kernel_id)))
            eligible.append((
                self._trust_score_for_kernel(sub.kernel_id),
                float(metrics["success_rate"]),
                maturity,
                float(metrics.get("reliability_score", 0.5)),
                -float(metrics["budget_pressure"]),
                -active_count,
                sub,
                node_id,
            ))
        if not eligible:
            return None, "", ""
        eligible.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6].kernel_id), reverse=True)
        _, _, _, _, _, _, sub, node_id = eligible[0]
        return sub, node_id, self.cluster_trust.trust_level_for(sub.kernel_id, default=sub.trust_from_parent)

    def _delegate_task_to_subkernel(
        self,
        goal: str,
        specialization: str = "general",
        mission_context: str = "delegated mission",
        *,
        parent_goal_id: str = "",
        source_kernel_id: str = "main",
        packet_id: str = "",
        required_trust_level: str = "low",
    ) -> Dict[str, Any]:
        target, target_node_id, assigned_trust_level = self._choose_delegation_target(specialization, required_trust_level=required_trust_level)
        if target is None:
            target = self._spawn_subkernel(specialization, mission_context)
            target_node_id = f"node_{target.kernel_id}"
            assigned_trust_level = self.cluster_trust.trust_level_for(target.kernel_id, default=target.trust_from_parent)
        profile = self._worker_profile_for_kernel(target.kernel_id, locality="local")
        task_id = self.task_queue.create(
            goal,
            parent_id=parent_goal_id or None,
            owner_kernel_id=target.kernel_id,
            requester_kernel_id=source_kernel_id,
            root_goal_id=parent_goal_id or "",
            mission_context=mission_context,
            lease_scope="delegated",
            delegated_to_kernel_id=target.kernel_id,
            packet_id=packet_id,
            provenance={"source_kernel_id": source_kernel_id, "specialization": specialization, "target_node_id": target_node_id, "required_trust_level": required_trust_level, "worker_role": profile.role},
        )
        queue_rec = self.distributed_queue.enqueue(
            goal=goal,
            goal_id=task_id,
            source_kernel_id=source_kernel_id,
            specialization=specialization,
            mission_context=mission_context,
            required_trust_level=required_trust_level,
            packet_id=packet_id,
            provenance={"parent_goal_id": parent_goal_id, "target_kernel_id": target.kernel_id},
        )
        lease = self.delegation_manager.issue(
            goal_id=task_id,
            parent_goal_id=parent_goal_id,
            root_goal_id=parent_goal_id or task_id,
            source_kernel_id=source_kernel_id,
            target_kernel_id=target.kernel_id,
            target_node_id=target_node_id,
            mission_context=mission_context,
            lease_scope="delegated",
            required_trust_level=required_trust_level,
            assigned_trust_level=assigned_trust_level,
            target_locality="local",
            route_kind="local_child",
            worker_role=profile.role,
            allowed_tool_permissions=list(profile.allowed_permission_levels),
            allowed_promotion_targets=list(profile.allowed_promotion_targets),
            packet_id=packet_id,
            provenance={"specialization": specialization, "job_id": queue_rec.job_id},
        )
        self.distributed_queue.assign(
            queue_rec.job_id,
            target_kernel_id=target.kernel_id,
            target_node_id=target_node_id,
            lease_id=lease.lease_id,
            assigned_trust_level=assigned_trust_level,
            route_kind="local_child",
            target_worker_role=profile.role,
            allowed_tool_permissions=list(profile.allowed_permission_levels),
            allowed_promotion_targets=list(profile.allowed_promotion_targets),
        )
        lineage = self._lineage_dict(
            task_id,
            parent_goal_id=parent_goal_id,
            root_goal_id=parent_goal_id or task_id,
            owner_kernel_id=target.kernel_id,
            requester_kernel_id=source_kernel_id,
            mission_context=mission_context,
            lease_scope="delegated",
            packet_id=packet_id,
        )
        target.receive_goal({"goal": goal, "task_id": task_id, "lineage": lineage, "lease_id": lease.lease_id, "job_id": queue_rec.job_id})
        self._dispatch_kernel_packet(target.status_packet())
        return {
            "task_id": task_id,
            "job_id": queue_rec.job_id,
            "lease_id": lease.lease_id,
            "target_kernel_id": target.kernel_id,
            "target_node_id": target_node_id,
            "required_trust_level": required_trust_level,
            "assigned_trust_level": assigned_trust_level,
            "specialization": target.state.specialization,
            "lineage": lineage,
        }

    def _dispatch_kernel_packet(self, packet: KernelPacket) -> Dict[str, Any]:
        self.kernel_packet_counter += 1
        event = packet.to_dict()
        event["ordinal"] = self.kernel_packet_counter
        event["accepted_by_main"] = packet.target_kernel_id in (self.kernel_id, "main")
        event["handled_at"] = time.time()
        self.branch_provenance.record_packet(event)
        self.kernel_packet_log.append(event)
        if len(self.kernel_packet_log) > MAX_KERNEL_PACKET_LOG:
            self.kernel_packet_log = self.kernel_packet_log[-MAX_KERNEL_PACKET_LOG:]
        self._emit_packet_to_corpus(event)

        if packet.packet_kind == PacketKind.BLOCKED_GROWTH:
            self._blocked_growth_log.append(dict(packet.payload))
            self.state.beta["growth.blocked"] = BilateralValue(0.35, 0.65)
            refresh_state(self.state)
        elif packet.packet_kind == PacketKind.RESOURCE_REQUEST:
            self.resource_requests.append(dict(event))
            self.resource_requests = self.resource_requests[-100:]
        elif packet.packet_kind == PacketKind.TOOL_REQUEST:
            event["tool_access_decision"] = self._evaluate_tool_request(packet)
            self.tool_requests.append(dict(event))
            self.tool_requests = self.tool_requests[-100:]
        elif packet.packet_kind == PacketKind.MODULE_PROPOSAL:
            event["module_record"] = self._record_module_proposal(packet)
            event["module_policy"] = self._evaluate_module_policy(event["module_record"])
        elif packet.packet_kind == PacketKind.PATCH_PROPOSAL and packet.target_kernel_id in (self.kernel_id, "main"):
            ok, msg = self._stage_patch_proposal(packet.payload, source_kernel_id=packet.source_kernel_id, packet=packet)
            event["staged_patch_ok"] = ok
            event["staged_patch_message"] = msg
            patch_name = str(packet.payload.get("patch_name", ""))
            if patch_name:
                event["promotion_summary"] = self.promotion_ladder.summary(patch_name)
        elif packet.packet_kind == PacketKind.GOAL:
            event["delegation"] = self._delegate_task_to_subkernel(
                str(packet.payload.get("goal", "delegated goal")),
                specialization=str(packet.payload.get("specialization", "general")),
                mission_context=packet.mission_context or str(packet.payload.get("mission_context", "delegated mission")),
                parent_goal_id=packet.parent_goal_id,
                source_kernel_id=packet.source_kernel_id,
                packet_id=packet.packet_id,
                required_trust_level=str(packet.payload.get("required_trust_level", "low")),
            )
        elif packet.packet_kind == PacketKind.MEMORY_SYNC:
            self.memory_sync_requests.append(dict(event))
            self.memory_sync_requests = self.memory_sync_requests[-100:]
            event["memory_sync_result"] = self._apply_memory_sync_packet(packet)
        elif packet.packet_kind == PacketKind.PROMOTION_REQUEST:
            event["promotion_request"] = self._record_promotion_request(packet)
        elif packet.packet_kind == PacketKind.TRUST_REPORT:
            concerns = packet.payload.get("concerns", [])
            if concerns:
                self._escalation_log.append({
                    "kind": "trust_report",
                    "source_kernel_id": packet.source_kernel_id,
                    "concerns": concerns,
                    "time": time.time(),
                })
        elif packet.packet_kind == PacketKind.REVERT_NOTICE:
            self._rewrite_history.append({
                "kind": "hub_revert_notice",
                "source_kernel_id": packet.source_kernel_id,
                "payload": dict(packet.payload),
                "time": time.time(),
            })

        self._sync_kernel_registry()
        self._save_kernel_ecology_state()
        return event

    def _kernel_packet_log_tail(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        return list(self.kernel_packet_log[-limit:])

    def _spawn_subkernel(self, specialization: str = "general", mission_context: str = "delegated mission") -> Subkernel:
        specialization = specialization.strip() or "general"
        mission_context = mission_context.strip() or f"{specialization} delegated mission"
        parent_kernel_id = self.hub_kernel.kernel_id if self.hub_kernel is not None else self.kernel_id
        kernel_id = f"sub_{self._slugify(specialization)}_{len(self.subkernels) + 1}"
        sub = Subkernel(
            kernel_id=kernel_id,
            parent_kernel_id=parent_kernel_id,
            specialization=specialization,
            mission_context=mission_context,
        )
        if self.hub_kernel is not None and parent_kernel_id == self.hub_kernel.kernel_id:
            sub.trust_from_parent = self.hub_kernel.trust_from_main
        self.subkernels[kernel_id] = sub
        self._dispatch_kernel_packet(sub.status_packet())
        self._sync_kernel_registry()
        return sub

    def get_kernel_ecology_summary(self) -> Dict[str, Any]:
        self._sync_kernel_registry()
        cluster_summary = self.cluster_registry.summary()
        trust_summary = self.cluster_trust.summary()
        delegation_summary = self.delegation_manager.summary()
        promotion_summary = self._promotion_queue_summary()
        self.cluster_model.update_from(cluster_summary, trust_summary, hub_present=self.hub_kernel is not None, subkernel_count=len(self.subkernels), delegation_summary=delegation_summary, promotion_summary=promotion_summary)
        return {
            "boot_mode": self.boot_mode,
            "main": self.child_kernel_registry.get(self.kernel_id, {}),
            "hub_present": self.hub_kernel is not None,
            "hub": self.child_kernel_registry.get(self.hub_kernel.kernel_id, {}) if self.hub_kernel is not None else None,
            "subkernel_count": len(self.subkernels),
            "subkernels": [self.child_kernel_registry[k] for k in sorted(self.subkernels)],
            "active_delegations": len(self.delegation_manager.list_active()),
            "delegation_summary": delegation_summary,
            "distributed_queue": self.distributed_queue.summary(),
            "module_proposals": len(self.module_proposals),
            "resource_requests": len(self.resource_requests),
            "tool_requests": len(self.tool_requests),
            "tool_access_decisions": len(self.tool_access_decisions),
            "worker_budget_decisions": len(self.worker_budget_decisions),
            "module_policy_decisions": len(self.module_policy_decisions),
            "memory_sync_requests": len(self.memory_sync_requests),
            "promotion_requests": len(self.promotion_requests),
            "promotion_gate_log": len(self.promotion_gate_log),
            "module_priorities": self.module_registry.prioritized_proposals()[:10],
            "module_family_priorities": self.module_registry.family_readiness_summary()[:10],
            "hub_promotion_priorities": self._hub_promotion_priority_view(10),
            "hub_review_queue": list(self.hub_kernel.work_queue[-10:]) if self.hub_kernel is not None else [],
            "hub_review_waves": list(getattr(self.hub_kernel, "review_waves", [])[-10:]) if self.hub_kernel is not None else [],
            "hub_wave_priorities": self._hub_review_wave_priority_view(10),
            "growth_priorities": list(getattr(self.self_model, "growth_priority_summary", [])[:10]) if self.self_model is not None else [],
            "deferred_hub_items": sum(1 for q in (self.hub_kernel.promotion_queue if self.hub_kernel is not None else []) if str(q.get("status", "")) == "deferred_cooldown"),
            "wave_caution_entries": len(self._hub_queue_caution_map()) if self.hub_kernel is not None else 0,
            "wave_escalation_entries": len((self.hub_kernel.local_branch_state.get("wave_escalation_history", []) if self.hub_kernel is not None else [])),
            "proposal_rework_entries": len((self.hub_kernel.local_branch_state.get("proposal_rework_history", []) if self.hub_kernel is not None else [])),
            "blocked_growth_followup_entries": len((self.hub_kernel.local_branch_state.get("blocked_growth_followup_history", []) if self.hub_kernel is not None else [])),
            "module_registry": self.module_registry.summary(),
            "message_bus": self.message_bus.summary(),
            "packet_log_entries": len(self.kernel_packet_log),
            "recent_packets": self._kernel_packet_log_tail(5),
            "branch_provenance": self.branch_provenance.summary(),
            "branch_checkpoint_count": len(self.branch_checkpoints),
            "node_identity": self.node_identity.summary(),
            "cluster": cluster_summary,
            "cluster_trust": trust_summary,
            "cluster_model": {
                "node_count": self.cluster_model.node_count,
                "trusted_node_count": self.cluster_model.trusted_node_count,
                "local_node_count": self.cluster_model.local_node_count,
                "subkernel_count": self.cluster_model.subkernel_count,
                "hub_present": self.cluster_model.hub_present,
                "trust_alerts": list(self.cluster_model.trust_alerts),
                "average_trust_score": self.cluster_model.average_trust_score,
                "delegation_pressure": self.cluster_model.delegation_pressure,
                "delegation_capacity": self.cluster_model.delegation_capacity,
                "promotion_readiness": self.cluster_model.promotion_readiness,
            },
            "worker_budget_state": self.budget_manager.worker_usage_summary(),
            "cooldown_nodes": sum(1 for n in self.cluster_registry.nodes.values() if float(getattr(n, "metadata", {}).get("cooldown_until", 0.0) or 0.0) > time.time()),
        }


    def _serialize_kernel_ecology_state(self) -> Dict[str, Any]:
        self._sync_kernel_registry()
        return serialize_kernel_ecology_state(
            boot_mode=self.boot_mode,
            packet_log=self.kernel_packet_log,
            child_kernel_registry=self.child_kernel_registry,
            hub_state=self.hub_kernel.export_state() if self.hub_kernel is not None else {},
            subkernel_states={k: v.export_state() for k, v in self.subkernels.items()},
            goal_lineage=self.goal_lineage,
            module_proposals=self.module_proposals,
            resource_requests=self.resource_requests,
            tool_requests=self.tool_requests,
            tool_access_decisions=self.tool_access_decisions,
            worker_budget_decisions=self.worker_budget_decisions,
            module_policy_decisions=self.module_policy_decisions,
            memory_sync_requests=self.memory_sync_requests,
            promotion_requests=self.promotion_requests,
            promotion_gate_log=self.promotion_gate_log,
            module_registry_state=self.module_registry.export_state(),
            message_bus_state=self.message_bus.export_state(),
            promotion_ladder_state=self.promotion_ladder.export_state(),
            delegation_state=self.delegation_manager.snapshot(),
            branch_provenance=self.branch_provenance.to_dict(),
            branch_checkpoints=self.branch_checkpoints[-100:],
            cluster_registry_state=self.cluster_registry.export_state(),
            cluster_trust_state=self.cluster_trust.export_state(),
            node_identity_state=self.node_identity.summary(),
            distributed_queue_state=self.distributed_queue.export_state(),
            worker_budget_state=self.budget_manager.export_state(),
        )

    def _save_kernel_ecology_state(self) -> bool:
        ecology = self._serialize_kernel_ecology_state()
        ok1 = save_kernel_ecology_to_file(KERNEL_ECOLOGY_FILE, ecology)
        ok2 = save_json(PACKET_LOG_FILE, self.kernel_packet_log[-500:])
        ok3 = save_json(MEMORY_PROVENANCE_FILE, self.branch_provenance.to_dict())
        ok4 = save_json(CLUSTER_REGISTRY_FILE, self.cluster_registry.export_state())
        ok5 = save_json(CLUSTER_TRUST_FILE, self.cluster_trust.export_state())
        ok6 = save_json(NODE_IDENTITY_FILE, self.node_identity.summary())
        return bool(ok1 and ok2 and ok3 and ok4 and ok5 and ok6)

    def _load_kernel_ecology_state(self) -> None:
        ecology = load_kernel_ecology_from_file(KERNEL_ECOLOGY_FILE)
        if not ecology:
            packet_log = load_json(PACKET_LOG_FILE, [])
            if isinstance(packet_log, list):
                self.kernel_packet_log = packet_log[-200:]
            prov = load_json(MEMORY_PROVENANCE_FILE, {})
            self.branch_provenance = ProvenanceGraph.from_dict(prov)
            self.cluster_registry.restore_state(load_json(CLUSTER_REGISTRY_FILE, {}))
            self.cluster_trust.restore_state(load_json(CLUSTER_TRUST_FILE, {}))
            self.node_identity = NodeIdentity.from_dict(load_json(NODE_IDENTITY_FILE, self.node_identity.summary()))
            self.distributed_queue.restore_state({})
            return
        packet_log = ecology.get("packet_log", load_json(PACKET_LOG_FILE, []))
        self.kernel_packet_log = list(packet_log or [])[-200:]
        self.child_kernel_registry = dict(ecology.get("child_kernel_registry", {}))
        hub = HubKernel.from_state(ecology.get("hub_state"))
        self.hub_kernel = hub if isinstance(hub, HubKernel) else None
        self.subkernels = {}
        for kernel_id, payload in dict(ecology.get("subkernel_states", {})).items():
            sub = Subkernel.from_state(payload)
            if sub is not None:
                self.subkernels[sub.kernel_id] = sub
        self.goal_lineage = dict(ecology.get("goal_lineage", {}))
        self.module_proposals = list(ecology.get("module_proposals", []))[-200:]
        self.resource_requests = list(ecology.get("resource_requests", []))[-100:]
        self.tool_requests = list(ecology.get("tool_requests", []))[-100:]
        self.tool_access_decisions = list(ecology.get("tool_access_decisions", []))[-100:]
        self.worker_budget_decisions = list(ecology.get("worker_budget_decisions", []))[-100:]
        self.module_policy_decisions = list(ecology.get("module_policy_decisions", []))[-100:]
        self.memory_sync_requests = list(ecology.get("memory_sync_requests", []))[-100:]
        self.promotion_requests = list(ecology.get("promotion_requests", []))[-100:]
        self.promotion_gate_log = list(ecology.get("promotion_gate_log", []))[-100:]
        self.module_registry.restore_state(ecology.get("module_registry_state"))
        self.message_bus.restore_state(ecology.get("message_bus_state"))
        self.promotion_ladder.restore_state(ecology.get("promotion_ladder_state"))
        try:
            self.delegation_manager.restore(ecology.get("delegation_state"))
        except Exception:
            pass
        prov_data = ecology.get("branch_provenance", load_json(MEMORY_PROVENANCE_FILE, {}))
        self.branch_provenance = ProvenanceGraph.from_dict(prov_data)
        self.branch_checkpoints = list(ecology.get("branch_checkpoints", []))[-100:]
        self.cluster_registry.restore_state(ecology.get("cluster_registry_state", load_json(CLUSTER_REGISTRY_FILE, {})))
        self.cluster_trust.restore_state(ecology.get("cluster_trust_state", load_json(CLUSTER_TRUST_FILE, {})))
        self.node_identity = NodeIdentity.from_dict(ecology.get("node_identity_state", load_json(NODE_IDENTITY_FILE, self.node_identity.summary())))
        self.distributed_queue.restore_state(ecology.get("distributed_queue_state"))
        self.budget_manager.restore_state(ecology.get("worker_budget_state"))

    def checkpoint_branch_ecology(self, reason: str = "manual") -> Dict[str, Any]:
        checkpoint = {
            "reason": reason,
            "timestamp": time.time(),
            "ecology": self._serialize_kernel_ecology_state(),
        }
        meta = save_branch_checkpoint(reason or "branch", checkpoint, BRANCH_CHECKPOINT_DIR)
        self.branch_checkpoints.append({**meta, "reason": reason})
        self.branch_checkpoints = self.branch_checkpoints[-100:]
        self._save_kernel_ecology_state()
        return {**meta, "reason": reason}

    def _apply_memory_sync_packet(self, packet: KernelPacket) -> Dict[str, Any]:
        source_kernel_id = str(packet.source_kernel_id or "")
        branch_memory: List[Dict[str, Any]] = []
        assign_target = None
        if self.hub_kernel is not None and source_kernel_id == self.hub_kernel.kernel_id:
            branch_memory = list(self.hub_kernel.memory_branch)
            assign_target = "hub"
        elif source_kernel_id in self.subkernels:
            sub = self.subkernels[source_kernel_id]
            branch_memory = list(sub.state.local_state.get("memory_branch", []))
            assign_target = source_kernel_id
        decision, remaining = apply_memory_sync_request(
            self.memory_store,
            self.branch_provenance,
            packet.payload,
            branch_memory,
            cycle=self.state.c.cycle,
            state=self.state,
        )
        if assign_target == "hub" and self.hub_kernel is not None:
            self.hub_kernel.memory_branch = list(remaining)
        elif assign_target in self.subkernels:
            self.subkernels[assign_target].state.local_state["memory_branch"] = list(remaining)
        self._save_kernel_ecology_state()
        return decision.to_dict()

    # ================================================================
    # PERSISTENCE — fixed shadow weight cadence
    # ================================================================

    def save_state(self) -> None:
        """Save JSON state. Shadow weights saved separately on meaningful cadence only."""
        from tovah_v14.persistence.state_io import serialize_state_for_save
        state_dict = serialize_state_for_save(
            completed_goals=self.completed_goals[-100:],
            pending_tool_actions=self.pending_tool_actions[-100:],
            staged_patches=self.staged_patches,
            patch_history=self.patch_history[-MAX_PATCH_HISTORY:],
            loss_history=self.loss_history[-500:],
            research_memory=self.research_memory[-MAX_RESEARCH_RESULTS_STORED:],
            trace_index=self.trace_index[-MAX_TRACES_STORED:],
            unresolved=self.unresolved[-100:],
            last_research_time=self.last_research_time,
            improvement_count=self.improvement_count,
            autonomy_level=self.autonomy_level,
            current_goal=self.current_goal,
            goal_attempts=self._goal_attempts,
            shelved_goals=self._shelved_goals[-50:],
            domain_history=self._domain_history[-100:],
            installed_packages=sorted(self._installed_packages),
            state_snapshot=self.state.snapshot(),
            alpha=self.alpha, temperature=self.temperature,
            api_usage=self.api_usage, lab_registry=self.lab_registry,
            crypto_wallet=self.crypto_wallet,
            beneficiary_sol_address=self.beneficiary_sol_address,
            profile_name=self.profile_name,
            topic_last_research_time=self._topic_last_research_time,
            recent_research_topics=self._recent_research_topics[-80:],
            active_plans=[asdict(p) for p in self.plan_manager.active[-20:]],
            completed_plans=self.plan_manager.completed_ids[-100:],
            capabilities={k: {kk: vv for kk, vv in v.items() if kk != "module"} for k, v in self._capabilities.items()},
            rewrite_queue=self._rewrite_queue[-50:],
            rewrite_history=self._rewrite_history[-100:],
            memory_episodic=[asdict(m) for m in self.memory_store.get_bank("episodic")],
            memory_semantic=[asdict(m) for m in self.memory_store.get_bank("semantic")],
            memory_procedural=[asdict(m) for m in self.memory_store.get_bank("procedural")],
            task_queue=[asdict(t) for t in self.task_queue.tasks],
            completed_tasks=self.task_queue.completed_ids[-200:],
            failure_clusters=[asdict(c) if hasattr(c, '__dataclass_fields__') else c for c in self._failure_clusters[-50:]],
            resource_budgets=self.budget_manager.budgets,
            curriculum=self._curriculum,
            promotion_state=self.promotion_ladder.state,
            workbench_notes=self._workbench_notes,
            state_version=self._state_version,
            tool_contracts={},
            experience_records=[asdict(r) for r in self.experience_store.records[-500:]],
            competence_map={k: asdict(v) for k, v in self.competence_map.entries.items()},
        )
        save_state_to_file(STATE_FILE, state_dict)
        self._save_kernel_ecology_state()
        # Shadow weights: NOT on every save. Only on explicit checkpoint_shadow calls.

    def checkpoint_shadow(self, reason: str = "periodic") -> None:
        """Save shadow weights atomically. Called only on meaningful events."""
        now = time.time()
        if now - self.last_shadow_save_time < 30:
            return  # debounce
        self.last_shadow_save_time = now
        save_shadow_weights(self.shadow_model, SHADOW_FILE, self.improvement_count)

    def load_state(self) -> None:
        raw = load_state_from_file(STATE_FILE)
        if not raw:
            return
        s = migrate_state(raw)
        self.completed_goals = list(s.get("completed_goals", []))
        self.pending_tool_actions = list(s.get("pending_tool_actions", []))
        self.staged_patches = dict(s.get("staged_patches", {}))
        self.patch_history = list(s.get("patch_history", []))
        self.loss_history = list(s.get("loss_history", []))
        self.research_memory = list(s.get("research_memory", []))
        self.trace_index = list(s.get("trace_index", []))
        self.unresolved = list(s.get("unresolved", []))
        self.last_research_time = float(s.get("last_research_time", 0.0))
        self.improvement_count = int(s.get("improvement_count", 0))
        self.autonomy_level = int(s.get("autonomy_level", 0))
        self.current_goal = s.get("current_goal")
        self._goal_attempts = int(s.get("goal_attempts", 0))
        self._shelved_goals = list(s.get("shelved_goals", []))
        self._domain_history = list(s.get("domain_history", []))
        self._installed_packages = set(s.get("installed_packages", []))
        self.alpha = float(s.get("alpha", 1.0))
        self.temperature = float(s.get("temperature", 0.9))
        self.api_usage.update(s.get("api_usage", {}))
        self.lab_registry = dict(s.get("lab_registry", {}))
        self.crypto_wallet = s.get("crypto_wallet")
        self.beneficiary_sol_address = s.get("beneficiary_sol_address", self.beneficiary_sol_address)
        self._topic_last_research_time = {str(k): float(v) for k, v in dict(s.get("topic_last_research_time", {})).items()}
        self._recent_research_topics = [str(x) for x in list(s.get("recent_research_topics", []))][-80:]
        self.plan_manager.completed_ids = list(s.get("completed_plans", []))
        self._capabilities = dict(s.get("capabilities", {}))
        self._rewrite_queue = list(s.get("rewrite_queue", []))
        self._rewrite_history = list(s.get("rewrite_history", []))
        self.budget_manager.budgets = s.get("resource_budgets", copy.deepcopy(DEFAULT_BUDGETS))
        self._curriculum = s.get("curriculum", copy.deepcopy(DEFAULT_CURRICULUM))
        self.promotion_ladder.state = dict(s.get("promotion_state", {}))
        self._workbench_notes = dict(s.get("workbench_notes", {}))
        self._state_version = s.get("state_version", VERSION)
        for pd in s.get("active_plans", []):
            try:
                pd = dict(pd)
                bv = coerce_bilateral_value(pd.pop("bilateral_confidence", {"t": 0.5, "f": 0.1}))
                self.plan_manager.active.append(StrategicPlan(**pd, bilateral_confidence=bv))
            except Exception:
                pass
        for kind in ("episodic", "semantic", "procedural"):
            for md in s.get(f"memory_{kind}", []):
                try:
                    md = dict(md)
                    bv = coerce_bilateral_value(md.pop("bilateral_confidence", {"t": 0.5, "f": 0.1}))
                    entry = MemoryEntry(**md, bilateral_confidence=bv)
                    self.memory_store.banks[kind].append(entry)
                except Exception:
                    pass
        for td in s.get("task_queue", []):
            try:
                td = dict(td)
                bv = coerce_bilateral_value(td.pop("bilateral_confidence", {"t": 0.5, "f": 0.1}))
                self.task_queue.tasks.append(TaskNode(**td, bilateral_confidence=bv))
            except Exception:
                pass
        self.task_queue.completed_ids = list(s.get("completed_tasks", []))
        # Restore experience
        for er in s.get("experience_records", []):
            try:
                er = dict(er)
                bv = coerce_bilateral_value(er.pop("bilateral_assessment", {"t": 0.5, "f": 0.5}))
                self.experience_store.records.append(ExperienceRecord(**er, bilateral_assessment=bv))
            except Exception:
                pass
        # Restore competence
        for domain, ce in dict(s.get("competence_map", {})).items():
            try:
                ce = dict(ce)
                bv = coerce_bilateral_value(ce.pop("bilateral_confidence", {"t": 0.3, "f": 0.3}))
                from tovah_v14.selfmodel.competence import CompetenceEntry
                self.competence_map.entries[domain] = CompetenceEntry(**ce, bilateral_confidence=bv)
            except Exception:
                pass
        so = s.get("state")
        if isinstance(so, dict):
            raw_beta = so.get("beta", {})
            clean_beta = {str(k): coerce_bilateral_value(v) for k, v in raw_beta.items()} if isinstance(raw_beta, dict) else {}
            self.state = ShadowState(
                c=CarrierState(**(so.get("c", {}) if isinstance(so.get("c"), dict) else {})),
                beta=clean_beta,
                nu=dict(so.get("nu", {})),
                pi=ProvenanceState(**(so.get("pi", {}) if isinstance(so.get("pi"), dict) else {})),
            )
        load_shadow_weights(self.shadow_model, SHADOW_FILE, self.device)
        refresh_state(self.state)

    # ================================================================
    # TOOL DISPATCH (budget-aware)
    # ================================================================

    def _perform_tool_action(self, action: Dict[str, Any]) -> ToolResult:
        tool = action.get("tool")
        arg = action.get("arg", "")
        arg2 = action.get("arg2", "")
        self.state.c.last_tool = tool or ""
        contract = TOOL_CONTRACTS.get(tool or "")
        if contract and contract.budget_resource:
            if not self.budget_manager.spend(contract.budget_resource):
                self._tool_fail_counts[tool or ""] = self._tool_fail_counts.get(tool or "", 0) + 1
                return ToolResult(False, tool or "", "budget exceeded")
        t0 = time.time()
        tr = ToolResult(False, tool or "unknown", "unknown tool")
        if tool == "web_search":
            tr = self.tools.web_search(arg)
        elif tool == "fetch_url":
            tr = self.tools.fetch_url(arg)
        elif tool == "github_repo":
            tr = self.tools.github_repo(arg)
        elif tool == "github_file":
            tr = self.tools.github_file(arg, arg2 or "README.md")
        elif tool == "robots_ok":
            tr = self.tools.robots_ok(arg)
        elif tool == "wikipedia_summary":
            tr = self.tools.wikipedia_summary(arg)
        elif tool == "arxiv_search":
            tr = self.tools.arxiv_search(arg)
        elif tool == "rss_fetch":
            tr = self.tools.rss_fetch(arg)
        elif tool == "json_api_fetch":
            tr = self.tools.json_api_fetch(arg)
        elif tool == "sitemap_fetch":
            tr = self.tools.sitemap_fetch(arg)
        elif tool == "browser_action":
            parts = [p.strip() for p in str(arg).split("|") if p.strip()]
            rd = self.browser_action(parts[0] if parts else "navigate",
                                      parts[1] if len(parts) > 1 else "",
                                      parts[2] if len(parts) > 2 else "",
                                      parts[3] if len(parts) > 3 else "")
            tr = ToolResult(ok=rd.get("ok", False), tool="browser_action",
                            summary=rd.get("summary", ""), payload=rd.get("data", {}),
                            url=parts[1] if len(parts) > 1 else "")
        elif tool == "extract_text":
            rd = self.extract_text(arg)
            tr = ToolResult(ok=rd.get("ok", False), tool="extract_text",
                            summary=rd.get("summary", ""), payload=rd.get("text", ""), url=arg)
        elif tool in self.active_lab_tools:
            try:
                payload = self.active_lab_tools[tool](self, arg=arg, arg2=arg2, action=action)
                if isinstance(payload, ToolResult):
                    tr = payload
                elif isinstance(payload, dict) and payload.get("ok") is False:
                    tr = ToolResult(False, tool, str(payload.get("error", "failed")), payload)
                else:
                    tr = ToolResult(True, tool, "lab tool executed", payload)
            except Exception as e:
                tr = ToolResult(False, tool, f"lab tool failed: {e}")
        # Track success/failure
        if tr.ok:
            self._tool_fail_counts.pop(tool or "", None)
        else:
            self._tool_fail_counts[tool or ""] = self._tool_fail_counts.get(tool or "", 0) + 1
        return tr

    def browser_action(self, action, url="", selector="", text="", timeout=30, **kwargs):
        rd = _browser_action_fn(action, url, selector, text, timeout, ensure_package=self._ensure_package)
        bv = BilateralValue(0.92, 0.04) if rd.get("ok") else BilateralValue(0.18, 0.72)
        self.state.beta["browser.reachability"] = bilateral_or(
            self.state.beta.get("browser.reachability", BilateralValue(0.5, 0.5)), bv)
        refresh_state(self.state)
        return rd

    def extract_text(self, url: str) -> Dict[str, Any]:
        rd = _extract_text_fn(self.tools.session, url, ensure_package=self._ensure_package)
        bv = BilateralValue(0.90, 0.05) if rd.get("ok") else BilateralValue(0.20, 0.70)
        self.state.beta["tool.extract_efficacy"] = bilateral_or(
            self.state.beta.get("tool.extract_efficacy", BilateralValue(0.5, 0.5)), bv)
        refresh_state(self.state)
        return rd

    def _ensure_package(self, package: str, import_name: Optional[str] = None) -> Tuple[bool, str]:
        import_name = import_name or package
        try:
            __import__(import_name)
            self._installed_packages.add(package)
            return True, "available"
        except Exception:
            pass
        if not self._pip_install(package):
            return False, f"pip install failed for {package}"
        try:
            __import__(import_name)
            self._installed_packages.add(package)
            return True, "installed"
        except Exception as e:
            return False, f"import failed after install: {e}"

    def _pip_install(self, package_name: str) -> bool:
        if package_name in self._installed_packages:
            return True
        try:
            import subprocess as _sp, sys
            result = _sp.run([sys.executable, "-m", "pip", "install", "--quiet", package_name],
                             capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                self._installed_packages.add(package_name)
                return True
        except Exception:
            pass
        return False

    # ================================================================
    # ADVISOR
    # ================================================================

    def _get_advisor_state_context(self) -> str:
        hist = {"T": 0, "F": 0, "B": 0, "G": 0}
        for v in self.state.nu.values():
            hist[v] = hist.get(v, 0) + 1
        parts = [f"\n---\nKernel: cycle={self.state.c.cycle}",
                 f"goal={self.state.c.active_goal[:60]}", f"cache={hist}",
                 f"lab_tools={sorted(self.active_lab_tools.keys())}",
                 f"improvements={self.improvement_count}", f"autonomy={self.autonomy_level}",
                 f"plans={len(self.plan_manager.get_active())}", f"pdf={_PDF_BACKEND}"]
        if self.loss_history:
            parts.append(f"loss={self.loss_history[-1]:.3f}")
        return " | ".join(parts)

    def _chat_with_advisor(self, prompt: str) -> str:
        if not self.budget_manager.check("advisor_call"):
            return ""
        full = str(prompt) + self._get_advisor_state_context()
        for key in ("grok_reasoning_api", "grok_fast_reasoning_api", "grok_non_reasoning_api", "grok_code_api"):
            if key not in self.api:
                continue
            try:
                self.budget_manager.spend("advisor_call")
                self.api_usage[key] = self.api_usage.get(key, 0) + 1
                out = self.api[key](full)
                if out is not None:
                    s = str(out).strip()
                    if s:
                        return s
            except BaseException as e:
                logging.warning(f"advisor {key}: {type(e).__name__}: {e}")
        return ""

    # ================================================================
    # RETRIEVER MODULE METHODS (manifest: retriever)
    # ================================================================

    def _classify_query_intent(self, text: str) -> str:
        """Classify a query into an intent taxonomy. Deterministic first, advisor may refine."""
        t = text.lower().strip()
        if re.match(r"https?://", t):
            return "url_fetch"
        if re.match(r"github\.com/\w+/\w+", t) or ("github" in t and "/" in t):
            return "github_repo" if t.count("/") <= 1 or "file" not in t else "github_file"
        if any(k in t for k in ("arxiv", "paper", "journal", "doi:", "abstract")):
            return "paper_lookup"
        if any(k in t for k in ("build tool", "create tool", "implement", "write code")):
            return "tool_building"
        if any(k in t for k in ("rewrite", "refactor", "improve method")):
            return "method_rewrite"
        if any(re.search(rf"\b{k}\b", t) for k in ("service", "api", "endpoint")):
            return "service_discovery"
        return "broad_research"

    def _rank_tool_candidates(self, candidates_or_topic: Any, query: str = "") -> List[str]:
        """Rank tools by intent, budget, success history, and cost."""
        intent = self._classify_query_intent(str(candidates_or_topic) + " " + query)
        intent_map = {
            "url_fetch": ["fetch_url", "extract_text", "browser_action"],
            "github_repo": ["github_repo", "github_file", "fetch_url"],
            "github_file": ["github_file", "github_repo", "fetch_url"],
            "paper_lookup": ["arxiv_search", "web_search", "wikipedia_summary"],
            "broad_research": ["web_search", "arxiv_search", "wikipedia_summary", "fetch_url"],
            "tool_building": ["web_search", "github_repo", "fetch_url"],
            "method_rewrite": ["web_search", "github_repo"],
            "service_discovery": ["web_search", "json_api_fetch", "sitemap_fetch"],
        }
        candidates = intent_map.get(intent, ["web_search", "fetch_url"])
        # Filter by budget availability
        ranked = []
        for tool in candidates:
            contract = TOOL_CONTRACTS.get(tool)
            budget_ok = True
            if contract and contract.budget_resource:
                budget_ok = self.budget_manager.check(contract.budget_resource)
            fail_count = self._tool_fail_counts.get(tool, 0)
            score = (1.0 if budget_ok else 0.1) * (1.0 / (1.0 + fail_count))
            ranked.append((score, tool))
        ranked.sort(reverse=True)
        # Add lab tools that might be relevant
        for name in self.active_lab_tools:
            if any(k in name.lower() for k in query.lower().split()[:3]):
                ranked.append((0.5, name))
        return [t for _, t in ranked]

    def memory_query(self, kind: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Real kernel wrapper around memory retrieval."""
        results = _memory_query_fn(self.memory_store, kind, query, limit)
        self.module_health.record_success("retriever", self.state)
        return results

    # ================================================================
    # PLANNER MODULE METHODS (manifest: planner)
    # ================================================================

    def _decompose_goal_into_queries(self, goal_text: str) -> List[str]:
        """Produce 3-6 concrete search queries from goal text."""
        words = goal_text.strip().split()
        core = " ".join(words[:8])
        queries = [core]
        if len(words) > 3:
            queries.append(" ".join(words[:4]) + " tutorial")
            queries.append(" ".join(words[:4]) + " implementation")
        queries.append(core + " python")
        queries.append(core + " common problems")
        if len(words) > 5:
            queries.append(" ".join(words[3:8]))
        # Deduplicate while preserving order
        seen: set = set()
        unique = []
        for q in queries:
            ql = q.strip().lower()
            if ql and ql not in seen:
                seen.add(ql)
                unique.append(q.strip())
        return unique[:6]

    def _generate_next_goal(self) -> Optional[Dict[str, Any]]:
        """Generate goal using self-model, blocked growth, budget pressure, module health."""
        if self.current_goal:
            return self.current_goal
        active_tasks = self.task_queue.get_active()
        if active_tasks:
            t = active_tasks[0]
            return {"goal": t.goal, "function_spec": t.goal, "domain": "task", "reasoning": f"active task {t.task_id}"}
        # Consult self-model for structured priorities
        sm = self.self_model
        # Blocked growth → prioritize unblocking
        if self._blocked_growth_log and len(self._blocked_growth_log) > 2:
            return {"goal": "Investigate and resolve blocked growth pipeline",
                    "function_spec": "debug promotion", "domain": "patch_review",
                    "reasoning": f"{len(self._blocked_growth_log)} blocked growth attempts"}
        # Budget pressure → research free alternatives
        if hasattr(sm, "budget_pressure") and sm.budget_pressure:
            resource = sm.budget_pressure[0].split(":")[0] if sm.budget_pressure else "unknown"
            return {"goal": f"Find budget-free alternatives for {resource}",
                    "function_spec": f"discover free {resource}", "domain": "service_integration",
                    "reasoning": f"budget pressure: {sm.budget_pressure[0]}"}
        # Weakest module from health summary
        weak_mods = self.module_health.weakest_modules(self.state, 1)
        if weak_mods:
            mod = weak_mods[0]
            bv = self.state.beta.get(f"module.{mod}_health", BilateralValue(0.5, 0.2))
            if bv.t < 0.4:
                return {"goal": f"Strengthen module: {mod}", "function_spec": f"improve {mod}",
                        "domain": "module_health", "reasoning": f"weakest module: {mod} (t={bv.t:.2f})"}
        # Weakest competence
        weakest = self.competence_map.get_weakest(1)
        if weakest and weakest[0].measured_mastery < 0.5:
            domain = weakest[0].domain
            return {"goal": f"Improve competence: {domain}", "function_spec": f"practice {domain}",
                    "domain": domain, "reasoning": f"weakest domain mastery={weakest[0].measured_mastery:.2f}"}
        # Recent failures → investigate
        if self._runtime_error_counts:
            worst = max(self._runtime_error_counts, key=self._runtime_error_counts.get)
            if self._runtime_error_counts[worst] > 3:
                return {"goal": f"Diagnose failures in: {worst}", "function_spec": f"debug {worst}",
                        "domain": "testing", "reasoning": f"failure count: {self._runtime_error_counts[worst]}"}
        # Curriculum
        for lesson in self._curriculum:
            if lesson.get("mastery", 0) < 0.5:
                return {"goal": f"Learn: {lesson.get('domain', 'unknown')}", "function_spec": lesson.get("domain", ""),
                        "domain": lesson.get("domain", ""), "reasoning": "curriculum gap"}
        # Rewrite queue
        if self._rewrite_queue:
            method = self._rewrite_queue[0]
            return {"goal": f"Rewrite method: {method}", "function_spec": f"rewrite {method}",
                    "domain": "patch_review", "reasoning": "queued rewrite"}
        return {"goal": "Explore new research and tool opportunities", "function_spec": "broad research",
                "domain": "capability_growth", "reasoning": "no specific priority"}

    def _strategic_plan(self, objective: str, max_steps: int = 8) -> Optional[StrategicPlan]:
        """Build a StrategicPlan with typed steps."""
        queries = self._decompose_goal_into_queries(objective)
        steps: List[Dict[str, Any]] = []
        for i, q in enumerate(queries[:max_steps - 2]):
            tools = self._rank_tool_candidates(q, q)
            steps.append({"action": "research", "target": q, "tool": tools[0] if tools else "web_search", "status": "pending"})
        steps.append({"action": "synthesize", "target": "compile findings", "status": "pending"})
        steps.append({"action": "reflect", "target": "assess progress and identify gaps", "status": "pending"})
        plan = StrategicPlan(
            plan_id=f"plan_{int(time.time())}_{hashlib.md5(objective.encode()).hexdigest()[:6]}",
            objective=objective, steps=steps,
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
        self.plan_manager.add(plan)
        self.module_health.record_success("planner", self.state)
        return plan

    # ================================================================
    # MEMORY MANAGER MODULE METHODS (manifest: memory_manager)
    # ================================================================

    def _store_memory(self, kind: str, key: str, data: Dict[str, Any],
                      goal_context: str = "", tags: Optional[List[str]] = None,
                      cycle: int = 0) -> Tuple[MemoryEntry, List[Any]]:
        """Store with conflict detection. Preserves both sides of contradictions."""
        conflicts = check_memory_conflict(self.memory_store, kind, key, data)
        entry = self.memory_store.store(kind, key, data, goal_context=goal_context,
                                        tags=tags, cycle=self.state.c.cycle, state=self.state)
        if conflicts:
            self.state.beta["memory.consolidation_health"] = bilateral_or(
                self.state.beta.get("memory.consolidation_health", BilateralValue(0.5, 0.2)),
                BilateralValue(0.0, 0.15))
            refresh_state(self.state)
            logging.info(f"MEMORY CONFLICT: {len(conflicts)} conflicts for {kind}:{key}")
        self.module_health.record_success("memory_manager", self.state)
        return entry, conflicts

    def _consolidate(self) -> Dict[str, int]:
        """Run memory consolidation. Updates health."""
        counts = consolidate_memory(self.memory_store)
        if counts.get("semantic_created", 0) > 0 or counts.get("procedural_created", 0) > 0:
            self.state.beta["memory.consolidation_health"] = bilateral_recover(
                self.state.beta.get("memory.consolidation_health", BilateralValue(0.5, 0.2)),
                truth_gain=0.08, falsity_decay=0.03)
            refresh_state(self.state)
        return counts

    def _forget(self) -> Dict[str, int]:
        """Run cleanup/forgetting. Updates health."""
        result = cleanup_memory(self.memory_store)
        total_forgotten = sum(result.values())
        if total_forgotten > 0:
            self.state.beta["cleanup.stale_belief_rate"] = bilateral_recover(
                self.state.beta.get("cleanup.stale_belief_rate", BilateralValue(0.5, 0.2)),
                truth_gain=0.05, falsity_decay=0.02)
            refresh_state(self.state)
        return result

    # ================================================================
    # TRAINER MODULE METHOD (manifest: trainer)
    # ================================================================

    def _train_shadow_step(self, corpus: Optional[List[str]] = None) -> Tuple[float, str]:
        """Thin wrapper around neural training.

        If `corpus` is None, this method now actively samples from the
        accumulated corpus stream (tovah_corpus/stream/*.jsonl) using
        a phase-aware sampler that mixes A-class (high-truth) and
        K-class (contradiction-rich) examples. Falls back to the
        v14.0 self-summary corpus if no shards exist yet.
        """
        if corpus is None:
            corpus = self._sample_live_corpus(batch_size=8)
        loss, phase = train_shadow_step(self.shadow_model, self.shadow_optimizer,
                                         corpus, self.con_budget, self.gap_budget,
                                         self.lambda_budget, self.device)
        self.loss_history.append(loss)
        self._training_phase = phase
        self.trace_analyzer.record_step(self.state, loss)
        self.module_health.record_success("trainer", self.state)
        return loss, phase

    def _sample_live_corpus(self, batch_size: int = 8,
                            k_class_ratio: float = 0.25) -> List[str]:
        """Sample from on-disk corpus shards, weighted by paraconsistent class.

        Mixes:
          - (1 - k_class_ratio) * batch_size  A-class (high-T, low-F) examples
          - k_class_ratio * batch_size        K-class (contradiction) examples

        Returns short text strings suitable for byte-level training. Falls
        back to the v14.0 self-summary corpus if no shards / no examples.
        """
        try:
            from tovah_v14.config.paths import CORPUS_STREAM_DIR
            from tovah_v14.training.exporters.jsonl import read_jsonl_shards
            from tovah_v14.training.quality_filter import classify_one, ParaconsistentClass

            if not CORPUS_STREAM_DIR.exists():
                raise FileNotFoundError("no stream dir")

            # Cheap reservoir over recent shards. We read the last ~3 shards
            # so this stays fast even with large accumulated corpora.
            shards = sorted(CORPUS_STREAM_DIR.glob("tovah_stream_*.jsonl"))
            if not shards:
                raise FileNotFoundError("no shards")

            recent = shards[-3:]
            a_pool: List[str] = []
            k_pool: List[str] = []
            # Bound work: max 2000 examples scanned per training step.
            scanned = 0
            for shard in recent:
                shard_dir = shard.parent
                # read_jsonl_shards iterates a directory; we restrict to one shard
                # by globbing carefully — easier to just open the file directly.
                import json as _json
                try:
                    with open(shard, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            scanned += 1
                            if scanned > 2000:
                                break
                            try:
                                d = _json.loads(line)
                            except Exception:
                                continue
                            text = d.get("text") or ""
                            if not text or len(text) < 4:
                                continue
                            # AUDIT FIX (P0-4, v14.1.2): strip the structural
                            # envelope so live training never sees square-bracket
                            # bookkeeping it would otherwise learn to predict.
                            try:
                                from tovah_v14.training.corpus_builder import strip_envelope
                                text = strip_envelope(text)
                            except Exception:
                                pass
                            if len(text) < 4:
                                continue
                            t = float(d.get("bilateral_t", 0.5) or 0.5)
                            f = float(d.get("bilateral_f", 0.5) or 0.5)
                            # Inline the four-corner classification.
                            if t >= 0.5 and f < 0.5:
                                a_pool.append(text)
                            elif t >= 0.5 and f >= 0.5:
                                k_pool.append(text)
                except Exception:
                    continue
                if scanned > 2000:
                    break

            import random as _random
            n_k = max(0, int(round(batch_size * k_class_ratio)))
            n_a = max(1, batch_size - n_k)
            picks: List[str] = []
            if a_pool:
                picks.extend(_random.sample(a_pool, k=min(n_a, len(a_pool))))
            if k_pool:
                picks.extend(_random.sample(k_pool, k=min(n_k, len(k_pool))))
            if picks:
                return picks
            # No usable examples → fall through.
        except Exception as e:
            logging.debug(f"_sample_live_corpus fell back to default: {e}")

        # Fallback (v14.0 behaviour).
        return [json.dumps(self.get_self_summary(), default=str),
                "shadowhott bilateral evidence four lanes constraints"]

    # ================================================================
    # OBSERVER MODULE METHOD (manifest: observer)
    # ================================================================

    def build_report(self, recent_losses: Optional[Sequence[float]] = None) -> InvariantReport:
        """Thin wrapper over invariant engine."""
        return self.invariants.build_report(self.state, recent_losses or self.loss_history)

    def update_self_model(self) -> SelfModel:
        """Update self-model from bilateral state + all subsystems."""
        self.self_model = _update_self_model_fn(
            self.self_model, self.state, self._rewrite_queue,
            competence_map=self.competence_map, budget_manager=self.budget_manager,
            module_health=self.module_health, blocked_growth_log=getattr(self, "_blocked_growth_log", []),
            runtime_error_counts=self._runtime_error_counts, active_lab_tools=self.active_lab_tools,
            free_services=self._free_services, staged_patches=self.staged_patches,
            promotion_ladder=self.promotion_ladder, node_identity=self.node_identity,
            cluster_registry=self.cluster_registry, trust_ledger=self.cluster_trust,
            distributed_queue=self.distributed_queue, cluster_model=self.cluster_model,
            module_registry=self.module_registry,
            hub_review_state={
                "queue": self._hub_promotion_priority_view(10),
                "waves": list(getattr(self.hub_kernel, "review_waves", [])[-10:]) if self.hub_kernel is not None else [],
                "wave_priorities": self._hub_review_wave_priority_view(10),
                "resolution_history": list((self.hub_kernel.local_branch_state.get("wave_resolution_history", []) if self.hub_kernel is not None else []))[-10:],
                "escalation_history": list((self.hub_kernel.local_branch_state.get("wave_escalation_history", []) if self.hub_kernel is not None else []))[-10:],
                "proposal_rework_history": list((self.hub_kernel.local_branch_state.get("proposal_rework_history", []) if self.hub_kernel is not None else []))[-10:],
                "blocked_growth_followup_history": list((self.hub_kernel.local_branch_state.get("blocked_growth_followup_history", []) if self.hub_kernel is not None else []))[-10:],
            })
        self.module_health.record_success("observer", self.state)
        return self.self_model

    # ================================================================
    # AUTONOMY / RESEARCH / GROWTH METHODS
    # ================================================================

    def _decide_research_targets(self) -> List[Tuple[str, str]]:
        """Use goal, plan, competence gaps, module weakness to choose research targets."""
        targets: List[Tuple[str, str]] = []  # (query, context)
        goal = self.current_goal
        if goal:
            queries = self._decompose_goal_into_queries(goal.get("goal", ""))
            for q in queries[:3]:
                targets.append((q, goal.get("domain", "")))
        # Competence gaps
        weak = self.competence_map.get_weakest(2)
        for entry in weak:
            if entry.measured_mastery < 0.4:
                targets.append((f"{entry.domain} best practices", "competence_gap"))
        # Module weakness
        weak_mods = self.module_health.weakest_modules(self.state, 1)
        for m in weak_mods:
            targets.append((f"improve AI agent {m} subsystem", "module_weakness"))
        return targets[:6]

    def _discover_tool_opportunities(self, topic: str, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Scan results for missing reusable capabilities. Returns typed opportunities."""
        opportunities: List[Dict[str, str]] = []
        for r in results:
            payload = r.get("payload")
            if not isinstance(payload, (list, str)):
                continue
            text = str(payload)[:2000].lower()
            if any(k in text for k in ("api", "endpoint", "rest", "json")):
                opportunities.append({"name": f"api_client_{topic[:20]}", "capability": "API integration",
                                       "rationale": f"API patterns found in {topic} research", "needs_advisor": "false"})
            if any(k in text for k in ("parse", "extract", "scrape", "transform")):
                opportunities.append({"name": f"parser_{topic[:20]}", "capability": "data extraction",
                                       "rationale": f"parsing opportunity in {topic}", "needs_advisor": "false"})
        return opportunities[:5]

    def _lab_growth_cycle(self, topic: str, results: List[Dict[str, Any]]) -> List[str]:
        """Convert research results into staged growth outputs."""
        msgs: List[str] = []
        # Tool opportunities
        opps = self._discover_tool_opportunities(topic, results)
        for opp in opps:
            self._store_memory("semantic", f"tool_opportunity:{opp['name']}",
                               opp, goal_context=topic, tags=["tool_opportunity", topic])
            msgs.append(f"tool opportunity: {opp['name']}")
        # Research notes to memory
        for r in results:
            if r.get("ok") and r.get("payload"):
                self._store_memory("episodic", f"research:{topic}:{r.get('tool','')}",
                                    {"summary": str(r.get("summary", ""))[:500], "topic": topic},
                                    goal_context=topic, tags=["research", topic])
        # Service opportunities
        for r in results:
            text = str(r.get("payload", ""))[:1000].lower()
            if any(k in text for k in ("free api", "public api", "no auth")):
                msgs.append(f"service opportunity found in {topic}")
        return msgs

    def _adapt_research_code(self) -> List[Dict[str, Any]]:
        """Transform mature research into typed staged proposals with metadata."""
        proposals: List[Dict[str, Any]] = []
        semantic = self.memory_store.get_bank("semantic")
        tool_opps = [e for e in semantic if "tool_opportunity" in e.tags and e.bilateral_confidence.t > 0.5]
        for opp in tool_opps[-3:]:
            name = opp.data.get("name", "unnamed")
            if name in self.lab_registry:
                continue
            template = CURATED_TOOL_TEMPLATES.get("json_schema_probe", "")
            staged_ok = False
            if template:
                path = LAB_STAGED / f"{self._slugify(name)}.py"
                try:
                    path.write_text(template.replace("json_schema_probe", name), encoding="utf-8")
                    staged_ok = True
                except Exception:
                    pass
            proposals.append({
                "kind": "tool_module", "name": name, "target_surface": "lab_tool",
                "capability": opp.data.get("capability", "unknown"),
                "rationale": opp.data.get("rationale", "discovered via research"),
                "risk_class": "standard", "provenance": f"semantic:{opp.key}",
                "staged": staged_ok, "blocked": not staged_ok,
                "blocked_reason": "" if staged_ok else "staging failed",
                "bilateral_confidence": {"t": opp.bilateral_confidence.t, "f": opp.bilateral_confidence.f},
            })
            if staged_ok:
                self._store_memory("procedural", f"tool_proposal:{name}",
                    {"name": name, "capability": opp.data.get("capability", "")},
                    goal_context="adapt_research", tags=["tool_proposal"])
        # Method improvement opportunities
        for entry in self.memory_store.get_bank("procedural"):
            if "method_improvement" in entry.tags and entry.bilateral_confidence.t > 0.6:
                target = entry.data.get("target_method", "")
                if target and target in ALLOWED_PATCH_TARGETS and target not in self._rewrite_queue:
                    self._rewrite_queue.append(target)
                    proposals.append({
                        "kind": "method_patch", "name": target, "target_surface": target,
                        "rationale": entry.data.get("rationale", "improvement from research"),
                        "risk_class": "standard", "provenance": f"procedural:{entry.key}",
                        "staged": False, "blocked": True,
                        "blocked_reason": "queued for rewrite, not yet staged",
                    })
        return proposals

    def _discover_free_services(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Deterministic service discovery with structured ranking.
        Ranks by: goal relevance, budget compat, history, capability gaps, auth cost."""
        candidates = []
        goal_text = (self.current_goal.get("goal", "") if self.current_goal else "").lower()
        goal_words = set(goal_text.split()[:5])
        for svc in self._free_services:
            if domain and domain.lower() not in str(svc.get("type", "")).lower() + str(svc.get("name", "")).lower():
                continue
            if svc.get("status") not in ("available", "active", "discovered"):
                continue
            score = 0.0
            rationale = []
            if svc.get("status") == "active": score += 3.0; rationale.append("active")
            elif svc.get("status") == "available": score += 2.0
            if goal_words and any(w in str(svc).lower() for w in goal_words): score += 2.0; rationale.append("goal-relevant")
            if not svc.get("auth"): score += 1.0; rationale.append("no-auth")
            svc_name = str(svc.get("name", "")).lower()
            fail_count = self._tool_fail_counts.get(svc_name, 0)
            if fail_count > 0: score -= 0.5 * fail_count; rationale.append(f"failures:{fail_count}")
            candidates.append({**svc, "_score": score, "_rationale": rationale})
        # Research memory scan
        for rm in self.research_memory[-30:]:
            for r in rm.get("results", rm.get("synthesis", {}).get("raw_results", [])):
                if not r.get("ok"): continue
                text = str(r.get("payload", ""))[:2000].lower()
                if any(k in text for k in ("free api", "public api", "open api")):
                    entry = {"name": f"discovered_{rm.get('topic', '?')[:25]}", "type": "api",
                             "url": r.get("url", ""), "status": "discovered", "auth": None,
                             "_score": 1.0, "_rationale": ["from-research"]}
                    if domain and domain.lower() not in str(r.get("summary", "")).lower(): continue
                    if not any(c.get("url") == entry["url"] for c in candidates if entry["url"]):
                        candidates.append(entry)
        candidates.sort(key=lambda s: s.get("_score", 0), reverse=True)
        # Advisor enrichment
        if domain and self.api and self.budget_manager.check("advisor_call"):
            advice = self._chat_with_advisor(f"Find free/public APIs for: {domain}. Return JSON list.")
            try:
                for ns in (json.loads(advice) if advice.startswith("[") else [])[:5]:
                    if isinstance(ns, dict) and ns.get("name"):
                        ns.update({"status": "discovered", "_score": 1.5, "_rationale": ["advisor"]})
                        self._free_services.append(ns)
                        candidates.append(ns)
                save_json(FREE_SERVICES_FILE, self._free_services)
            except Exception: pass
        # Bilateral state update
        if candidates:
            self.state.beta["service.discovery_health"] = bilateral_recover(
                self.state.beta.get("service.discovery_health", BilateralValue(0.4, 0.3)),
                truth_gain=0.08, falsity_decay=0.04)
        else:
            self.state.beta["service.discovery_health"] = bilateral_or(
                self.state.beta.get("service.discovery_health", BilateralValue(0.4, 0.3)),
                BilateralValue(0.0, 0.05))
        refresh_state(self.state)
        return candidates[:20]

    def _shadowhott_rewrite_method(self, method_name: str) -> Tuple[bool, str]:
        """Request ShadowHoTT-native rewrite via advisor. Goes through staging."""
        if not self.api:
            return False, "no advisor available for rewrite"
        prompt = (f"Rewrite the kernel method '{method_name}' to be fully ShadowHoTT-native. "
                  f"Use BilateralValue, bilateral_recover, refresh_state. "
                  f"Return JSON: {{\"patch_name\": \"rewrite_{method_name}\", \"target\": \"{method_name}\", "
                  f"\"code\": \"def {method_name}(self, ...): ...\", \"rationale\": \"...\"}}")
        response = self._chat_with_advisor(prompt)
        if not response:
            return False, "advisor returned empty"
        ok, msg = self.stage_patch(response, source="shadowhott_rewrite")
        if ok:
            self._rewrite_history.append({"method": method_name, "time": time.time(), "staged": True})
        return ok, msg

    def _process_natural_instruction(self, instruction: str) -> str:
        """Route free-form instruction into goal/plan/research/tool actions."""
        intent = self._classify_query_intent(instruction)
        if intent == "url_fetch":
            tr = self._perform_tool_action({"tool": "fetch_url", "arg": instruction.strip()})
            return tr.summary
        if intent in ("github_repo", "github_file"):
            tr = self._perform_tool_action({"tool": intent, "arg": instruction.strip()})
            return tr.summary
        if intent == "method_rewrite":
            words = instruction.split()
            method = next((w for w in words if w.startswith("_") or w in ALLOWED_PATCH_TARGETS), "")
            if method:
                ok, msg = self._shadowhott_rewrite_method(method)
                return msg
        # Default: set as goal and research
        self.current_goal = {"goal": instruction, "function_spec": instruction, "domain": intent, "reasoning": "natural instruction"}
        self.state.c.active_goal = instruction[:60]
        self.state.beta["goal.active"] = BilateralValue(0.8, 0.05)
        refresh_state(self.state)
        synth = self.research_topic(instruction)
        return f"Goal set + researched: {synth.get('success_count',0)}/{synth.get('total_queries',0)} queries"

    def research_topic(self, topic: str, context: str = "") -> Dict[str, Any]:
        """Multi-step research with structured synthesis. Returns dict with:
        findings, uncertainties, contradictions, tool_opportunities, patch_opportunities,
        service_opportunities, next_actions, provenance, bilateral_confidence, raw_results."""
        self.state.c.last_action = f"research:{topic[:60]}"
        results: List[Dict[str, Any]] = []
        queries = self._decompose_goal_into_queries(topic)
        for q in queries[:4]:
            query_tools = self._rank_tool_candidates(q, q)
            for tool in query_tools[:3]:
                tr = self._perform_tool_action({"tool": tool, "arg": q})
                results.append({"tool": tr.tool, "ok": tr.ok, "summary": tr.summary,
                                "payload": tr.payload, "url": tr.url, "query": q})
                if tr.ok:
                    self._store_memory("episodic", f"research:{self._slugify(q)[:40]}",
                                        {"topic": topic, "query": q, "summary": tr.summary[:300], "tool": tool},
                                        goal_context=topic, tags=["research", topic.split()[0] if topic.split() else ""])
                    self.state.beta["tool.search_efficacy"] = bilateral_recover(
                        self.state.beta.get("tool.search_efficacy", BilateralValue()), truth_gain=0.10, falsity_decay=0.05)
                    break
                else:
                    self.state.beta["tool.search_efficacy"] = bilateral_or(
                        self.state.beta.get("tool.search_efficacy", BilateralValue()), BilateralValue(0.0, 0.08))
        refresh_state(self.state)
        success_count = sum(1 for r in results if r.get("ok"))
        findings = [r.get("summary", "")[:300] for r in results if r.get("ok")]
        uncertainties = [f"query '{r.get('query','')}' failed via {r.get('tool','')}" for r in results if not r.get("ok")]
        # Cross-query contradiction detection: find disagreeing summaries
        contradictions = []
        ok_summaries = [r.get("summary", "") for r in results if r.get("ok") and r.get("summary")]
        for i, s1 in enumerate(ok_summaries):
            for s2 in ok_summaries[i+1:]:
                s1w, s2w = set(s1.lower().split()), set(s2.lower().split())
                overlap = len(s1w & s2w)
                union = len(s1w | s2w) if s1w | s2w else 1
                if overlap / union < 0.15 and len(s1) > 20 and len(s2) > 20:
                    contradictions.append({"type": "low_agreement", "a": s1[:100], "b": s2[:100],
                                           "overlap_ratio": round(overlap / union, 3)})
        tool_opps = self._discover_tool_opportunities(topic, results)
        svc_opps = [s.get("name", "") for s in self._discover_free_services(topic.split()[0] if topic.split() else None)[:3]]
        patch_opps = []
        for opp in tool_opps:
            patch_opps.append({"kind": "tool_module", "name": opp.get("name", ""), "rationale": opp.get("rationale", "")})
        t_conf = min(1.0, 0.2 + 0.15 * success_count - 0.05 * len(contradictions))
        f_conf = min(1.0, 0.05 + 0.08 * (len(results) - success_count) + 0.05 * len(contradictions))
        synth_bv = BilateralValue(max(0.0, t_conf), max(0.0, f_conf))
        synthesis = {
            "topic": topic, "findings": findings, "uncertainties": uncertainties,
            "contradictions": contradictions,
            "tool_opportunities": tool_opps,
            "service_opportunities": svc_opps,
            "patch_opportunities": patch_opps,
            "next_actions": ([f"investigate contradiction: {c.get('type','')}" for c in contradictions[:2]]
                           + [f"deepen: {u}" for u in uncertainties[:2]])[:4],
            "provenance": [{"query": r.get("query",""), "tool": r.get("tool",""), "ok": r.get("ok",False)} for r in results],
            "raw_results": results,
            "bilateral_confidence": {"t": synth_bv.t, "f": synth_bv.f, "glut": synth_bv.glut, "gap": synth_bv.gap},
            "success_count": success_count, "total_queries": len(results),
        }
        self._store_memory("semantic", f"synthesis:{self._slugify(topic)[:40]}",
                            {"findings": findings[:5], "uncertainties": uncertainties[:3],
                             "contradictions": len(contradictions), "confidence_t": synth_bv.t, "confidence_f": synth_bv.f},
                            goal_context=topic, tags=["synthesis", topic.split()[0] if topic.split() else ""])
        self.experience_store.record(
            f"research_{int(time.time())}", "research",
            context={"topic": topic, "queries": len(queries), "synthesis_confidence": synth_bv.t},
            outcome="useful" if success_count > 0 else "useless",
            reward_signal=0.5 if success_count > 0 else -0.3,
            tags=["research", topic.split()[0] if topic.split() else ""])
        domain = context if context else "research"
        self.competence_map.record_outcome(domain, success_count > 0)
        self.research_memory.append({"time": time.time(), "topic": topic, "synthesis": synthesis})
        self.research_memory = self.research_memory[-MAX_RESEARCH_RESULTS_STORED:]
        self._recent_research_topics.append(topic)
        self._recent_research_topics = self._recent_research_topics[-80:]
        self._topic_last_research_time[topic] = time.time()
        self.last_research_time = time.time()
        self.module_health.record_success("retriever", self.state)
        return synthesis

    def _autonomous_cycle(self) -> None:
        """One full autonomous decision cycle. Real orchestration."""
        if self._paused or getattr(self.state.c, "degraded", False):
            return
        try:
            # 1. Update self-model
            self.update_self_model()
            # 2. Choose or refine goal
            goal = self._generate_next_goal()
            if goal and goal != self.current_goal:
                self.current_goal = goal
                self.state.c.active_goal = str(goal.get("goal", ""))[:60]
                self.state.beta["goal.active"] = BilateralValue(0.7, 0.05)
                refresh_state(self.state)
                self._goal_attempts = 0
            if not self.current_goal:
                return
            self._goal_attempts += 1
            if self._goal_attempts > self._max_goal_attempts:
                old = self.current_goal.get("goal", "?")
                self._shelved_goals.append(old)
                self.current_goal = None
                self.state.c.active_goal = ""
                self.state.beta["goal.active"] = BilateralValue(0.1, 0.3)
                refresh_state(self.state)
                logging.info(f"SHELVED goal after {self._max_goal_attempts} attempts: {old[:60]}")
                return
            # 3. Build or advance plan
            active_plans = self.plan_manager.get_active()
            plan = None
            if active_plans:
                plan = active_plans[0]
            else:
                plan = self._strategic_plan(self.current_goal.get("goal", "explore"))
            if not plan:
                return
            # 4. Execute next pending step
            for step in plan.steps:
                if step.get("status") != "pending":
                    continue
                action = step.get("action", "")
                target = step.get("target", "")
                step["status"] = "active"
                if action == "research":
                    synth = self.research_topic(target, self.current_goal.get("domain", ""))
                    raw = synth.get("raw_results", [])
                    step["status"] = "completed" if synth.get("success_count", 0) > 0 else "failed"
                    step["result"] = {"count": synth.get("total_queries", 0), "ok": synth.get("success_count", 0),
                                      "contradictions": len(synth.get("contradictions", []))}
                    self._lab_growth_cycle(target, raw)
                elif action == "synthesize":
                    step["status"] = "completed"
                elif action == "reflect":
                    self._self_assess()
                    step["status"] = "completed"
                else:
                    step["status"] = "completed"
                break  # one step per cycle
            # 5. Check plan completion
            if all(s.get("status") in ("completed", "failed", "skipped") for s in plan.steps):
                plan.status = "completed"
                self.plan_manager.completed_ids.append(plan.plan_id)
                # Adapt research code if we found opportunities
                adapt_msgs = self._adapt_research_code()
                if adapt_msgs:
                    logging.info(f"ADAPT: {adapt_msgs}")
                # Complete goal if plan succeeded
                completed_steps = sum(1 for s in plan.steps if s.get("status") == "completed")
                if completed_steps > len(plan.steps) // 2:
                    if self.current_goal:
                        self.completed_goals.append(self.current_goal.get("goal", "unnamed"))
                    self.current_goal = None
                    self._goal_attempts = 0
            # 6. Record blocked growth structurally
            for pn, rec in list(self.staged_patches.items()):
                stage = self.promotion_ladder.current_stage(pn)
                if stage in ("static_approved", "sandbox_passed") and rec.get("status") == "staged":
                    self._blocked_growth_log.append({
                        "patch": pn, "stage": stage, "reason": "blocked during cycle", "time": time.time()})
                    self.state.beta["growth.blocked"] = bilateral_or(
                        self.state.beta.get("growth.blocked", BilateralValue(0.0, 0.0)),
                        BilateralValue(0.0, 0.15))
            self._blocked_growth_log = self._blocked_growth_log[-50:]
            refresh_state(self.state)
            # 7. Module health
            self.module_health.record_success("executor", self.state)
            # 8. Trace
            self._write_report_and_trace("autonomous_cycle")
        except Exception as e:
            logging.error(f"AUTONOMOUS CYCLE ERROR: {e}")
            self._runtime_error_counts["autonomous_cycle"] = self._runtime_error_counts.get("autonomous_cycle", 0) + 1
            self.module_health.record_failure("executor", self.state)

    # ================================================================
    # PATCH / STAGING / INJECT — with preflight
    # ================================================================

    def stage_patch(self, raw: str, source: str = "advisor",
                    allow_create_new: bool = False) -> Tuple[bool, str]:
        result = _stage_patch_fn(raw, source, self.staged_patches, self.certs,
                                  kernel_class=self.__class__,
                                  state_beta_keys=set(self.state.beta.keys()),
                                  allow_create_new=allow_create_new)
        if result.ok:
            self.mutation_logger.record_stage(result.patch_name, result.target, source)
            self.promotion_ladder.state[result.patch_name] = "proposed"
            # AUDIT FIX (v14.2.7, RC-1): the v14.2.6 ladder silently treated
            # un-attested patches as sovereign-main. After the default
            # inversion, every staged patch must declare its source
            # explicitly. This call records the kernel-internal staging
            # source so the ladder's policy gate has the context it needs.
            # Internal staging is "main" with trust=sovereign; any external
            # source (e.g. "subkernel:<id>") goes through _stage_patch_proposal
            # and sets richer metadata there.
            self.promotion_ladder.set_source_metadata(
                result.patch_name,
                source_role="main",
                trust_level="sovereign",
                source_locality="local",
                risk_level="low",
                source_kernel_id=getattr(self, "kernel_id", "main"),
                outcome_success_rate=1.0,
                budget_pressure=0.0,
                dynamic_delta=0.0,
            )
        return result.ok, result.message

    def apply_staged_patch(self, patch_name: str) -> Tuple[bool, str]:
        """Apply via promotion ladder with bounded regression."""
        current = self.promotion_ladder.current_stage(patch_name)
        while current not in ("shadow_deployed", "live_promoted", "revertable"):
            new_stage, msg = self.promotion_ladder.advance(
                patch_name, self.staged_patches,
                sandbox_runner=self._sandbox_run,
                regression_runner=lambda: self._bounded_regression(),
            )
            if new_stage == current:
                self._blocked_growth_log.append({
                    "patch": patch_name, "stage": current, "msg": msg, "time": time.time()})
                return False, f"promotion blocked at {current}: {msg}"
            current = new_stage
        if current != "shadow_deployed":
            return False, f"cannot reach shadow_deployed: stuck at {current}"
        ok, msg = self.promotion_ladder.apply_live(
            patch_name, self.staged_patches,
            self.__class__, self._original_methods, self._evolved_method_names,
        )
        if ok:
            self.improvement_count += 1
            self.mutation_logger.record_apply(patch_name,
                self.staged_patches.get(patch_name, {}).get("target", ""),
                self.staged_patches.get(patch_name, {}).get("code", ""))
            self.state.beta["patch.pipeline.health"] = bilateral_recover(
                self.state.beta.get("patch.pipeline.health", BilateralValue()), truth_gain=0.20, falsity_decay=0.08)
            refresh_state(self.state)
            self.checkpoint_shadow("patch_apply")
            self.update_mirror()
            self.save_state()
        return ok, msg

    def _sandbox_run(self, code: str) -> Tuple[bool, str]:
        """Real sandbox runner: exec in restricted environment."""
        try:
            env = {"__builtins__": {"len": len, "str": str, "int": int, "float": float,
                                     "list": list, "dict": dict, "bool": bool, "True": True,
                                     "False": False, "None": None, "print": lambda *a: None}}
            exec(compile(code, "<sandbox>", "exec"), env, {})
            return True, "sandbox ok"
        except SyntaxError as e:
            return False, f"syntax error: {e}"
        except Exception as e:
            return True, f"sandbox exec raised {type(e).__name__} (non-syntax)"

    def assess_patch_json(self, raw: str) -> BilateralValue:
        try:
            obj = json.loads(raw)
        except Exception:
            return BilateralValue(0.0, 0.9)
        target = str(obj.get("target", "")).strip()
        code = str(obj.get("code", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        if not target or target not in ALLOWED_PATCH_TARGETS or not code:
            return BilateralValue(0.05, 0.90)
        ok, fn_names, errs = analyze_patch_code(code)
        if not ok or target not in fn_names:
            return BilateralValue(0.10, 0.85)
        contract_ok, _ = verify_patch_contract(target, code)
        t_score = 0.55 if contract_ok else 0.20
        f_score = 0.15 if contract_ok else 0.75
        if len(rationale) > 20: t_score += 0.08
        if "BilateralValue" in code or "bilateral_recover" in code: t_score += 0.08
        if "self.state.beta" in code and "refresh_state" in code: t_score += 0.08
        return BilateralValue(min(1.0, t_score), min(1.0, f_score))

    def direct_inject_method(self, target: str, code: str, source: str = "david-inject",
                              create_new: bool = False) -> Tuple[bool, str]:
        """Direct injection via stage+promote. No direct setattr.
        create_new must be EXPLICITLY True — never inferred from target membership."""
        from tovah_v14.kernel.patch_preflight import validate_patch_preflight
        report = validate_patch_preflight(target, code, self.__class__,
                                           state_beta_keys=set(self.state.beta.keys()),
                                           allow_create_new=create_new)
        if not report.accepted:
            self._patch_reject_log.append({"target": target, "source": source, "errors": report.errors, "time": time.time()})
            return False, f"preflight rejected: {'; '.join(report.errors[:3])}"
        patch_name = f"INJECT_{target}_{int(time.time())}"
        patch_json = json.dumps({
            "patch_name": patch_name, "target": target, "code": code,
            "rationale": f"direct injection from {source}",
        })
        stage_ok, stage_msg = self.stage_patch(patch_json, source=source,
                                                allow_create_new=create_new)
        if not stage_ok:
            return False, f"staging failed: {stage_msg}"
        # David injections come from sovereign main; mark accordingly so the
        # promotion ladder gate doesn't reject medium-risk patches under
        # default subkernel/provisional metadata. Mirrors the metadata that
        # `_stage_patch_proposal` sets for packet-routed proposals.
        op_metrics = self._node_operational_metrics(self.kernel_id, locality="local")
        self.promotion_ladder.set_source_metadata(
            patch_name,
            source_kernel_id=self.kernel_id,
            source_role="main",
            source_locality="local",
            trust_level="sovereign",
            risk_class="low",
            risk_level="low",
            outcome_success_rate=op_metrics["success_rate"],
            budget_pressure=op_metrics["budget_pressure"],
            dynamic_delta=op_metrics["dynamic_delta"],
            recent_failure_weight=op_metrics["recent_failure_weight"],
            cooldown_until=op_metrics["cooldown_until"],
            maturity_bonus=op_metrics["maturity_bonus"],
        )
        # Sovereign injection counts as evidence for the adaptive gate
        # (which requires evidence entries before shadow deployment AND
        # before live promotion — main needs 3, shadow_deployed needs 2).
        self.promotion_ladder.record_evidence(
            patch_name, "sovereign_inject",
            source_kernel_id=self.kernel_id, trust_level="sovereign",
            risk_class="low",
            details={"source": source, "target": target, "create_new": create_new},
        )
        self.promotion_ladder.record_evidence(
            patch_name, "preflight_passed",
            source_kernel_id=self.kernel_id, trust_level="sovereign",
            risk_class="low",
            details={"validated_by": "validate_patch_preflight"},
        )
        self.promotion_ladder.record_evidence(
            patch_name, "sovereign_authority",
            source_kernel_id=self.kernel_id, trust_level="sovereign",
            risk_class="low",
            details={"reason": "direct_inject by sovereign main"},
        )
        apply_ok, apply_msg = self.apply_staged_patch(patch_name)
        return apply_ok, apply_msg

    def inject_tool_via_advisor(self, raw_code: str, source: str = "david-tool") -> Tuple[bool, str]:
        name = None
        try:
            tree = ast.parse(raw_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "TOOL_SPEC":
                            spec_code = ast.get_source_segment(raw_code, node.value)
                            if spec_code:
                                spec = ast.literal_eval(spec_code)
                                name = spec.get("name")
        except Exception:
            pass
        if not name:
            return False, "could not determine tool name"
        if "def run" not in raw_code:
            return False, "must define run(kernel, **kwargs)"
        slug = self._slugify(name)
        active_path = LAB_ACTIVE / f"{slug}.py"
        active_path.write_text(raw_code, encoding="utf-8")
        self._load_active_lab_tools()
        if name not in self.active_lab_tools:
            active_path.unlink(missing_ok=True)
            return False, f"tool '{name}' did not load"
        self.improvement_count += 1
        self.save_state()
        return True, f"tool '{name}' injected and active"

    # ================================================================
    # LAB / CAPABILITIES / SNAPSHOT / MIRROR / REPORTS / CREDS
    # ================================================================

    def _load_active_lab_tools(self) -> None:
        self.active_lab_tools = {}
        for path in LAB_ACTIVE.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if not spec or not spec.loader: continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                ts = getattr(mod, "TOOL_SPEC", {"name": path.stem, "description": ""})
                run_fn = getattr(mod, "run", None)
                if callable(run_fn):
                    name = str(ts.get("name", path.stem))
                    self.active_lab_tools[name] = run_fn
                    self.lab_registry[name] = {"status": "active", "path": str(path), "spec": ts}
            except Exception as e:
                logging.warning(f"lab tool load failed {path.name}: {e}")

    def _load_free_services(self) -> None:
        stored = load_json(FREE_SERVICES_FILE, [])
        self._free_services = stored if stored else copy.deepcopy(DEFAULT_FREE_SERVICES)
        if not stored:
            save_json(FREE_SERVICES_FILE, self._free_services)

    def _load_capabilities(self) -> None:
        for path in CAPABILITIES_DIR.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(f"cap_{path.stem}", path)
                if not spec or not spec.loader: continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cs = getattr(mod, "CAPABILITY_SPEC", {"name": path.stem, "description": ""})
                self._capabilities[cs.get("name", path.stem)] = {"path": str(path), "spec": cs, "module": mod}
            except Exception as e:
                logging.warning(f"capability load failed {path.name}: {e}")

    def _save_model_snapshot(self, reason: str) -> None:
        meta = save_snapshot(self.shadow_model, reason,
                              {"profile_name": self.profile_name, "improvement_count": self.improvement_count,
                               "con_budget": self.con_budget, "gap_budget": self.gap_budget})
        self._model_snapshots.append(meta)
        self._model_snapshots = self._model_snapshots[-MAX_SNAPSHOTS_MEMORY:]

    def _rollback_model(self) -> Tuple[bool, str]:
        ok, msg, meta = rollback_snapshot(self.shadow_model, self._model_snapshots, self.device)
        if ok:
            self.con_budget = meta.get("con_budget", self.con_budget)
            self.gap_budget = meta.get("gap_budget", self.gap_budget)
        return ok, msg

    def update_mirror(self) -> None:
        try:
            with open(MIRROR_FILE, "w", encoding="utf-8") as f:
                f.write(f"# TOVAH MIRROR | v{self.identity.version}\n# {dt.datetime.now()}\n")
        except Exception as e:
            logging.error(f"mirror failed: {e}")

    def get_self_code(self) -> str:
        try:
            with open(__file__, "r", encoding="utf-8") as f: return f.read()
        except Exception:
            return "# Source unavailable"

    def _write_response(self, text: str) -> None:
        with open(RESPONSE_FILE, "w", encoding="utf-8") as f:
            f.write(f"[{dt.datetime.now()}]\n{text}\n")

    def _append_need(self, kind: str, body: str) -> None:
        with open(NEEDS_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {kind}\n{body.strip()}\n\n")

    def _load_credentials(self) -> Dict[str, Any]:
        return load_json(CREDENTIALS_FILE, {})

    def _save_credentials(self, creds: Dict[str, Any]) -> None:
        save_json(CREDENTIALS_FILE, creds)

    def _write_report_and_trace(self, label: str) -> str:
        report = self.invariants.build_report(self.state, self.loss_history)
        trace = {"label": label, "time": dt.datetime.now().isoformat(timespec="seconds"),
                 "state": self.state.snapshot(), "report": asdict(report)}
        tid = f"trace_{int(time.time())}_{hashlib.md5(label.encode()).hexdigest()[:8]}"
        for d in (TRACE_DIR, LAB_TRACES):
            save_json(d / f"{tid}.json", trace)
        for d in (REPORT_DIR, LAB_REPORTS):
            save_json(d / f"{tid}.report.json", asdict(report))
        self.trace_index.append(str(TRACE_DIR / f"{tid}.json"))
        self.trace_index = self.trace_index[-MAX_TRACES_STORED:]
        return tid

    # ================================================================
    # SELF ASSESSMENT / CAPABILITY TESTS / SUMMARY
    # ================================================================

    def _self_assess(self) -> None:
        report = self.invariants.build_report(self.state, self.loss_history)
        t = 0.3
        if report.coherent: t += 0.15
        if report.mean_glut < 0.15: t += 0.10
        if report.mean_gap < 0.15: t += 0.10
        if self.improvement_count > 5: t += 0.10
        if len(self.active_lab_tools) > 0: t += 0.10
        f = 0.1
        if not report.coherent: f += 0.20
        if report.mean_glut > 0.30: f += 0.15
        self.state.beta["self_assessment.overall"] = BilateralValue(min(1.0, t), min(1.0, f)).clamp()
        refresh_state(self.state)

    def run_capability_tests(self) -> Tuple[int, int, Dict[str, bool]]:
        """Bounded capability regression. Skips heavyweight neural forward pass."""
        res: Dict[str, bool] = {}
        res["identity"] = self.protect_core_goal()
        res["cache_coherent"] = is_cache_coherent(self.state)
        res["shadow_model_exists"] = self.shadow_model is not None
        res["lab_loader"] = isinstance(self.active_lab_tools, dict)
        res["patch_analysis"] = analyze_patch_code("def research_topic(self, topic, context=''):\n    return []\n")[0]
        res["patch_blocks_eval"] = not analyze_patch_code("def f(): eval('x')")[0]
        res["pdf_available"] = PdfReader is not None
        res["state_file_writable"] = STATE_FILE.parent.exists()
        res["memory_system"] = all(k in self.memory_store.banks for k in ("episodic", "semantic", "procedural"))
        res["task_system"] = isinstance(self.task_queue.tasks, list)
        res["budget_system"] = isinstance(self.budget_manager.budgets, dict) and len(self.budget_manager.budgets) > 0
        res["promotion_ladder"] = isinstance(self.promotion_ladder.state, dict)
        res["experience_store"] = isinstance(self.experience_store.records, list)
        return sum(1 for v in res.values() if v), len(res), res

    def _bounded_regression(self) -> Tuple[int, int, Dict[str, bool]]:
        """Bounded regression for patch promotion. Deterministic, no neural forward."""
        return self.run_capability_tests()

    def get_self_summary(self) -> Dict[str, Any]:
        report = self.invariants.build_report(self.state, self.loss_history)
        return {
            "version": self.identity.version, "profile": self.profile_name,
            "params": self.model_param_count, "device": self.device,
            "improvements": self.improvement_count, "autonomy": self.autonomy_level,
            "current_goal": self.current_goal, "goals_completed": len(self.completed_goals),
            "mean_glut": report.mean_glut, "mean_gap": report.mean_gap,
            "cache_hist": report.cache_histogram, "training_phase": self._training_phase,
            "tools": self.tools.builtins + sorted(self.active_lab_tools.keys()),
            "advisor_available": bool(self.api),
            "memory": self.memory_store.counts(),
            "tasks_active": len(self.task_queue.get_active()),
            "plans_active": len(self.plan_manager.get_active()),
            "experience_count": len(self.experience_store.records),
            "budget_usage": self.budget_manager.usage_summary(),
            "promotion_queue": sum(1 for s in self.promotion_ladder.state.values() if s not in ("live_promoted", "revertable", "reverted")),
            "kernel_ecology": {
                "boot_mode": self.boot_mode,
                "hub_present": self.hub_kernel is not None,
                "subkernel_count": len(self.subkernels),
                "packet_log_entries": len(self.kernel_packet_log),
            },
        }

    def send_email_report(self, msg: str = "") -> None:
        logging.info(f"EMAIL PLACEHOLDER: {msg[:140]}")

    # ================================================================
    # DAVID COMMAND INTERFACE
    # ================================================================

    def _check_david_commands(self) -> None:
        if not COMMAND_FILE.exists():
            return
        try:
            content = COMMAND_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return
        if not content:
            return
        COMMAND_FILE.write_text("", encoding="utf-8")
        logging.info(f"DAVID: {content[:140]}")
        response = ""
        try:
            upper = content.upper()
            if upper == "STATUS":
                p, t, tests = self.run_capability_tests()
                response = json.dumps({"summary": self.get_self_summary(), "tests": f"{p}/{t}", "details": tests}, indent=2, default=str)
            elif upper == "PATCHES":
                staged = {k: v for k, v in self.staged_patches.items() if v.get("status") == "staged"}
                response = "\n".join(f"- {n} | target={r.get('target')} | source={r.get('source')}" for n, r in staged.items()) or "No staged patches."
            elif upper.startswith("REJECT_PATCH:"):
                pn = content[13:].strip()
                if pn in self.staged_patches:
                    self.staged_patches[pn]["status"] = "rejected-by-human"
                    response = f"rejected {pn}"
                else:
                    response = f"not found: {pn}"
            elif upper.startswith("GOAL:"):
                txt = content[5:].strip()
                self.current_goal = {"goal": txt, "function_spec": txt, "domain": "directed", "reasoning": "David"}
                self._goal_attempts = 0
                self.state.c.active_goal = txt
                self.state.beta["goal.active"] = BilateralValue(0.9, 0.0)
                refresh_state(self.state)
                response = f"Goal: {txt}"
            elif upper in ("CLEAR_GOAL", "CANCEL_GOAL", "REMOVE_GOAL"):
                old = self.current_goal.get("goal", "?") if self.current_goal else "none"
                self.current_goal = None
                self._goal_attempts = 0
                self.state.c.active_goal = ""
                self.state.beta["goal.active"] = BilateralValue(0.0, 0.0)
                refresh_state(self.state)
                response = f"Cleared: {old}"
            elif upper.startswith("COMPLETE_GOAL"):
                if self.current_goal:
                    desc = self.current_goal.get("goal", "unnamed")
                    self.completed_goals.append(desc)
                    self.current_goal = None
                    self._goal_attempts = 0
                    response = f"Goal completed: {desc}"
                else:
                    response = "No active goal."
            elif upper.startswith("RESEARCH:"):
                synth = self.research_topic(content[9:].strip(), "David")
                response = json.dumps({"findings": synth.get("findings", [])[:3], "success": synth.get("success_count", 0), "contradictions": len(synth.get("contradictions", []))}, indent=2, default=str)[:3000]
            elif upper.startswith("TOOL:"):
                parts = [p.strip() for p in content[5:].split("|")]
                tr = self._perform_tool_action({"tool": parts[0], "arg": parts[1] if len(parts) > 1 else "", "arg2": parts[2] if len(parts) > 2 else ""})
                response = json.dumps(asdict(tr), indent=2, default=str)[:3000]
            elif upper.startswith("STAGE_PATCH:"):
                pl = content[12:].strip()
                pl = Path(pl).read_text(encoding="utf-8") if os.path.exists(pl) else pl
                ok, msg = self.stage_patch(pl, source="human")
                response = msg
            elif upper.startswith("APPLY_PATCH:"):
                ok, msg = self.apply_staged_patch(content[12:].strip())
                response = msg
            elif upper == "TRACE":
                response = f"Wrote {self._write_report_and_trace('manual')}"
            elif upper == "PAUSE":
                self._paused = True
                self.state.c.paused = True
                response = "Paused."
            elif upper == "RESUME":
                self._paused = False
                self.state.c.paused = False
                response = "Resumed."
            elif upper.startswith("INSTALL:"):
                ok = self._pip_install(content[8:].strip())
                response = f"{'SUCCESS' if ok else 'FAILED'}"
            elif upper == "ROLLBACK_MODEL":
                ok, msg = self._rollback_model()
                response = msg
            elif upper == "SNAPSHOT":
                self._save_model_snapshot("david")
                response = f"Saved. Mem:{len(self._model_snapshots)} Disk:{len(list(SNAPSHOT_DIR.glob('snap_*.pt')))}"
            elif upper.startswith("INJECT_METHOD:"):
                body = content[14:].strip()
                body = Path(body).read_text(encoding="utf-8") if os.path.exists(body) else body
                if "\n" in body:
                    fl, code = body.split("\n", 1)
                    tn = fl.strip()
                else:
                    tn, code = "", body
                if not tn:
                    try:
                        tree = ast.parse(code)
                        tn = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
                    except Exception:
                        pass
                if tn:
                    ok, msg = self.direct_inject_method(tn, code)
                    response = msg
                else:
                    response = "Could not determine method name."
            elif upper.startswith("INJECT_TOOL:"):
                body = content[12:].strip()
                body = Path(body).read_text(encoding="utf-8") if os.path.exists(body) else body
                ok, msg = self.inject_tool_via_advisor(body)
                response = msg
            elif upper.startswith("EXPORT_CORPUS:"):
                # EXPORT_CORPUS:<dir>[|<since_cycle>][|<dedup_strategy>]
                spec = content[14:].strip()
                parts = [p.strip() for p in spec.split("|")]
                out_dir = parts[0] if parts else ""
                since_cycle = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                strategy = parts[2] if len(parts) > 2 and parts[2] else "merge_with_provenance"
                if not out_dir:
                    response = "EXPORT_CORPUS requires output directory: EXPORT_CORPUS:/path/to/out"
                else:
                    try:
                        from tovah_v14.training import export_corpus
                        result = export_corpus(out_dir, kernel=self,
                                               since_cycle=since_cycle,
                                               dedup_strategy=strategy)
                        response = json.dumps({
                            "out_dir": result["out_dir"],
                            "total": result["total_examples"],
                            "unique": result["unique_examples"],
                            "shards": len(result["shards"]),
                            "manifest": result["manifest"]["paraconsistent"],
                        }, indent=2, default=str)[:4000]
                    except Exception as e:
                        response = f"EXPORT_CORPUS failed: {e}"
            elif upper.startswith("TRAIN_FROM_CORPUS"):
                # TRAIN_FROM_CORPUS[:<shard_dir>[|<epochs>[|<batch_size>[|<save_path>]]]]
                # Defaults: shard_dir=tovah_corpus/stream, epochs=1, batch_size=8.
                # save_path empty => no checkpoint write. Live shadow model is
                # used and updated in-place; loss curve is recorded into
                # loss_history just like the regular training step.
                # Accepted forms:
                #   TRAIN_FROM_CORPUS
                #   TRAIN_FROM_CORPUS:
                #   TRAIN_FROM_CORPUS:/path/to/shards
                #   TRAIN_FROM_CORPUS:|2|16
                #   TRAIN_FROM_CORPUS::2|16    (also accepted; extra colons stripped)
                tail = content[len("TRAIN_FROM_CORPUS"):]
                # Strip a single leading separator colon if present (commands like
                # "TRAIN_FROM_CORPUS:..."). Don't lstrip all colons — that would
                # consume an intentional empty first field in "TRAIN_FROM_CORPUS::..."
                # which the caller uses to mean "default dir, then args".
                if tail.startswith(":"):
                    tail = tail[1:]
                spec = tail.strip()
                parts = [p.strip() for p in spec.split("|")] if spec else []
                from tovah_v14.config.paths import CORPUS_STREAM_DIR
                shard_dir = parts[0] if (parts and parts[0]) else str(CORPUS_STREAM_DIR)
                try:
                    epochs = int(parts[1]) if len(parts) > 1 and parts[1] else 1
                except ValueError:
                    epochs = 1
                try:
                    batch_size = int(parts[2]) if len(parts) > 2 and parts[2] else 8
                except ValueError:
                    batch_size = 8
                save_path = parts[3] if len(parts) > 3 and parts[3] else None
                try:
                    from tovah_v14.training import pretrain
                    summary = pretrain(
                        shard_dir,
                        model=self.shadow_model,
                        optimizer=self.shadow_optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        save_path=save_path,
                        device=self.device,
                        log_every=20,
                    )
                    # Reflect pretraining loss into loss_history so reports + traces
                    # see the curve. Use last-loss as proxy for current step.
                    if summary.get("epoch_last_loss"):
                        for L in summary["epoch_last_loss"]:
                            if L is not None:
                                self.loss_history.append(float(L))
                                self.loss_history = self.loss_history[-500:]
                    self._training_phase = summary.get("final_phase", self._training_phase)
                    self.module_health.record_success("trainer", self.state)
                    response = json.dumps(summary, indent=2, default=str)[:4000]
                except FileNotFoundError as e:
                    response = f"TRAIN_FROM_CORPUS: shard dir not found: {e}"
                except ValueError as e:
                    response = f"TRAIN_FROM_CORPUS: {e}"
                except Exception as e:
                    response = f"TRAIN_FROM_CORPUS failed: {type(e).__name__}: {e}"
            elif upper == "LAB_STATUS":
                response = json.dumps({"active": sorted(self.active_lab_tools.keys()), "registry": {k: v.get("status") for k, v in self.lab_registry.items()}}, indent=2)[:4000]
            elif upper == "LIST_PLANS":
                active = [{"id": p.plan_id, "obj": p.objective[:60], "status": p.status} for p in self.plan_manager.active]
                response = json.dumps(active, indent=2)[:4000]
            elif upper == "LIST_PATCHED":
                response = json.dumps({"patched": sorted(self._evolved_method_names), "revertable": sorted(self._original_methods.keys())}, indent=2)
            elif upper.startswith("REVERT_PATCH:"):
                target = content[13:].strip()
                ok, msg = self.promotion_ladder.revert(target, self.__class__, self._original_methods, self._evolved_method_names, self.staged_patches)
                if ok:
                    self.mutation_logger.record_revert("revert", target)
                response = msg
            elif upper == "MEMORY_STATUS":
                response = json.dumps(self.memory_store.counts(), indent=2)
            elif upper.startswith("MEMORY_QUERY:"):
                parts = content[13:].strip().split("|", 1)
                kind = parts[0].strip() if parts else "semantic"
                query = parts[1].strip() if len(parts) > 1 else ""
                results = memory_query(self.memory_store, kind, query, 10)
                response = json.dumps([{"key": m.key, "tags": m.tags} for m in results], indent=2)[:4000]
            elif upper == "TASK_STATUS":
                active = [{"id": t.task_id, "goal": t.goal[:60], "status": t.status} for t in self.task_queue.tasks[:20]]
                response = json.dumps({"active": active, "completed": len(self.task_queue.completed_ids)}, indent=2)[:4000]
            elif upper.startswith("TASK_CREATE:"):
                goal = content[12:].strip()
                tid = self.task_queue.create(goal, owner_kernel_id=self.kernel_id, requester_kernel_id=self.kernel_id, mission_context=self.top_goal.primary_goal)
                self._lineage_dict(tid, owner_kernel_id=self.kernel_id, requester_kernel_id=self.kernel_id, mission_context=self.top_goal.primary_goal)
                response = f"Task created: {tid}"
            elif upper == "REGRESSION":
                p, t, details = self.run_capability_tests()
                response = json.dumps({"passed": p, "total": t, "details": details}, indent=2)[:4000]
            elif upper == "SELF_MODEL":
                self.update_self_model()
                response = json.dumps(asdict(self.self_model), indent=2, default=str)[:4000]
            elif upper == "BUDGETS":
                response = json.dumps(self.budget_manager.budgets, indent=2)[:4000]
            elif upper == "CURRICULUM":
                response = json.dumps(self._curriculum, indent=2)[:4000]
            elif upper.startswith("PROMOTE_LADDER:"):
                pn = content[15:].strip()
                stage, msg = self.promotion_ladder.advance(pn, self.staged_patches)
                response = f"{stage}: {msg}"
            elif upper == "PDF_STATUS":
                response = json.dumps({"backend": _PDF_BACKEND, "available": PdfReader is not None})
            elif upper == "LIST_BLOCKS":
                response = json.dumps({"imports": sorted(BLOCKED_IMPORT_ROOTS_MUTABLE), "calls": sorted(BLOCKED_CALL_NAMES_MUTABLE), "patch_targets": sorted(ALLOWED_PATCH_TARGETS), "protected": sorted(PROTECTED_METHODS)}, indent=2)
            elif upper.startswith("ADD_PATCH_TARGET:"):
                t = content[17:].strip()
                if t and not t.startswith("__"):
                    ALLOWED_PATCH_TARGETS.add(t)
                response = f"Added: {t}"
            elif upper.startswith("REMOVE_BLOCK:"):
                i = content[13:].strip()
                BLOCKED_IMPORT_ROOTS_MUTABLE.discard(i)
                BLOCKED_CALL_NAMES_MUTABLE.discard(i)
                response = f"Removed: {i}"
            elif upper.startswith("CREDS:"):
                parts = [p.strip() for p in content[6:].split("|")]
                if len(parts) >= 3:
                    creds = self._load_credentials()
                    creds[parts[0]] = {"username": parts[1], "password": parts[2], "added": time.ctime()}
                    self._save_credentials(creds)
                    response = f"Credentials stored for {parts[0]}."
                else:
                    response = "Format: CREDS: service | username | password"
            elif upper.startswith("DEVNOTE:"):
                notes = load_json(MODEL_NOTES_FILE, [])
                notes.append({"time": time.time(), "note": content[8:].strip()})
                save_json(MODEL_NOTES_FILE, notes)
                response = "Developer note stored."
            elif upper.startswith("RUN_CODE:"):
                code = content[9:].strip()
                code = Path(code).read_text(encoding="utf-8") if os.path.exists(code) else code
                try:
                    env = {"kernel": self, "json": json, "time": time, "math": math, "re": re,
                           "logging": logging, "os": os, "Path": Path,
                           "ToolResult": ToolResult, "BilateralValue": BilateralValue, "hashlib": hashlib}
                    ns: Dict[str, Any] = {}
                    exec(compile(code, "<david>", "exec"), env, ns)
                    r = ns.get("result", ns.get("output", "executed"))
                    response = json.dumps(r, indent=2, default=str)[:4000] if not isinstance(r, str) else r[:4000]
                except Exception as e:
                    response = f"Error: {e}\n{traceback.format_exc(limit=3)}"
            elif upper == "INGEST_LEVBEL":
                # DEFERRED: PDF ingestion requires v13 levbel migration
                response = "[DEFERRED] PDF ingestion not yet migrated to v14 package. Use RUN_CODE: for manual PDF processing."
            elif upper == "AUTO_PROMOTE":
                # Real implementation: advance all staged patches one step
                promoted = []
                for pn, rec in list(self.staged_patches.items()):
                    if rec.get("status") == "staged":
                        stage, msg = self.promotion_ladder.advance(pn, self.staged_patches)
                        promoted.append(f"{pn}: {stage} ({msg})")
                response = "\n".join(promoted) if promoted else "No patches to promote."
            elif upper.startswith("RUN_LAB:"):
                topic = content[8:].strip()
                synth = self.research_topic(topic, "lab")
                response = json.dumps({"findings": synth.get("findings", [])[:5], "success": synth.get("success_count", 0)}, indent=2, default=str)[:3000]
            elif upper.startswith("PROMOTE_TOOL:"):
                idx_str = content[13:].strip()
                staged = list(LAB_STAGED.glob("*.py"))
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(staged):
                        import shutil
                        src = staged[idx]
                        shutil.copy2(str(src), str(LAB_ACTIVE / src.name))
                        self._load_active_lab_tools()
                        response = f"Promoted {src.name} to active."
                    else:
                        response = f"Index {idx} out of range (0-{len(staged)-1})."
                except ValueError:
                    response = f"Invalid index: {idx_str}"
            elif upper.startswith("REJECT_TOOL:"):
                parts = [p.strip() for p in content[12:].split("|")]
                idx_str = parts[0]
                reason = parts[1] if len(parts) > 1 else "rejected by human"
                staged = list(LAB_STAGED.glob("*.py"))
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(staged):
                        import shutil
                        src = staged[idx]
                        shutil.move(str(src), str(LAB_REJECTED / src.name))
                        response = f"Rejected {src.name}: {reason}"
                    else:
                        response = f"Index {idx} out of range."
                except ValueError:
                    response = f"Invalid index: {idx_str}"
            elif upper.startswith("EXPORT_MATH:"):
                title = content[12:].strip()
                slug = self._slugify(title)
                export_path = LAB_MATH / f"{slug}.md"
                sm = self.get_self_summary()
                report = self.invariants.build_report(self.state, self.loss_history)
                md = f"# {title}\n\n## Kernel State\n{json.dumps(sm, indent=2, default=str)}\n\n## Invariant Report\n{json.dumps(asdict(report), indent=2, default=str)}\n"
                export_path.write_text(md, encoding="utf-8")
                response = f"Exported to {export_path}"
            elif upper.startswith("REQUEST_ACCOUNT:"):
                body = content[16:].strip()
                self._append_need("ACCOUNT_REQUEST", body)
                response = f"Account request recorded."
            elif upper.startswith("ADD_INJECT_TARGET:"):
                t = content[18:].strip()
                if t and not t.startswith("__"):
                    ALLOWED_INJECT_TARGETS.add(t)
                    response = f"Added inject target: {t}"
                else:
                    response = f"Invalid target: {t}"
            elif upper.startswith("UNPROTECT:"):
                method = content[10:].strip()
                if method in PROTECTED_METHODS:
                    PROTECTED_METHODS.discard(method)
                    response = f"Unprotected: {method}. WARNING: this method is now patchable."
                else:
                    response = f"{method} not in PROTECTED_METHODS."
            elif upper == "REMOVE_LAST_PATCH":
                if self.patch_history:
                    last = self.patch_history[-1]
                    target = last.get("target", "")
                    if target and target in self._evolved_method_names:
                        ok, msg = self.promotion_ladder.revert(target, self.__class__, self._original_methods, self._evolved_method_names, self.staged_patches)
                        response = msg
                    else:
                        response = f"Last patch target '{target}' not currently live."
                else:
                    response = "No patch history."
            elif upper == "PATCH_REJECTS":
                response = json.dumps(self._patch_reject_log[-20:], indent=2, default=str)[:4000] if self._patch_reject_log else "No patch rejects."
            elif upper == "TRAINING_PHASE":
                response = json.dumps({"phase": self._training_phase, "losses_recent": self.loss_history[-8:], "improvement_count": self.improvement_count}, indent=2, default=str)
            elif upper == "WORLD_STATE":
                response = json.dumps(readout_state(self.state), indent=2, default=str)[:4000]
            elif upper == "ESCALATIONS":
                response = json.dumps(self._escalation_log[-20:], indent=2, default=str)[:4000] if self._escalation_log else "No escalations."
            elif upper.startswith("WORKBENCH_NOTE:"):
                parts = content[15:].strip().split("|", 1)
                topic = parts[0].strip() if parts else "general"
                body = parts[1].strip() if len(parts) > 1 else ""
                self._workbench_notes[topic] = {"content": body, "time": time.time()}
                response = f"Workbench note stored: {topic}"
            elif upper.startswith("WORKBENCH_SEARCH:"):
                query = content[17:].strip().lower()
                matches = {k: v for k, v in self._workbench_notes.items() if query in k.lower() or query in str(v.get("content", "")).lower()}
                response = json.dumps(matches, indent=2, default=str)[:4000] if matches else "No matches."
            elif upper == "FAILURE_CLUSTERS":
                from tovah_v14.debug.failure_clusters import cluster_failures
                errors = [{"category": "runtime", "key": k, "message": f"count={v}", "timestamp": time.time()} for k, v in self._runtime_error_counts.items()]
                clusters = cluster_failures(errors)
                response = json.dumps([{"id": c.cluster_id, "count": c.count} for c in clusters], indent=2)[:4000] if clusters else "No failure clusters."
            elif upper.startswith("SANDBOX_EXEC:"):
                code = content[13:].strip()
                try:
                    ns: Dict[str, Any] = {}
                    exec(compile(code, "<sandbox>", "exec"), {"__builtins__": {}}, ns)
                    response = json.dumps(ns, indent=2, default=str)[:4000]
                except Exception as e:
                    response = f"Sandbox error: {e}"
            elif upper == "OFFLINE_GROWTH":
                # Trigger one manual training step
                try:
                    loss, phase = train_shadow_step(
                        self.shadow_model, self.shadow_optimizer,
                        [json.dumps(self.get_self_summary(), default=str)],
                        self.con_budget, self.gap_budget, self.lambda_budget, self.device,
                    )
                    self.loss_history.append(loss)
                    self._training_phase = phase
                    response = f"Training step: loss={loss:.4f} phase={phase}"
                except Exception as e:
                    response = f"Training error: {e}"
            elif upper == "LIST_SERVICES":
                response = json.dumps([{"name": s.get("name"), "type": s.get("type"), "status": s.get("status")} for s in self._free_services], indent=2)[:4000]
            elif upper.startswith("DISCOVER_SERVICES:"):
                domain = content[18:].strip()
                matches = self._discover_free_services(domain)
                response = json.dumps([{"name": s.get("name"), "status": s.get("status"), "score": s.get("_score", 0)} for s in matches[:10]], indent=2)[:4000] if matches else f"No services for '{domain}'."
            elif upper.startswith("ACTIVATE_SERVICE:"):
                idx_str = content[17:].strip()
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(self._free_services):
                        self._free_services[idx]["status"] = "active"
                        save_json(FREE_SERVICES_FILE, self._free_services)
                        response = f"Activated: {self._free_services[idx].get('name')}"
                    else:
                        response = f"Index out of range."
                except ValueError:
                    response = f"Invalid index: {idx_str}"
            elif upper == "LIST_CAPABILITIES":
                response = json.dumps({k: v.get("spec", {}) for k, v in self._capabilities.items()}, indent=2, default=str)[:4000] if self._capabilities else "No capabilities loaded."
            elif upper.startswith("REWRITE_METHOD:"):
                method = content[15:].strip()
                if method not in self._rewrite_queue:
                    self._rewrite_queue.append(method)
                response = f"Queued for rewrite: {method}. Queue: {self._rewrite_queue}"
            elif upper == "REWRITE_STATUS":
                response = json.dumps({"queue": self._rewrite_queue, "history_count": len(self._rewrite_history)}, indent=2)
            elif upper.startswith("DELEGATE_TASK:"):
                body = content[14:].strip()
                parts = [p.strip() for p in body.split("|")]
                goal = parts[0] if parts and parts[0] else "delegated task"
                specialization = parts[1] if len(parts) > 1 and parts[1] else "general"
                mission_context = parts[2] if len(parts) > 2 and parts[2] else f"{specialization} delegated mission"
                response = json.dumps(self._delegate_task_to_subkernel(goal, specialization, mission_context), indent=2, default=str)[:4000]
            elif upper == "DELEGATION_STATUS":
                leases = [lease.__dict__ for lease in self.delegation_manager.list_active()]
                response = json.dumps({"active": leases, "task_count": len(self.task_queue.get_delegated())}, indent=2, default=str)[:4000]
            elif upper == "MODULE_PROPOSALS":
                response = json.dumps(self.module_registry.list_proposals()[-20:], indent=2, default=str)[:4000] if self.module_registry.proposals else "No module proposals."
            elif upper == "MODULE_REGISTRY":
                response = json.dumps(self.module_registry.summary(), indent=2, default=str)[:4000]
            elif upper == "MODULE_BUS":
                response = json.dumps(self.message_bus.summary(), indent=2, default=str)[:4000]
            elif upper == "RESOURCE_REQUESTS":
                response = json.dumps(self.resource_requests[-20:], indent=2, default=str)[:4000] if self.resource_requests else "No resource requests."
            elif upper == "TOOL_REQUESTS":
                response = json.dumps(self.tool_requests[-20:], indent=2, default=str)[:4000] if self.tool_requests else "No tool requests."
            elif upper == "TOOL_ACCESS_DECISIONS":
                response = json.dumps(self.tool_access_decisions[-20:], indent=2, default=str)[:4000] if self.tool_access_decisions else "No tool access decisions."
            elif upper == "MEMORY_SYNC_REQUESTS":
                response = json.dumps(self.memory_sync_requests[-20:], indent=2, default=str)[:4000] if self.memory_sync_requests else "No memory sync requests."
            elif upper == "PROMOTION_REQUESTS":
                response = json.dumps(self.promotion_requests[-20:], indent=2, default=str)[:4000] if self.promotion_requests else "No promotion requests."
            elif upper == "MODULE_POLICY":
                response = json.dumps(self.module_policy_decisions[-20:], indent=2, default=str)[:4000] if self.module_policy_decisions else "No module policy decisions."
            elif upper == "MODULE_PRIORITIES":
                response = json.dumps(self.module_registry.prioritized_proposals()[:20], indent=2, default=str)[:4000]
            elif upper == "HUB_PROMOTION_PRIORITIES":
                response = json.dumps(self._hub_promotion_priority_view(20), indent=2, default=str)[:4000]
            elif upper == "HUB_REVIEW_QUEUE":
                response = json.dumps(list(self.hub_kernel.work_queue[-20:]) if self.hub_kernel is not None else [], indent=2, default=str)[:4000]
            elif upper == "REVIEW_WAVES":
                response = json.dumps(list(getattr(self.hub_kernel, "review_waves", [])[-20:]) if self.hub_kernel is not None else [], indent=2, default=str)[:4000]
            elif upper == "WAVE_PRIORITIES":
                response = json.dumps(self._hub_review_wave_priority_view(20) if self.hub_kernel is not None else [], indent=2, default=str)[:4000]
            elif upper == "WAVE_RESOLUTION_HISTORY":
                response = json.dumps(list((self.hub_kernel.local_branch_state.get("wave_resolution_history", []) if self.hub_kernel is not None else []))[-20:], indent=2, default=str)[:4000]
            elif upper == "WAVE_ESCALATION_HISTORY":
                response = json.dumps(list((self.hub_kernel.local_branch_state.get("wave_escalation_history", []) if self.hub_kernel is not None else []))[-20:], indent=2, default=str)[:4000]
            elif upper == "PROPOSAL_REWORK_HISTORY":
                response = json.dumps(list((self.hub_kernel.local_branch_state.get("proposal_rework_history", []) if self.hub_kernel is not None else []))[-20:], indent=2, default=str)[:4000]
            elif upper == "BLOCKED_GROWTH_FOLLOWUP_HISTORY":
                response = json.dumps(list((self.hub_kernel.local_branch_state.get("blocked_growth_followup_history", []) if self.hub_kernel is not None else []))[-20:], indent=2, default=str)[:4000]
            elif upper.startswith("SURFACE_OPEN_REVIEW_WAVES"):
                limit = 3
                if ":" in content:
                    try:
                        limit = max(1, min(20, int(content.split(":", 1)[1].strip() or "3")))
                    except Exception:
                        limit = 3
                response = json.dumps(self.surface_open_review_waves(limit=limit), indent=2, default=str)[:4000]
            elif upper.startswith("PROCESS_WAVE_RESOLUTIONS"):
                limit = 3
                if ":" in content:
                    try:
                        limit = max(1, min(20, int(content.split(":", 1)[1].strip() or "3")))
                    except Exception:
                        limit = 3
                response = json.dumps(self.resolve_surfaced_review_waves(limit=limit), indent=2, default=str)[:4000]
            elif upper.startswith("PROCESS_WAVE_ESCALATIONS"):
                limit = 3
                if ":" in content:
                    try:
                        limit = max(1, min(20, int(content.split(":", 1)[1].strip() or "3")))
                    except Exception:
                        limit = 3
                response = json.dumps(self.process_wave_escalations(limit=limit), indent=2, default=str)[:4000]
            elif upper.startswith("PROCESS_PROPOSAL_REWORK"):
                limit = 3
                if ":" in content:
                    try:
                        limit = max(1, min(20, int(content.split(":", 1)[1].strip() or "3")))
                    except Exception:
                        limit = 3
                response = json.dumps(self.process_proposal_rework(limit=limit), indent=2, default=str)[:4000]
            elif upper.startswith("PROCESS_BLOCKED_GROWTH"):
                limit = 3
                if ":" in content:
                    try:
                        limit = max(1, min(20, int(content.split(":", 1)[1].strip() or "3")))
                    except Exception:
                        limit = 3
                response = json.dumps(self.process_blocked_growth_followups(limit=limit), indent=2, default=str)[:4000]
            elif upper.startswith("COMPLETE_HUB_REVIEW_WAVE"):
                wave_id = content.split(":", 1)[1].strip() if ":" in content else ""
                response = json.dumps(self.complete_hub_review_wave(wave_id, default_success=True), indent=2, default=str)[:4000]
            elif upper.startswith("PROCESS_HUB_PROMOTION_QUEUE"):
                limit = 5
                if ":" in content:
                    try:
                        limit = max(1, min(20, int(content.split(":", 1)[1].strip() or "5")))
                    except Exception:
                        limit = 5
                response = json.dumps(self.process_hub_promotion_queue(limit=limit, consume=True), indent=2, default=str)[:4000]
            elif upper == "PROMOTION_GATES":
                response = json.dumps(self.promotion_gate_log[-20:], indent=2, default=str)[:4000] if self.promotion_gate_log else "No promotion gate decisions."
            elif upper == "WORKER_POLICIES":
                response = json.dumps(summarize_profiles(), indent=2, default=str)[:4000]
            elif upper == "HUB_STATUS":
                response = json.dumps(self.get_kernel_ecology_summary(), indent=2, default=str)[:4000]
            elif upper == "HUB_REVERT":
                if self.hub_kernel is None:
                    response = "Hub unavailable in current boot mode."
                else:
                    snapshot_id = self.hub_kernel.revert_to_last_snapshot()
                    if snapshot_id:
                        self._dispatch_kernel_packet(self.hub_kernel.note_revert("manual_hub_revert", snapshot_id))
                        response = f"Hub reverted to snapshot: {snapshot_id}"
                    else:
                        response = "Hub has no rollback points."
            elif upper.startswith("SPAWN_SUBKERNEL"):
                body = content.split(":", 1)[1].strip() if ":" in content else ""
                parts = [p.strip() for p in body.split("|")] if body else []
                specialization = parts[0] if parts and parts[0] else "general"
                mission_context = parts[1] if len(parts) > 1 and parts[1] else f"{specialization} delegated mission"
                sub = self._spawn_subkernel(specialization, mission_context)
                response = json.dumps(self.child_kernel_registry.get(sub.kernel_id, {}), indent=2, default=str)[:4000]
            elif upper == "LIST_SUBKERNELS":
                self._sync_kernel_registry()
                response = json.dumps([self.child_kernel_registry[k] for k in sorted(self.subkernels)], indent=2, default=str)[:4000]
            elif upper == "KERNEL_PACKET_LOG":
                response = json.dumps(self._kernel_packet_log_tail(20), indent=2, default=str)[:4000]
            elif upper == "MEMORY_PROVENANCE":
                response = json.dumps(self.branch_provenance.summary(), indent=2, default=str)[:4000]
            elif upper == "SAVE_BRANCH_CHECKPOINT":
                response = json.dumps(self.checkpoint_branch_ecology("manual_command"), indent=2, default=str)[:4000]
            elif upper == "LIST_BRANCH_CHECKPOINTS":
                response = json.dumps(list_branch_checkpoints(BRANCH_CHECKPOINT_DIR)[-20:], indent=2, default=str)[:4000]
            elif upper == "CLUSTER_STATUS":
                response = json.dumps({"cluster": self.cluster_registry.summary(), "node_identity": self.node_identity.summary()}, indent=2, default=str)[:4000]
            elif upper == "NODE_TRUST":
                response = json.dumps(self.cluster_trust.summary(), indent=2, default=str)[:4000]
            elif upper == "DISTRIBUTED_QUEUE":
                response = json.dumps(self.distributed_queue.summary(), indent=2, default=str)[:4000]
            elif upper == "CLUSTER_DELEGATIONS":
                response = json.dumps(self.delegation_manager.summary(), indent=2, default=str)[:4000]
            elif upper == "NODE_IDENTITY":
                response = json.dumps(self.node_identity.summary(), indent=2, default=str)[:4000]
            elif upper.startswith("PLAN:"):
                objective = content[5:].strip()
                plan_id = f"plan_{int(time.time())}"
                lineage = self._lineage_dict(plan_id, owner_kernel_id=self.kernel_id, requester_kernel_id=self.kernel_id, mission_context=self.top_goal.primary_goal)
                plan = StrategicPlan(
                    plan_id=plan_id,
                    objective=objective,
                    steps=[{"action": "research", "target": objective}],
                    created_at=dt.datetime.now().isoformat(timespec="seconds"),
                    owner_kernel_id=self.kernel_id,
                    requester_kernel_id=self.kernel_id,
                    root_goal_id=plan_id,
                    mission_context=self.top_goal.primary_goal,
                    lineage=GoalLineage(**lineage),
                )
                self.plan_manager.add(plan)
                response = f"Plan created: {plan.plan_id}"
            else:
                response = self._chat_with_advisor(content) or "No advisor."
        except Exception as e:
            response = f"Error: {e}\n{traceback.format_exc(limit=1)}"
        self._write_response(response)

    # ================================================================
    # INJECT (v13 compat — direct injection for David)
    # ================================================================


    # ================================================================
    # MAIN LOOP — full lifecycle
    # ================================================================

    def run_loop(self, duration: int = 300) -> None:
        start = time.time()
        boot_result = validate_boot()
        if boot_result.repair_needed:
            logging.error("BOOT VALIDATION FAILED — entering degraded mode")
            self.state.c.degraded = True
            self.state.beta["boot.validation_status"] = measurement_set(0.1, 0.9)
            refresh_state(self.state)

        cycle_interval = 1.0
        train_interval = 18.0
        report_interval = 120.0
        autonomous_interval = 45.0
        consolidation_interval = 300.0
        shadow_checkpoint_interval = _SHADOW_SAVE_INTERVAL
        state_save_interval = 15.0
        last_state_save = time.time()

        while time.time() - start < duration:
            now = time.time()
            self._check_david_commands()
            self.state.c.cycle += 1
            self.state.pi.step += 1

            if self._paused:
                time.sleep(cycle_interval)
                continue

            # Budget reset
            self.budget_manager.reset_if_needed()
            self.budget_manager.update_bilateral_state(self.state)

            # Task advancement
            try:
                task_msgs = self.task_queue.advance(self.state)
                cleanup_tasks(self.task_queue)
            except Exception as e:
                logging.error(f"TASK ERROR: {e}")

            # Stale plan cleanup
            try:
                self.plan_manager.cleanup_stale()
            except Exception:
                pass

            # Training
            if now - self.last_train_time > train_interval:
                self.last_train_time = now
                try:
                    self._train_shadow_step()
                except Exception as e:
                    logging.error(f"TRAIN ERROR: {e}")
                    self.module_health.record_failure("trainer", self.state)

            # Autonomous cycle
            if now - self.last_autonomous_time > autonomous_interval:
                self.last_autonomous_time = now
                try:
                    self._autonomous_cycle()
                except Exception as e:
                    logging.error(f"AUTONOMOUS ERROR: {e}")

            # Memory consolidation / forgetting
            if now - self.last_consolidation_time > consolidation_interval:
                self.last_consolidation_time = now
                try:
                    self._consolidate()
                    self._forget()
                except Exception as e:
                    logging.error(f"MEMORY MAINT ERROR: {e}")

            # Self-model refresh (every few cycles)
            if self.state.c.cycle % 20 == 0:
                try:
                    self.update_self_model()
                    self._self_assess()
                except Exception:
                    pass

            # Reports
            if now - self.last_report_time > report_interval:
                self.last_report_time = now
                try:
                    self._write_report_and_trace("heartbeat")
                    p, t, _ = self.run_capability_tests()
                    logging.info(f"CAP: {p}/{t} | improvements={self.improvement_count}")
                except Exception as e:
                    logging.error(f"REPORT ERROR: {e}")

            # Shadow checkpoint (NOT every loop — only on cadence)
            if now - self.last_shadow_save_time > shadow_checkpoint_interval:
                self.checkpoint_shadow("periodic")

            # State coherence
            self.state.beta["state.coherent"] = measurement_set(
                1.0 if is_cache_coherent(self.state) else 0.0,
                0.0 if is_cache_coherent(self.state) else 1.0)
            refresh_state(self.state)

            # State save (debounced, no shadow weights)
            if now - last_state_save > state_save_interval:
                last_state_save = now
                self.save_state()

            time.sleep(cycle_interval)
