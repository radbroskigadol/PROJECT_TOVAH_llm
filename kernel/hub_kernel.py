"""
TOVAH v14 kernel/hub_kernel.py — Governed experimental hub scaffold.

This remains intentionally lightweight, but now owns branch-local queues so it
can incubate proposals and work before asking sovereign main for review.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.kernel.action_model import (
    BlockedGrowthRecord,
    MemorySyncRequest,
    ModuleProposal,
    PatchProposal,
    PromotionRequest,
    ResourceRequest,
    SpawnRequest,
    ToolAccessRequest,
    TrustReport,
)
from tovah_v14.kernel.kernel_roles import KernelLifecycle, KernelRole, RiskClass, TrustLevel
from tovah_v14.kernel.packet import KernelPacket, PacketKind, make_packet


@dataclass
class HubSnapshot:
    snapshot_id: str
    created_at: float
    lifecycle: str
    local_branch_state: Dict[str, Any]
    memory_branch: List[Dict[str, Any]]
    module_registry: Dict[str, Dict[str, Any]]
    experimental_tool_registry: Dict[str, Dict[str, Any]]
    blocked_growth: List[Dict[str, Any]]
    proposal_queue: List[Dict[str, Any]]
    work_queue: List[Dict[str, Any]]
    promotion_queue: List[Dict[str, Any]]




def artifact_dedup_key(payload: Dict[str, Any] | None) -> str:
    """Compute a stable dedup key for a hub-routed artifact.

    Deterministic by design: the key for a given (kind, name, module_kind,
    desired_stage) tuple is the same regardless of when in the lifecycle
    `proposal_id` becomes known. Without this, the same logical artifact
    can produce two different keys (one before, one after the proposal_id
    is assigned) and dedup breaks. (Audit S-4.)
    """
    payload = dict(payload or {})
    artifact_kind = str(payload.get("artifact_kind", payload.get("kind", "")) or "")
    artifact_name = str(payload.get("artifact_name", payload.get("module_name", "")) or "")
    proposal_id = str(payload.get("proposal_id", "") or "")
    desired_stage = str(payload.get("desired_stage", payload.get("target", payload.get("target_kernel_id", ""))) or "")
    review_wave_id = str(payload.get("review_wave_id", payload.get("wave_id", "")) or "")
    module_kind = str(payload.get("module_kind", "") or "")
    if artifact_kind == "module":
        # Pin to (name, kind, target). proposal_id is intentionally excluded
        # so the key doesn't change once the proposal_id is assigned.
        base = artifact_name or proposal_id or "unknown"
        return f"module::{base}::{desired_stage or 'hub'}::{module_kind}"
    if artifact_kind == "patch":
        return f"patch::{artifact_name}::{desired_stage or 'sandbox_passed'}"
    if artifact_name or proposal_id:
        return f"{artifact_kind or 'work'}::{artifact_name or proposal_id}::{desired_stage}::{review_wave_id}"
    return f"{str(payload.get('kind','work'))}::{review_wave_id}::{desired_stage}"

class HubKernel:
    """Experimental branch kernel with rollback, local queues, and packetized reporting."""

    def __init__(
        self,
        kernel_id: str = "hub",
        *,
        parent_kernel_id: str = "main",
        mission_context: str = "experimental branch",
    ) -> None:
        self.kernel_id = kernel_id
        self.role = KernelRole.HUB.value
        self.parent_kernel_id = parent_kernel_id
        self.mission_context = mission_context
        self.lifecycle = KernelLifecycle.BORN.value
        self.local_branch_state: Dict[str, Any] = {}
        self.memory_branch: List[Dict[str, Any]] = []
        self.module_registry: Dict[str, Dict[str, Any]] = {}
        self.experimental_tool_registry: Dict[str, Dict[str, Any]] = {}
        self.blocked_growth: List[BlockedGrowthRecord] = []
        self.rollback_points: List[HubSnapshot] = []
        self.trust_from_main = TrustLevel.PROVISIONAL.value
        self.packet_log: List[KernelPacket] = []
        self.proposal_queue: List[Dict[str, Any]] = []
        self.work_queue: List[Dict[str, Any]] = []
        self.promotion_queue: List[Dict[str, Any]] = []
        self.review_waves: List[Dict[str, Any]] = []

    def transition_to(self, lifecycle: str) -> None:
        self.lifecycle = lifecycle

    def snapshot(self, label: str = "") -> str:
        snapshot_id = label or f"hubsnap_{int(time.time() * 1000)}"
        snap = HubSnapshot(
            snapshot_id=snapshot_id,
            created_at=time.time(),
            lifecycle=self.lifecycle,
            local_branch_state=copy.deepcopy(self.local_branch_state),
            memory_branch=copy.deepcopy(self.memory_branch),
            module_registry=copy.deepcopy(self.module_registry),
            experimental_tool_registry=copy.deepcopy(self.experimental_tool_registry),
            blocked_growth=[item.to_dict() for item in self.blocked_growth],
            proposal_queue=copy.deepcopy(self.proposal_queue),
            work_queue=copy.deepcopy(self.work_queue),
            promotion_queue=copy.deepcopy(self.promotion_queue),
            # review waves are durable queue-learning state
        )
        self.rollback_points.append(snap)
        return snapshot_id

    def revert_to_last_snapshot(self) -> Optional[str]:
        if not self.rollback_points:
            return None
        self.transition_to(KernelLifecycle.REVERTING.value)
        snap = self.rollback_points[-1]
        self.local_branch_state = copy.deepcopy(snap.local_branch_state)
        self.memory_branch = copy.deepcopy(snap.memory_branch)
        self.module_registry = copy.deepcopy(snap.module_registry)
        self.experimental_tool_registry = copy.deepcopy(snap.experimental_tool_registry)
        self.blocked_growth = [BlockedGrowthRecord(**item) for item in snap.blocked_growth]
        self.proposal_queue = copy.deepcopy(snap.proposal_queue)
        self.work_queue = copy.deepcopy(snap.work_queue)
        self.promotion_queue = copy.deepcopy(snap.promotion_queue)
        self.lifecycle = snap.lifecycle
        return snap.snapshot_id

    def _emit(self, packet: KernelPacket) -> KernelPacket:
        self.packet_log.append(packet)
        return packet

    def queue_goal_work(self, goal: str, *, specialization: str = "general", parent_goal_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        item = {
            "goal": goal,
            "specialization": specialization,
            "parent_goal_id": parent_goal_id,
            "mission_context": self.mission_context,
            "metadata": dict(metadata or {}),
            "queued_at": time.time(),
            "status": "queued",
        }
        self.work_queue.append(item)
        self.work_queue = self.work_queue[-200:]
        return item

    def queue_module_proposal(self, proposal: ModuleProposal | Dict[str, Any]) -> Dict[str, Any]:
        payload = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
        payload.update({"queued_at": time.time(), "kind": "module", "status": "queued"})
        self.proposal_queue.append(payload)
        self.proposal_queue = self.proposal_queue[-200:]
        self.module_registry[payload.get("module_name", f"proposal_{len(self.proposal_queue)}")] = {
            "proposal_id": payload.get("proposal_id", ""),
            "module_kind": payload.get("module_kind", ""),
            "status": payload.get("status", "queued"),
            "promotion_target": payload.get("promotion_target", "hub"),
        }
        return payload

    def queue_patch_proposal(self, proposal: PatchProposal | Dict[str, Any]) -> Dict[str, Any]:
        payload = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
        payload.update({"queued_at": time.time(), "kind": "patch", "status": "queued"})
        self.proposal_queue.append(payload)
        self.proposal_queue = self.proposal_queue[-200:]
        return payload

    def queue_promotion_request(self, request: PromotionRequest | Dict[str, Any]) -> Dict[str, Any]:
        payload = request.to_dict() if hasattr(request, "to_dict") else dict(request)
        payload.setdefault("queued_at", time.time())
        payload["status"] = payload.get("status", "queued") or "queued"
        artifact_kind = str(payload.get("artifact_kind", "") or "")
        artifact_name = str(payload.get("artifact_name", "") or "")
        proposal_id = str(payload.get("proposal_id", "") or "")
        desired_stage = str(payload.get("desired_stage", payload.get("target", payload.get("target_kernel_id", ""))) or "")
        merged = dict(payload)
        merged["artifact_key"] = artifact_dedup_key(merged)
        kept = []
        for item in self.promotion_queue:
            same = str(item.get("artifact_key", artifact_dedup_key(item)) or "") == str(merged.get("artifact_key", "") or "")
            if same:
                existing = dict(item)
                merged["queued_at"] = min(
                    float(existing.get("queued_at", merged.get("queued_at", time.time())) or time.time()),
                    float(merged.get("queued_at", time.time()) or time.time()),
                )
                merged["rework_quality"] = max(float(existing.get("rework_quality", 0.0) or 0.0), float(merged.get("rework_quality", 0.0) or 0.0))
                merged["evidence_quality"] = max(float(existing.get("evidence_quality", 0.0) or 0.0), float(merged.get("evidence_quality", 0.0) or 0.0))
                merged["duplicate_count"] = int(existing.get("duplicate_count", 1) or 1) + int(merged.get("duplicate_count", 1) or 1)
                status_order = {"evidence_ready": 4, "reworked_ready": 4, "queued": 3, "deferred_cooldown": 2}
                if status_order.get(str(existing.get("status", "queued")), 0) > status_order.get(str(merged.get("status", "queued")), 0):
                    merged["status"] = existing.get("status", merged.get("status", "queued"))
                    merged["queue_status"] = existing.get("queue_status", merged.get("queue_status", merged.get("status", "queued")))
                continue
            kept.append(item)
        kept.append(merged)
        self.promotion_queue = kept[-200:]
        return merged


    def export_state(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "role": self.role,
            "parent_kernel_id": self.parent_kernel_id,
            "mission_context": self.mission_context,
            "lifecycle": self.lifecycle,
            "local_branch_state": copy.deepcopy(self.local_branch_state),
            "memory_branch": copy.deepcopy(self.memory_branch),
            "module_registry": copy.deepcopy(self.module_registry),
            "experimental_tool_registry": copy.deepcopy(self.experimental_tool_registry),
            "blocked_growth": [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in self.blocked_growth],
            "rollback_points": [
                {
                    "snapshot_id": snap.snapshot_id,
                    "created_at": snap.created_at,
                    "lifecycle": snap.lifecycle,
                    "local_branch_state": copy.deepcopy(snap.local_branch_state),
                    "memory_branch": copy.deepcopy(snap.memory_branch),
                    "module_registry": copy.deepcopy(snap.module_registry),
                    "experimental_tool_registry": copy.deepcopy(snap.experimental_tool_registry),
                    "blocked_growth": copy.deepcopy(snap.blocked_growth),
                    "proposal_queue": copy.deepcopy(snap.proposal_queue),
                    "work_queue": copy.deepcopy(snap.work_queue),
                    "promotion_queue": copy.deepcopy(snap.promotion_queue),
                }
                for snap in self.rollback_points
            ],
            "trust_from_main": self.trust_from_main,
            "packet_log": [pkt.to_dict() if hasattr(pkt, "to_dict") else dict(pkt) for pkt in self.packet_log],
            "proposal_queue": copy.deepcopy(self.proposal_queue),
            "work_queue": copy.deepcopy(self.work_queue),
            "promotion_queue": copy.deepcopy(self.promotion_queue),
            "review_waves": copy.deepcopy(self.review_waves),
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any] | None) -> "HubKernel | None":
        if not isinstance(data, dict) or not data:
            return None
        hub = cls(
            kernel_id=str(data.get("kernel_id", "hub")),
            parent_kernel_id=str(data.get("parent_kernel_id", "main")),
            mission_context=str(data.get("mission_context", "experimental branch")),
        )
        hub.lifecycle = str(data.get("lifecycle", hub.lifecycle))
        hub.local_branch_state = copy.deepcopy(dict(data.get("local_branch_state", {})))
        hub.memory_branch = copy.deepcopy(list(data.get("memory_branch", [])))
        hub.module_registry = copy.deepcopy(dict(data.get("module_registry", {})))
        hub.experimental_tool_registry = copy.deepcopy(dict(data.get("experimental_tool_registry", {})))
        hub.blocked_growth = [BlockedGrowthRecord(**dict(item)) for item in list(data.get("blocked_growth", [])) if isinstance(item, dict)]
        hub.rollback_points = []
        for snap in list(data.get("rollback_points", [])):
            try:
                hub.rollback_points.append(HubSnapshot(**dict(snap)))
            except Exception:
                pass
        hub.trust_from_main = str(data.get("trust_from_main", hub.trust_from_main))
        hub.packet_log = [KernelPacket(**dict(pkt)) for pkt in list(data.get("packet_log", [])) if isinstance(pkt, dict)]
        hub.proposal_queue = copy.deepcopy(list(data.get("proposal_queue", [])))
        hub.work_queue = copy.deepcopy(list(data.get("work_queue", [])))
        hub.promotion_queue = copy.deepcopy(list(data.get("promotion_queue", [])))
        hub.review_waves = copy.deepcopy(list(data.get("review_waves", [])))[-200:]
        return hub

    def status_packet(self) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.STATUS,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload={
                    "lifecycle": self.lifecycle,
                    "module_count": len(self.module_registry),
                    "experimental_tool_count": len(self.experimental_tool_registry),
                    "rollback_points": len(self.rollback_points),
                    "blocked_growth_count": len(self.blocked_growth),
                    "proposal_queue": len(self.proposal_queue),
                    "work_queue": len(self.work_queue),
                    "promotion_queue": len(self.promotion_queue),
                    "review_waves": len(self.review_waves),
                },
                required_action="observe",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=RiskClass.LOW.value,
            )
        )

    def request_budget(self, request: ResourceRequest) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.RESOURCE_REQUEST,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=request.to_dict(),
                required_action="review_resource_request",
                parent_goal_id=request.parent_goal_id,
                mission_context=self.mission_context,
                priority=request.priority,
                trust_level=self.trust_from_main,
                risk_class=request.risk_class,
                reply_expected=True,
            )
        )

    def request_tool_access(self, request: ToolAccessRequest) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.TOOL_REQUEST,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=request.to_dict(),
                required_action="review_tool_request",
                parent_goal_id=request.parent_goal_id,
                mission_context=self.mission_context,
                priority=request.priority,
                trust_level=self.trust_from_main,
                risk_class=request.risk_class,
                reply_expected=True,
            )
        )

    def propose_module(self, proposal: ModuleProposal) -> KernelPacket:
        self.queue_module_proposal(proposal)
        return self._emit(
            make_packet(
                PacketKind.MODULE_PROPOSAL,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=proposal.to_dict(),
                required_action="review_module_proposal",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=proposal.risk_class,
                reply_expected=True,
            )
        )

    def propose_patch(self, proposal: PatchProposal) -> KernelPacket:
        self.queue_patch_proposal(proposal)
        return self._emit(
            make_packet(
                PacketKind.PATCH_PROPOSAL,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=proposal.to_dict(),
                required_action="review_patch_proposal",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=proposal.risk_level,
                reply_expected=True,
            )
        )

    def request_promotion(self, request: PromotionRequest) -> KernelPacket:
        self.queue_promotion_request(request)
        return self._emit(
            make_packet(
                PacketKind.PROMOTION_REQUEST,
                source_kernel_id=self.kernel_id,
                target_kernel_id=request.target_kernel_id or self.parent_kernel_id,
                payload=request.to_dict(),
                required_action="review_promotion_request",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=request.risk_class,
                reply_expected=True,
            )
        )

    def request_spawn(self, request: SpawnRequest) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.SPAWN_REQUEST,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=request.to_dict(),
                required_action="review_spawn_request",
                parent_goal_id=request.parent_goal_id,
                mission_context=request.mission_context or self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=request.risk_class,
                reply_expected=True,
            )
        )

    def request_memory_sync(self, request: MemorySyncRequest | Dict[str, Any]) -> KernelPacket:
        if isinstance(request, dict):
            request = MemorySyncRequest(**dict(request))
        return self._emit(
            make_packet(
                PacketKind.MEMORY_SYNC,
                source_kernel_id=self.kernel_id,
                target_kernel_id=request.target_kernel_id,
                payload=request.to_dict(),
                required_action="review_memory_sync",
                parent_goal_id=request.parent_goal_id,
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=RiskClass.MEDIUM.value,
                reply_expected=True,
            )
        )

    def report_trust(self, report: TrustReport) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.TRUST_REPORT,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=report.to_dict(),
                required_action="review_trust_report",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=RiskClass.MEDIUM.value,
            )
        )

    def record_blocked_growth(self, record: BlockedGrowthRecord) -> KernelPacket:
        self.blocked_growth.append(record)
        self.transition_to(KernelLifecycle.DEGRADED.value)
        return self._emit(
            make_packet(
                PacketKind.BLOCKED_GROWTH,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=record.to_dict(),
                required_action="assist_blocked_growth",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=RiskClass.HIGH.value,
                reply_expected=True,
            )
        )

    def note_revert(self, reason: str, snapshot_id: str = "") -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.REVERT_NOTICE,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload={"reason": reason, "snapshot_id": snapshot_id},
                required_action="observe_revert_notice",
                mission_context=self.mission_context,
                trust_level=self.trust_from_main,
                risk_class=RiskClass.MEDIUM.value,
            )
        )
