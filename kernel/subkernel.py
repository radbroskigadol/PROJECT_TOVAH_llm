"""
TOVAH v14 kernel/subkernel.py — Specialized child-kernel scaffold.

Subkernels are narrow execution identities that report upward through packets.
They do not own canonical state and cannot mutate the sovereign kernel directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from tovah_v14.kernel.action_model import (
    BlockedGrowthRecord,
    ModuleProposal,
    PatchProposal,
    PromotionRequest,
    ResourceRequest,
    ToolAccessRequest,
)
from tovah_v14.kernel.kernel_roles import KernelLifecycle, KernelRole, RiskClass, TrustLevel
from tovah_v14.kernel.packet import KernelPacket, PacketKind, make_packet


@dataclass
class SubkernelState:
    specialization: str = ""
    local_state: Dict[str, Any] = field(default_factory=dict)
    local_tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    local_modules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_goals: List[Dict[str, Any]] = field(default_factory=list)
    blocked_growth: List[Dict[str, Any]] = field(default_factory=list)


class Subkernel:
    def __init__(
        self,
        kernel_id: str,
        *,
        parent_kernel_id: str = "main",
        specialization: str = "",
        mission_context: str = "delegated mission",
    ) -> None:
        self.kernel_id = kernel_id
        self.role = KernelRole.SUBKERNEL.value
        self.parent_kernel_id = parent_kernel_id
        self.lifecycle = KernelLifecycle.BORN.value
        self.mission_context = mission_context
        self.trust_from_parent = TrustLevel.LOW.value
        self.state = SubkernelState(specialization=specialization)
        self.packet_log: List[KernelPacket] = []

    def _emit(self, packet: KernelPacket) -> KernelPacket:
        self.packet_log.append(packet)
        return packet


    def export_state(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "role": self.role,
            "parent_kernel_id": self.parent_kernel_id,
            "lifecycle": self.lifecycle,
            "mission_context": self.mission_context,
            "trust_from_parent": self.trust_from_parent,
            "state": {
                "specialization": self.state.specialization,
                "local_state": dict(self.state.local_state),
                "local_tools": dict(self.state.local_tools),
                "local_modules": dict(self.state.local_modules),
                "pending_goals": list(self.state.pending_goals),
                "blocked_growth": list(self.state.blocked_growth),
            },
            "packet_log": [pkt.to_dict() if hasattr(pkt, "to_dict") else dict(pkt) for pkt in self.packet_log],
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any] | None) -> "Subkernel | None":
        if not isinstance(data, dict) or not data:
            return None
        state_data = dict(data.get("state", {}))
        sub = cls(
            kernel_id=str(data.get("kernel_id", "subkernel")),
            parent_kernel_id=str(data.get("parent_kernel_id", "main")),
            specialization=str(state_data.get("specialization", "")),
            mission_context=str(data.get("mission_context", "delegated mission")),
        )
        sub.lifecycle = str(data.get("lifecycle", sub.lifecycle))
        sub.trust_from_parent = str(data.get("trust_from_parent", sub.trust_from_parent))
        sub.state.local_state = dict(state_data.get("local_state", {}))
        sub.state.local_tools = dict(state_data.get("local_tools", {}))
        sub.state.local_modules = dict(state_data.get("local_modules", {}))
        sub.state.pending_goals = list(state_data.get("pending_goals", []))
        sub.state.blocked_growth = list(state_data.get("blocked_growth", []))
        sub.packet_log = [KernelPacket(**dict(pkt)) for pkt in list(data.get("packet_log", [])) if isinstance(pkt, dict)]
        return sub

    def status_packet(self) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.STATUS,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload={
                    "lifecycle": self.lifecycle,
                    "specialization": self.state.specialization,
                    "tool_count": len(self.state.local_tools),
                    "module_count": len(self.state.local_modules),
                    "pending_goals": len(self.state.pending_goals),
                },
                required_action="observe",
                mission_context=self.mission_context,
                trust_level=self.trust_from_parent,
                risk_class=RiskClass.LOW.value,
            )
        )

    def receive_goal(self, goal_payload: Dict[str, Any]) -> None:
        self.state.pending_goals.append(dict(goal_payload))
        self.lifecycle = KernelLifecycle.EXPERIMENTAL.value

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
                trust_level=self.trust_from_parent,
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
                trust_level=self.trust_from_parent,
                risk_class=request.risk_class,
                reply_expected=True,
            )
        )

    def propose_module(self, proposal: ModuleProposal) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.MODULE_PROPOSAL,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=proposal.to_dict(),
                required_action="review_module_proposal",
                mission_context=self.mission_context,
                trust_level=self.trust_from_parent,
                risk_class=proposal.risk_class,
                reply_expected=True,
            )
        )

    def propose_patch(self, proposal: PatchProposal) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.PATCH_PROPOSAL,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=proposal.to_dict(),
                required_action="review_patch_proposal",
                mission_context=self.mission_context,
                trust_level=self.trust_from_parent,
                risk_class=proposal.risk_level,
                reply_expected=True,
            )
        )

    def request_promotion(self, request: PromotionRequest) -> KernelPacket:
        return self._emit(
            make_packet(
                PacketKind.PROMOTION_REQUEST,
                source_kernel_id=self.kernel_id,
                target_kernel_id=request.target_kernel_id or self.parent_kernel_id,
                payload=request.to_dict(),
                required_action="review_promotion_request",
                mission_context=self.mission_context,
                trust_level=self.trust_from_parent,
                risk_class=request.risk_class,
                reply_expected=True,
            )
        )

    def record_blocked_growth(self, record: BlockedGrowthRecord) -> KernelPacket:
        self.state.blocked_growth.append(record.to_dict())
        self.lifecycle = KernelLifecycle.DEGRADED.value
        return self._emit(
            make_packet(
                PacketKind.BLOCKED_GROWTH,
                source_kernel_id=self.kernel_id,
                target_kernel_id=self.parent_kernel_id,
                payload=record.to_dict(),
                required_action="assist_blocked_growth",
                mission_context=self.mission_context,
                trust_level=self.trust_from_parent,
                risk_class=RiskClass.HIGH.value,
                reply_expected=True,
            )
        )
