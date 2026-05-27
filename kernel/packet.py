"""
TOVAH v14 kernel/packet.py — Typed packet transport for kernel ecology.

Hub and subkernels report upward through governed packets rather than logs or
free-form strings.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

from tovah_v14.kernel.kernel_roles import RiskClass, TrustLevel


def _packet_id() -> str:
    return f"pkt_{uuid.uuid4().hex[:16]}"


class PacketKind(str):
    STATUS = "status_packet"
    GOAL = "goal_packet"
    RESOURCE_REQUEST = "resource_request_packet"
    TOOL_REQUEST = "tool_request_packet"
    MODULE_PROPOSAL = "module_proposal_packet"
    PATCH_PROPOSAL = "patch_proposal_packet"
    PROMOTION_REQUEST = "promotion_request_packet"
    BLOCKED_GROWTH = "blocked_growth_packet"
    SPAWN_REQUEST = "spawn_request_packet"
    TRUST_REPORT = "trust_report_packet"
    REVERT_NOTICE = "revert_notice_packet"
    MEMORY_SYNC = "memory_sync_packet"


@dataclass
class KernelPacket:
    """Canonical transport object between sovereign, hub, and subkernels."""

    packet_kind: str
    source_kernel_id: str
    target_kernel_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    required_action: str = "observe"
    packet_id: str = field(default_factory=_packet_id)
    parent_goal_id: str = ""
    mission_context: str = ""
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    trust_level: str = TrustLevel.PROVISIONAL.value
    risk_class: str = RiskClass.MEDIUM.value
    provenance: Dict[str, Any] = field(default_factory=dict)
    reply_expected: bool = False

    def __post_init__(self) -> None:
        self.packet_kind = str(self.packet_kind)
        self.trust_level = str(self.trust_level)
        self.risk_class = str(self.risk_class)
        if not isinstance(self.payload, dict):
            self.payload = dict(self.payload or {})
        if not isinstance(self.provenance, dict):
            self.provenance = dict(self.provenance or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "packet_kind": self.packet_kind,
            "source_kernel_id": self.source_kernel_id,
            "target_kernel_id": self.target_kernel_id,
            "parent_goal_id": self.parent_goal_id,
            "mission_context": self.mission_context,
            "payload": dict(self.payload),
            "required_action": self.required_action,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "trust_level": self.trust_level,
            "risk_class": self.risk_class,
            "provenance": dict(self.provenance),
            "reply_expected": self.reply_expected,
        }

    def reply(
        self,
        *,
        source_kernel_id: str,
        payload: Optional[Mapping[str, Any]] = None,
        required_action: str = "observe",
        trust_level: str = TrustLevel.PROVISIONAL.value,
        risk_class: str = RiskClass.LOW.value,
        reply_expected: bool = False,
    ) -> "KernelPacket":
        provenance = dict(self.provenance)
        provenance["reply_to"] = self.packet_id
        return KernelPacket(
            packet_kind=self.packet_kind,
            source_kernel_id=source_kernel_id,
            target_kernel_id=self.source_kernel_id,
            parent_goal_id=self.parent_goal_id,
            mission_context=self.mission_context,
            payload=dict(payload or {}),
            required_action=required_action,
            priority=self.priority,
            trust_level=trust_level,
            risk_class=risk_class,
            provenance=provenance,
            reply_expected=reply_expected,
        )


def make_packet(
    packet_kind: str,
    *,
    source_kernel_id: str,
    target_kernel_id: str,
    payload: Optional[MutableMapping[str, Any]] = None,
    required_action: str = "observe",
    parent_goal_id: str = "",
    mission_context: str = "",
    priority: int = 0,
    trust_level: str = TrustLevel.PROVISIONAL.value,
    risk_class: str = RiskClass.MEDIUM.value,
    provenance: Optional[MutableMapping[str, Any]] = None,
    reply_expected: bool = False,
) -> KernelPacket:
    return KernelPacket(
        packet_kind=packet_kind,
        source_kernel_id=source_kernel_id,
        target_kernel_id=target_kernel_id,
        parent_goal_id=parent_goal_id,
        mission_context=mission_context,
        payload=dict(payload or {}),
        required_action=required_action,
        priority=priority,
        trust_level=trust_level,
        risk_class=risk_class,
        provenance=dict(provenance or {}),
        reply_expected=reply_expected,
    )
