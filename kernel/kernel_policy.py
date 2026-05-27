"""
TOVAH v14 kernel/kernel_policy.py — Authority and trust boundaries.

Defines what main, hub, and subkernels are allowed to own or mutate.
This is an architectural contract; runtime enforcement can layer on later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from tovah_v14.kernel.kernel_roles import KernelRole, TrustLevel
from tovah_v14.kernel.packet import PacketKind


@dataclass(frozen=True)
class KernelCapabilityPolicy:
    role: KernelRole
    authoritative_state: bool
    may_directly_mutate_main: bool
    may_own_global_truth: bool
    may_rewrite_core_semantics: bool
    may_promote_to_main: bool
    may_spawn_children: bool
    may_request_resources: bool
    may_request_tools: bool
    default_trust: TrustLevel
    allowed_packet_kinds: List[str] = field(default_factory=list)
    notes: str = ""


DEFAULT_KERNEL_POLICIES: Dict[KernelRole, KernelCapabilityPolicy] = {
    KernelRole.MAIN: KernelCapabilityPolicy(
        role=KernelRole.MAIN,
        authoritative_state=True,
        may_directly_mutate_main=True,
        may_own_global_truth=True,
        may_rewrite_core_semantics=True,
        may_promote_to_main=True,
        may_spawn_children=True,
        may_request_resources=True,
        may_request_tools=True,
        default_trust=TrustLevel.SOVEREIGN,
        allowed_packet_kinds=[
            PacketKind.STATUS,
            PacketKind.GOAL,
            PacketKind.RESOURCE_REQUEST,
            PacketKind.TOOL_REQUEST,
            PacketKind.MODULE_PROPOSAL,
            PacketKind.PATCH_PROPOSAL,
            PacketKind.PROMOTION_REQUEST,
            PacketKind.BLOCKED_GROWTH,
            PacketKind.SPAWN_REQUEST,
            PacketKind.TRUST_REPORT,
            PacketKind.REVERT_NOTICE,
            PacketKind.MEMORY_SYNC,
        ],
        notes="Sovereign kernel owns canonical state and final approval.",
    ),
    KernelRole.HUB: KernelCapabilityPolicy(
        role=KernelRole.HUB,
        authoritative_state=False,
        may_directly_mutate_main=False,
        may_own_global_truth=False,
        may_rewrite_core_semantics=False,
        may_promote_to_main=False,
        may_spawn_children=True,
        may_request_resources=True,
        may_request_tools=True,
        default_trust=TrustLevel.PROVISIONAL,
        allowed_packet_kinds=[
            PacketKind.STATUS,
            PacketKind.GOAL,
            PacketKind.RESOURCE_REQUEST,
            PacketKind.TOOL_REQUEST,
            PacketKind.MODULE_PROPOSAL,
            PacketKind.PATCH_PROPOSAL,
            PacketKind.PROMOTION_REQUEST,
            PacketKind.BLOCKED_GROWTH,
            PacketKind.SPAWN_REQUEST,
            PacketKind.TRUST_REPORT,
            PacketKind.REVERT_NOTICE,
            PacketKind.MEMORY_SYNC,
        ],
        notes="Experimental layer may branch, rehearse, and propose but not directly overwrite main.",
    ),
    KernelRole.SUBKERNEL: KernelCapabilityPolicy(
        role=KernelRole.SUBKERNEL,
        authoritative_state=False,
        may_directly_mutate_main=False,
        may_own_global_truth=False,
        may_rewrite_core_semantics=False,
        may_promote_to_main=False,
        may_spawn_children=False,
        may_request_resources=True,
        may_request_tools=True,
        default_trust=TrustLevel.LOW,
        allowed_packet_kinds=[
            PacketKind.STATUS,
            PacketKind.GOAL,
            PacketKind.RESOURCE_REQUEST,
            PacketKind.TOOL_REQUEST,
            PacketKind.MODULE_PROPOSAL,
            PacketKind.PATCH_PROPOSAL,
            PacketKind.PROMOTION_REQUEST,
            PacketKind.BLOCKED_GROWTH,
            PacketKind.TRUST_REPORT,
            PacketKind.REVERT_NOTICE,
            PacketKind.MEMORY_SYNC,
        ],
        notes="Specialized child kernels execute narrow missions and report upward.",
    ),
}


def get_policy(role: KernelRole | str) -> KernelCapabilityPolicy:
    if isinstance(role, str):
        role = KernelRole(role)
    return DEFAULT_KERNEL_POLICIES[role]


def role_can_send(role: KernelRole | str, packet_kind: str) -> bool:
    return packet_kind in get_policy(role).allowed_packet_kinds


def can_directly_mutate(source_role: KernelRole | str, target_role: KernelRole | str) -> bool:
    source = get_policy(source_role)
    if isinstance(target_role, str):
        target_role = KernelRole(target_role)
    if target_role == KernelRole.MAIN:
        return source.may_directly_mutate_main
    return source.role == target_role or source.role == KernelRole.MAIN


def requires_approval_for_promotion(source_role: KernelRole | str, target_role: KernelRole | str) -> bool:
    if isinstance(source_role, str):
        source_role = KernelRole(source_role)
    if isinstance(target_role, str):
        target_role = KernelRole(target_role)
    return not (source_role == KernelRole.MAIN and target_role == KernelRole.MAIN)


def ownership_summary() -> Dict[str, Dict[str, bool | str]]:
    summary: Dict[str, Dict[str, bool | str]] = {}
    for role, policy in DEFAULT_KERNEL_POLICIES.items():
        summary[role.value] = {
            "authoritative_state": policy.authoritative_state,
            "may_directly_mutate_main": policy.may_directly_mutate_main,
            "may_own_global_truth": policy.may_own_global_truth,
            "may_rewrite_core_semantics": policy.may_rewrite_core_semantics,
            "may_promote_to_main": policy.may_promote_to_main,
            "may_spawn_children": policy.may_spawn_children,
            "default_trust": policy.default_trust.value,
        }
    return summary
