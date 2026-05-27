"""
TOVAH v14 tasks/delegation.py — Governed delegation leases for hub/subkernel work.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.kernel.action_model import GoalLineage


def _lease_id() -> str:
    return f"lease_{uuid.uuid4().hex[:12]}"


@dataclass
class DelegationLease:
    lease_id: str = field(default_factory=_lease_id)
    goal_id: str = ""
    parent_goal_id: str = ""
    root_goal_id: str = ""
    source_kernel_id: str = "main"
    target_kernel_id: str = ""
    target_node_id: str = ""
    mission_context: str = ""
    lease_scope: str = "delegated"
    required_trust_level: str = "low"
    assigned_trust_level: str = ""
    target_locality: str = "local"
    route_kind: str = "local_child"
    worker_role: str = "subkernel"
    allowed_tool_permissions: List[str] = field(default_factory=list)
    allowed_promotion_targets: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    completed_at: Optional[float] = None
    packet_id: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)

    def to_lineage(self) -> GoalLineage:
        return GoalLineage(
            goal_id=self.goal_id,
            parent_goal_id=self.parent_goal_id,
            root_goal_id=self.root_goal_id or self.goal_id,
            mission_context=self.mission_context,
            owner_kernel_id=self.target_kernel_id or self.source_kernel_id,
            requester_kernel_id=self.source_kernel_id,
            lease_scope=self.lease_scope,
            lease_expires_at=self.expires_at,
            provenance=[self.packet_id] if self.packet_id else [],
        )


class DelegationManager:
    def __init__(self) -> None:
        self.leases: Dict[str, DelegationLease] = {}
        self.history: List[Dict[str, Any]] = []

    def issue(
        self,
        *,
        goal_id: str,
        source_kernel_id: str,
        target_kernel_id: str,
        mission_context: str = "",
        parent_goal_id: str = "",
        root_goal_id: str = "",
        lease_scope: str = "delegated",
        expires_at: Optional[float] = None,
        packet_id: str = "",
        target_node_id: str = "",
        required_trust_level: str = "low",
        assigned_trust_level: str = "",
        target_locality: str = "local",
        route_kind: str = "local_child",
        worker_role: str = "subkernel",
        allowed_tool_permissions: Optional[List[str]] = None,
        allowed_promotion_targets: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> DelegationLease:
        lease = DelegationLease(
            goal_id=goal_id,
            parent_goal_id=parent_goal_id,
            root_goal_id=root_goal_id or goal_id,
            source_kernel_id=source_kernel_id,
            target_kernel_id=target_kernel_id,
            mission_context=mission_context,
            lease_scope=lease_scope,
            required_trust_level=required_trust_level,
            assigned_trust_level=assigned_trust_level,
            target_node_id=target_node_id,
            target_locality=target_locality,
            route_kind=route_kind,
            worker_role=worker_role,
            allowed_tool_permissions=list(allowed_tool_permissions or []),
            allowed_promotion_targets=list(allowed_promotion_targets or []),
            expires_at=expires_at,
            packet_id=packet_id,
            provenance=dict(provenance or {}),
        )
        self.leases[lease.lease_id] = lease
        self.history.append({
            "event": "issued",
            "lease_id": lease.lease_id,
            "goal_id": goal_id,
            "time": time.time(),
            "target_kernel_id": target_kernel_id,
            "worker_role": worker_role,
        })
        self.history = self.history[-200:]
        return lease

    def complete(self, lease_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        lease = self.leases.get(lease_id)
        if lease is None:
            return False
        lease.status = "completed"
        lease.completed_at = time.time()
        lease.outcome = {"success": True, **dict(result or {})}
        self.history.append({"event": "completed", "lease_id": lease_id, "goal_id": lease.goal_id, "time": lease.completed_at, "result": dict(result or {})})
        self.history = self.history[-200:]
        return True

    def fail(self, lease_id: str, reason: str = "", result: Optional[Dict[str, Any]] = None) -> bool:
        lease = self.leases.get(lease_id)
        if lease is None:
            return False
        lease.status = "failed"
        lease.completed_at = time.time()
        lease.outcome = {"success": False, "reason": reason, **dict(result or {})}
        self.history.append({"event": "failed", "lease_id": lease_id, "goal_id": lease.goal_id, "time": lease.completed_at, "reason": reason, "result": dict(result or {})})
        self.history = self.history[-200:]
        return True

    def revoke(self, lease_id: str, reason: str = "") -> bool:
        lease = self.leases.get(lease_id)
        if lease is None:
            return False
        lease.status = "revoked"
        lease.completed_at = time.time()
        lease.outcome = {"success": False, "reason": reason}
        self.history.append({"event": "revoked", "lease_id": lease_id, "goal_id": lease.goal_id, "time": lease.completed_at, "reason": reason})
        self.history = self.history[-200:]
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "leases": {k: v.__dict__ for k, v in self.leases.items()},
            "history": list(self.history),
        }

    def restore(self, data: Dict[str, Any] | None) -> None:
        self.leases = {}
        self.history = []
        if not isinstance(data, dict):
            return
        for lease_id, payload in dict(data.get("leases", {})).items():
            try:
                self.leases[str(lease_id)] = DelegationLease(**dict(payload))
            except Exception:
                pass
        self.history = list(data.get("history", []))[-200:]

    def list_active(self, target_kernel_id: str = "") -> List[DelegationLease]:
        leases = [l for l in self.leases.values() if l.status == "active"]
        if target_kernel_id:
            leases = [l for l in leases if l.target_kernel_id == target_kernel_id]
        return sorted(leases, key=lambda l: l.created_at)

    def summary(self) -> Dict[str, Any]:
        active = self.list_active()
        completed = [l for l in self.leases.values() if l.status == "completed"]
        failed = [l for l in self.leases.values() if l.status in {"failed", "revoked"}]
        by_kernel = {k: len([l for l in active if l.target_kernel_id == k]) for k in sorted({l.target_kernel_id for l in active if l.target_kernel_id})}
        by_role = {k: len([l for l in active if l.worker_role == k]) for k in sorted({l.worker_role for l in active if l.worker_role})}
        closed = completed + failed
        success_rate = len(completed) / max(1, len(closed)) if closed else 0.0
        return {
            "active_count": len(active),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "success_rate": success_rate,
            "active_by_kernel": by_kernel,
            "active_by_role": by_role,
            "recent_history": list(self.history[-10:]),
        }
