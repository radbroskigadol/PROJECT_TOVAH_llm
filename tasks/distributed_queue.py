"""TOVAH v16 tasks/distributed_queue.py — local-first distributed-ready delegation queue.

This is not a network transport. It is a governed queue for cluster-aware work
assignment, lease routing, and observability that can later back real remote
execution without changing the task semantics.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _job_id() -> str:
    return f"djob_{uuid.uuid4().hex[:12]}"


@dataclass
class DistributedTaskRecord:
    job_id: str = field(default_factory=_job_id)
    goal: str = ""
    goal_id: str = ""
    source_kernel_id: str = "main"
    target_kernel_id: str = ""
    target_node_id: str = ""
    specialization: str = "general"
    mission_context: str = ""
    required_trust_level: str = "low"
    assigned_trust_level: str = ""
    route_kind: str = "local_child"
    target_worker_role: str = "subkernel"
    allowed_tool_permissions: List[str] = field(default_factory=list)
    allowed_promotion_targets: List[str] = field(default_factory=list)
    status: str = "queued"
    lease_id: str = ""
    packet_id: str = ""
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    failed_at: Optional[float] = None
    outcome: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedQueue:
    def __init__(self) -> None:
        self.records: Dict[str, DistributedTaskRecord] = {}
        self.history: List[Dict[str, Any]] = []

    def enqueue(self, *, goal: str, goal_id: str, source_kernel_id: str, specialization: str = "general", mission_context: str = "", required_trust_level: str = "low", packet_id: str = "", provenance: Dict[str, Any] | None = None) -> DistributedTaskRecord:
        rec = DistributedTaskRecord(
            goal=goal,
            goal_id=goal_id,
            source_kernel_id=source_kernel_id,
            specialization=specialization or "general",
            mission_context=mission_context,
            required_trust_level=required_trust_level or "low",
            packet_id=packet_id,
            provenance=dict(provenance or {}),
        )
        self.records[rec.job_id] = rec
        self.history.append({"event": "queued", "job_id": rec.job_id, "goal_id": goal_id, "time": rec.created_at, "specialization": rec.specialization})
        self.history = self.history[-500:]
        return rec

    def assign(self, job_id: str, *, target_kernel_id: str, target_node_id: str, lease_id: str = "", assigned_trust_level: str = "", route_kind: str = "local_child", target_worker_role: str = "subkernel", allowed_tool_permissions: Optional[List[str]] = None, allowed_promotion_targets: Optional[List[str]] = None) -> Dict[str, Any] | None:
        rec = self.records.get(job_id)
        if rec is None:
            return None
        rec.target_kernel_id = target_kernel_id
        rec.target_node_id = target_node_id
        rec.lease_id = lease_id
        rec.assigned_trust_level = assigned_trust_level
        rec.route_kind = route_kind
        rec.target_worker_role = target_worker_role
        rec.allowed_tool_permissions = list(allowed_tool_permissions or [])
        rec.allowed_promotion_targets = list(allowed_promotion_targets or [])
        rec.status = "assigned"
        rec.assigned_at = time.time()
        event = {"event": "assigned", "job_id": job_id, "goal_id": rec.goal_id, "target_kernel_id": target_kernel_id, "target_node_id": target_node_id, "time": rec.assigned_at, "route_kind": route_kind, "target_worker_role": target_worker_role}
        self.history.append(event)
        self.history = self.history[-500:]
        return dict(event)

    def complete(self, job_id: str, *, result: Dict[str, Any] | None = None) -> bool:
        rec = self.records.get(job_id)
        if rec is None:
            return False
        rec.status = "completed"
        rec.completed_at = time.time()
        rec.outcome = {"success": True, **dict(result or {})}
        self.history.append({"event": "completed", "job_id": job_id, "goal_id": rec.goal_id, "target_kernel_id": rec.target_kernel_id, "time": rec.completed_at, "result": dict(result or {})})
        self.history = self.history[-500:]
        return True

    def fail(self, job_id: str, *, reason: str = "", result: Dict[str, Any] | None = None) -> bool:
        rec = self.records.get(job_id)
        if rec is None:
            return False
        rec.status = "failed"
        rec.failed_at = time.time()
        rec.outcome = {"success": False, "reason": reason, **dict(result or {})}
        self.history.append({"event": "failed", "job_id": job_id, "goal_id": rec.goal_id, "target_kernel_id": rec.target_kernel_id, "time": rec.failed_at, "reason": reason, "result": dict(result or {})})
        self.history = self.history[-500:]
        return True

    def queued(self) -> List[DistributedTaskRecord]:
        return sorted([r for r in self.records.values() if r.status == "queued"], key=lambda r: r.created_at)

    def active(self) -> List[DistributedTaskRecord]:
        return sorted([r for r in self.records.values() if r.status == "assigned"], key=lambda r: r.assigned_at or r.created_at)

    def export_state(self) -> Dict[str, Any]:
        return {
            "records": {job_id: rec.to_dict() for job_id, rec in self.records.items()},
            "history": list(self.history[-500:]),
        }

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        self.records = {}
        self.history = []
        if not isinstance(data, dict):
            return
        for job_id, payload in dict(data.get("records", {})).items():
            try:
                self.records[str(job_id)] = DistributedTaskRecord(**dict(payload))
            except Exception:
                pass
        self.history = list(data.get("history", []))[-500:]

    def summary(self) -> Dict[str, Any]:
        queued = [r for r in self.records.values() if r.status == "queued"]
        active = [r for r in self.records.values() if r.status == "assigned"]
        completed = [r for r in self.records.values() if r.status == "completed"]
        failed = [r for r in self.records.values() if r.status == "failed"]
        by_specialization: Dict[str, int] = {}
        by_role: Dict[str, int] = {}
        for rec in self.records.values():
            by_specialization[rec.specialization] = by_specialization.get(rec.specialization, 0) + 1
            by_role[rec.target_worker_role] = by_role.get(rec.target_worker_role, 0) + 1
        closed = completed + failed
        success_rate = len(completed) / max(1, len(closed)) if closed else 0.0
        return {
            "queued_count": len(queued),
            "active_count": len(active),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "success_rate": success_rate,
            "specializations": by_specialization,
            "worker_roles": by_role,
            "recent_history": list(self.history[-10:]),
        }
