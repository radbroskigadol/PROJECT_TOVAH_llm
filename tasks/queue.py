"""
TOVAH v14 tasks/queue.py — Task queue management.

SEMANTIC PRESERVATION: TaskNode shape preserved from v13.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.kernel.action_model import GoalLineage
from tovah_v14.config.constants import MAX_TASK_QUEUE
from tovah_v14.core.primitives import BilateralValue, bilateral_or, bilateral_recover
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.state import ShadowState


@dataclass
class TaskNode:
    """Structured task. Shape preserved from v13, extended with ecology lineage."""
    task_id: str
    goal: str
    status: str = "pending"  # pending, active, completed, failed, stale, suspended
    deps: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    staleness_seconds: float = 7200.0
    success_criteria: str = ""
    retries: int = 0
    max_retries: int = 3
    backoff_seconds: float = 60.0
    last_attempt: float = 0.0
    result: Optional[Dict[str, Any]] = None
    bilateral_confidence: BilateralValue = field(default_factory=lambda: BilateralValue(0.5, 0.1))
    owner_kernel_id: str = "main"
    requester_kernel_id: str = "main"
    root_goal_id: str = ""
    mission_context: str = ""
    lease_scope: str = "local"
    delegated_to_kernel_id: str = ""
    packet_id: str = ""
    lineage: Optional[GoalLineage] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """Manages task lifecycle with bilateral tracking."""

    def __init__(self) -> None:
        self.tasks: List[TaskNode] = []
        self.completed_ids: List[str] = []

    def create(
        self,
        goal: str,
        deps: Optional[List[str]] = None,
        deadline: Optional[float] = None,
        criteria: str = "",
        parent_id: Optional[str] = None,
        *,
        owner_kernel_id: str = "main",
        requester_kernel_id: str = "main",
        root_goal_id: str = "",
        mission_context: str = "",
        lease_scope: str = "local",
        delegated_to_kernel_id: str = "",
        packet_id: str = "",
        lineage: Optional[GoalLineage] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new task. Returns task_id."""
        tid = f"task_{int(time.time())}_{hashlib.md5(goal.encode()).hexdigest()[:6]}"
        if lineage is None:
            lineage = GoalLineage(
                goal_id=tid,
                parent_goal_id=parent_id or "",
                root_goal_id=root_goal_id or tid,
                mission_context=mission_context,
                owner_kernel_id=owner_kernel_id,
                requester_kernel_id=requester_kernel_id,
                lease_scope=lease_scope,
                provenance=[packet_id] if packet_id else [],
            )
        task = TaskNode(
            task_id=tid, goal=goal, deps=deps or [],
            deadline=deadline, success_criteria=criteria or f"Complete: {goal[:60]}",
            parent_id=parent_id, owner_kernel_id=owner_kernel_id,
            requester_kernel_id=requester_kernel_id, root_goal_id=root_goal_id or tid,
            mission_context=mission_context, lease_scope=lease_scope,
            delegated_to_kernel_id=delegated_to_kernel_id, packet_id=packet_id,
            lineage=lineage, provenance=dict(provenance or {}),
        )
        self.tasks.append(task)
        self.tasks = self.tasks[-MAX_TASK_QUEUE:]
        return tid

    def advance(self, state: Optional[ShadowState] = None) -> List[str]:
        """Advance ready tasks. Returns status messages.

        - pending → active when all deps are completed
        - active → stale when past deadline
        - active → failed when retries exhausted
        """
        msgs: List[str] = []
        now = time.time()
        all_completed = set(self.completed_ids) | {t.task_id for t in self.tasks if t.status == "completed"}

        for task in self.tasks:
            if task.status == "pending":
                if all(d in all_completed for d in task.deps):
                    task.status = "active"
                    msgs.append(f"task {task.task_id} activated")
            elif task.status == "active":
                if task.deadline and now > task.deadline:
                    task.status = "stale"
                    msgs.append(f"task {task.task_id} stale (deadline)")
                elif now - task.created_at > task.staleness_seconds and task.retries >= task.max_retries:
                    task.status = "failed"
                    msgs.append(f"task {task.task_id} failed (max retries)")

        if state is not None:
            active_count = sum(1 for t in self.tasks if t.status in ("pending", "active"))
            state.beta["task.queue_health"] = bilateral_recover(
                state.beta.get("task.queue_health", BilateralValue(0.5, 0.2)),
                truth_gain=0.08 if active_count < 10 else 0.02,
                falsity_decay=0.03,
            )
            refresh_state(state)

        return msgs

    def complete(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark a task as completed."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = "completed"
                task.result = result
                task.bilateral_confidence = bilateral_recover(
                    task.bilateral_confidence, truth_gain=0.3, falsity_decay=0.1,
                )
                self.completed_ids.append(task_id)
                return True
        return False

    def get_delegated(self, kernel_id: str = "") -> List[TaskNode]:
        """Return tasks delegated to a child kernel, optionally filtered by kernel id."""
        if kernel_id:
            return [t for t in self.tasks if t.delegated_to_kernel_id == kernel_id]
        return [t for t in self.tasks if t.delegated_to_kernel_id]

    def get_active(self) -> List[TaskNode]:
        return [t for t in self.tasks if t.status == "active"]

    def get_by_id(self, task_id: str) -> Optional[TaskNode]:
        return next((t for t in self.tasks if t.task_id == task_id), None)
