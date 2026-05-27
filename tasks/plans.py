"""
TOVAH v14 tasks/plans.py — Strategic plan management.

SEMANTIC PRESERVATION: StrategicPlan shape preserved from v13.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.core.primitives import BilateralValue, bilateral_recover, bilateral_or
from tovah_v14.kernel.action_model import GoalLineage
from tovah_v14.config.constants import PLAN_MAX_AGE_SECONDS


@dataclass
class StrategicPlan:
    """A strategic plan. Shape preserved from v13, extended with ecology lineage."""
    plan_id: str
    objective: str
    steps: List[Dict[str, Any]]
    created_at: str
    status: str = "active"
    current_step: int = 0
    bilateral_confidence: BilateralValue = field(default_factory=lambda: BilateralValue(0.5, 0.1))
    results: List[Dict[str, Any]] = field(default_factory=list)
    owner_kernel_id: str = "main"
    requester_kernel_id: str = "main"
    root_goal_id: str = ""
    mission_context: str = ""
    plan_kind: str = "local"
    delegated_to_kernel_id: str = ""
    packet_id: str = ""
    lineage: Optional[GoalLineage] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


class PlanManager:
    """Manages strategic plans with bilateral tracking."""

    def __init__(self) -> None:
        self.active: List[StrategicPlan] = []
        self.completed_ids: List[str] = []

    def add(self, plan: StrategicPlan) -> None:
        if plan.lineage is None:
            plan.lineage = GoalLineage(
                goal_id=plan.plan_id,
                root_goal_id=plan.root_goal_id or plan.plan_id,
                mission_context=plan.mission_context,
                owner_kernel_id=plan.owner_kernel_id,
                requester_kernel_id=plan.requester_kernel_id,
                lease_scope=plan.plan_kind,
                provenance=[plan.packet_id] if plan.packet_id else [],
            )
        self.active.append(plan)
        self.active = self.active[-20:]

    def get_delegated(self, kernel_id: str = "") -> List[StrategicPlan]:
        if kernel_id:
            return [p for p in self.active if p.status == "active" and p.delegated_to_kernel_id == kernel_id]
        return [p for p in self.active if p.status == "active" and p.delegated_to_kernel_id]

    def get_active(self) -> List[StrategicPlan]:
        return [p for p in self.active if p.status == "active"]

    def complete(self, plan_id: str) -> bool:
        for p in self.active:
            if p.plan_id == plan_id:
                p.status = "completed"
                self.completed_ids.append(plan_id)
                return True
        return False

    def cleanup_stale(self) -> List[str]:
        """Archive plans that have been active too long."""
        now = time.time()
        archived: List[str] = []
        for p in self.active:
            if p.status != "active":
                continue
            try:
                import datetime as dt
                created = dt.datetime.fromisoformat(p.created_at).timestamp()
            except Exception:
                created = now
            if now - created > PLAN_MAX_AGE_SECONDS:
                p.status = "stale"
                archived.append(p.plan_id)
        return archived
