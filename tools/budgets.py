"""
TOVAH v14 tools/budgets.py — Resource budget management.

Enforces rate limits on tool usage, advisor calls, pip installs, etc.
Budget state is owned by the kernel and passed to BudgetManager.

This is a v14 improvement: v13 had budget checking scattered through
the kernel. Now it is centralized and must be called before dispatch.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict

from tovah_v14.config.constants import DEFAULT_BUDGETS, BUDGET_RESET_INTERVAL
from tovah_v14.core.primitives import BilateralValue, bilateral_or
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.state import ShadowState


class BudgetManager:
    """Manages resource budgets with periodic reset.

    Budget state is a Dict[str, Dict[str, Any]] with:
      {resource_name: {"limit": int, "used": int, "reset_at": float}}

    v16 extension:
      - worker-local tool budgets by permission level
      - lease-exhaustion checks for delegated workers
      - lightweight recovery when delegated work completes successfully
    """

    def __init__(self, budgets: Dict[str, Dict[str, Any]] | None = None):
        self.budgets: Dict[str, Dict[str, Any]] = budgets or copy.deepcopy(DEFAULT_BUDGETS)
        self.worker_usage: Dict[str, Dict[str, int]] = {}
        self.worker_quota_overrides: Dict[str, Dict[str, int]] = {}

    def check(self, resource: str, cost: int = 1) -> bool:
        b = self.budgets.get(resource)
        if b is None:
            return True
        return b["used"] + cost <= b["limit"]

    def spend(self, resource: str, cost: int = 1) -> bool:
        b = self.budgets.get(resource)
        if b is None:
            return True
        if b["used"] + cost > b["limit"]:
            return False
        b["used"] += cost
        return True

    def reset_if_needed(self) -> None:
        now = time.time()
        for _name, b in self.budgets.items():
            if now - b.get("reset_at", 0.0) > BUDGET_RESET_INTERVAL:
                b["used"] = 0
                b["reset_at"] = now

    def usage_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, b in self.budgets.items():
            limit = max(1, b.get("limit", 1))
            summary[name] = b.get("used", 0) / limit
        return summary

    def _worker_quota(self, role: str, permission: str) -> int:
        role = str(role or "subkernel")
        permission = str(permission or "safe_autonomous")
        override = self.worker_quota_overrides.get(role, {}).get(permission)
        if override is not None:
            return max(0, int(override))
        defaults = {
            "main": {
                "safe_autonomous": 64,
                "safe_logged": 24,
                "sandbox_only": 12,
                "approval_required": 6,
            },
            "hub": {
                "safe_autonomous": 20,
                "safe_logged": 8,
                "sandbox_only": 0,
                "approval_required": 0,
            },
            "subkernel": {
                "safe_autonomous": 10,
                "safe_logged": 0,
                "sandbox_only": 0,
                "approval_required": 0,
            },
        }
        return int(defaults.get(role, defaults["subkernel"]).get(permission, 0))

    def check_worker_request(
        self,
        worker_id: str,
        *,
        role: str,
        permission: str,
        active_leases: int = 0,
        max_active_leases: int = 3,
    ) -> Dict[str, Any]:
        worker_id = str(worker_id or "unknown")
        permission = str(permission or "safe_autonomous")
        role = str(role or "subkernel")
        quota = self._worker_quota(role, permission)
        used = int(self.worker_usage.get(worker_id, {}).get(permission, 0))
        allowed = True
        reason = "budget_ok"
        if max_active_leases > 0 and active_leases >= max_active_leases:
            allowed = False
            reason = "lease_capacity_exhausted"
        elif quota <= 0:
            allowed = False
            reason = "permission_budget_unavailable"
        elif used + 1 > quota:
            allowed = False
            reason = "worker_budget_exhausted"
        return {
            "worker_id": worker_id,
            "role": role,
            "permission": permission,
            "used": used,
            "limit": quota,
            "active_leases": int(active_leases),
            "max_active_leases": int(max_active_leases),
            "allowed": allowed,
            "reason": reason,
        }

    def spend_worker_request(self, worker_id: str, *, role: str, permission: str) -> bool:
        decision = self.check_worker_request(worker_id, role=role, permission=permission)
        if not decision["allowed"]:
            return False
        usage = self.worker_usage.setdefault(str(worker_id or "unknown"), {})
        permission = str(permission or "safe_autonomous")
        usage[permission] = int(usage.get(permission, 0)) + 1
        return True

    def recover_worker_request(self, worker_id: str, *, permission: str, amount: int = 1) -> Dict[str, Any]:
        worker_id = str(worker_id or "unknown")
        permission = str(permission or "safe_autonomous")
        amount = max(1, int(amount))
        usage = self.worker_usage.setdefault(worker_id, {})
        before = int(usage.get(permission, 0))
        usage[permission] = max(0, before - amount)
        return {
            "worker_id": worker_id,
            "permission": permission,
            "before": before,
            "after": int(usage.get(permission, 0)),
            "recovered": before - int(usage.get(permission, 0)),
        }

    def worker_usage_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for worker_id, usage in self.worker_usage.items():
            pressure: Dict[str, Any] = {}
            for perm, count in usage.items():
                limit_guess = max(1, int(count))
                pressure[perm] = {
                    "used": int(count),
                    "limit_hint": limit_guess,
                    "pressure": float(count) / float(limit_guess),
                }
            summary[worker_id] = pressure
        return summary

    def export_state(self) -> Dict[str, Any]:
        return {
            "budgets": copy.deepcopy(self.budgets),
            "worker_usage": copy.deepcopy(self.worker_usage),
            "worker_quota_overrides": copy.deepcopy(self.worker_quota_overrides),
        }

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        if not isinstance(data, dict):
            return
        budgets = data.get("budgets")
        if isinstance(budgets, dict):
            self.budgets = copy.deepcopy(budgets)
        self.worker_usage = {str(k): {str(pk): int(pv) for pk, pv in dict(v).items()} for k, v in dict(data.get("worker_usage", {})).items()}
        self.worker_quota_overrides = {str(k): {str(pk): int(pv) for pk, pv in dict(v).items()} for k, v in dict(data.get("worker_quota_overrides", {})).items()}

    def update_bilateral_state(self, state: ShadowState) -> None:
        over = any(
            b.get("used", 0) > b.get("limit", 999)
            for b in self.budgets.values()
        )
        if over:
            state.beta["budget.compliance"] = bilateral_or(
                state.beta.get("budget.compliance", BilateralValue(0.5, 0.2)),
                BilateralValue(0.0, 0.15),
            )
        else:
            from tovah_v14.core.primitives import bilateral_recover
            state.beta["budget.compliance"] = bilateral_recover(
                state.beta.get("budget.compliance", BilateralValue(0.5, 0.2)),
                truth_gain=0.05, falsity_decay=0.03,
            )
        refresh_state(state)
