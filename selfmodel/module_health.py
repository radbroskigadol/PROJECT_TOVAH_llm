"""
TOVAH v14 selfmodel/module_health.py — Module-role health tracking.

Each module role (planner, executor, critic, etc.) has bilateral health.
This tracks reliability per role for self-model-driven goal selection.
"""
from __future__ import annotations

from typing import Dict

from tovah_v14.core.primitives import BilateralValue, bilateral_recover, bilateral_or
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.state import ShadowState


# Module role → beta key mapping
MODULE_HEALTH_KEYS: Dict[str, str] = {
    "planner": "module.planner_health",
    "executor": "module.executor_health",
    "critic": "module.critic_health",
    "memory_manager": "module.memory_health",
    "trainer": "module.trainer_health",
    "retriever": "module.retriever_health",
    "patcher": "module.patcher_health",
    "observer": "module.observer_health",
}


class ModuleHealthTracker:
    """Tracks bilateral health for each module role."""

    def record_success(self, role: str, state: ShadowState, gain: float = 0.10) -> None:
        """Record a successful operation for a module role."""
        key = MODULE_HEALTH_KEYS.get(role)
        if key and key in state.beta:
            state.beta[key] = bilateral_recover(
                state.beta.get(key, BilateralValue(0.5, 0.2)),
                truth_gain=gain, falsity_decay=0.04,
            )
            refresh_state(state)

    def record_failure(self, role: str, state: ShadowState, penalty: float = 0.12) -> None:
        """Record a failure for a module role."""
        key = MODULE_HEALTH_KEYS.get(role)
        if key and key in state.beta:
            state.beta[key] = bilateral_or(
                state.beta.get(key, BilateralValue(0.5, 0.2)),
                BilateralValue(0.0, penalty),
            )
            refresh_state(state)

    def get_health_summary(self, state: ShadowState) -> Dict[str, Dict[str, float]]:
        """Get bilateral health for all module roles."""
        summary: Dict[str, Dict[str, float]] = {}
        for role, key in MODULE_HEALTH_KEYS.items():
            bv = state.beta.get(key, BilateralValue(0.5, 0.2))
            summary[role] = {"t": bv.t, "f": bv.f, "delta": bv.delta, "glut": bv.glut}
        return summary

    def weakest_modules(self, state: ShadowState, limit: int = 3) -> list:
        """Get module roles with lowest truth support."""
        pairs = []
        for role, key in MODULE_HEALTH_KEYS.items():
            bv = state.beta.get(key, BilateralValue(0.5, 0.2))
            pairs.append((bv.t, role))
        pairs.sort()
        return [role for _, role in pairs[:limit]]
