"""
TOVAH v14 modules/roles.py — Module role definitions.

Each kernel subsystem has a role. Each role has a bilateral health key.
This is the foundation for module-level health tracking and
future distribution.
"""
from __future__ import annotations

import enum
from typing import Dict


class ModuleRole(enum.Enum):
    """Kernel subsystem roles."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    MEMORY_MANAGER = "memory_manager"
    TRAINER = "trainer"
    RETRIEVER = "retriever"
    PATCHER = "patcher"
    OBSERVER = "observer"


# Role → beta key for bilateral health tracking
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
