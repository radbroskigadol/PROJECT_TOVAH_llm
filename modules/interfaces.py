"""
TOVAH v14 modules/interfaces.py — Interface contracts.

Typed message contracts for inter-module communication.
Currently used as documentation of boundaries.
Future: becomes the actual message bus types.

We do NOT implement fake networking or distributed execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModuleRequest:
    """Request from one module to another."""
    from_role: str
    to_role: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    priority: int = 0  # 0=normal, 1=high, 2=critical


@dataclass
class ModuleResponse:
    """Response from a module."""
    from_role: str
    ok: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    trace_id: str = ""


@dataclass
class TaskLease:
    """Lease contract for a module claiming a task.
    Future: prevents two modules from working the same task.
    """
    task_id: str
    leased_by: str
    leased_at: float = 0.0
    expires_at: float = 0.0
    status: str = "active"  # active, completed, expired, cancelled
