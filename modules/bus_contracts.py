"""
TOVAH v14 modules/bus_contracts.py — Message bus contracts.

Still does NOT implement networking or distributed execution.
It now tracks routes, proposal bindings, and recorded module messages so the
kernel ecology can reason about inter-module traffic without improvising dicts.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from tovah_v14.debug.trace_writer import evict_records
from tovah_v14.modules.interfaces import ModuleRequest, ModuleResponse


@dataclass
class BusRoute:
    """Declares that a role handles a specific action."""

    role: str
    action: str
    handler_name: str  # method name on the module


@dataclass
class BusMessage:
    """Recorded module-level bus traffic for observability and governance."""

    from_role: str
    to_role: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    priority: int = 0
    kind: str = "request"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MessageBusContract:
    """Contract for a future message bus.

    Currently: route registry + proposal binding + message log.
    Future: becomes an actual dispatch layer.
    """

    def __init__(self) -> None:
        self.routes: List[BusRoute] = []
        self.message_log: List[BusMessage] = []
        self.proposal_routes: Dict[str, Dict[str, Any]] = {}

    def register(self, role: str, action: str, handler_name: str) -> None:
        self.routes.append(BusRoute(role=role, action=action, handler_name=handler_name))

    def bind_proposal(self, proposal_id: str, *, target_role: str, handler_name: str = "review_module_proposal") -> Dict[str, Any]:
        binding = {
            "proposal_id": proposal_id,
            "target_role": target_role,
            "action": "review_module_proposal",
            "handler_name": handler_name,
            "bound_at": time.time(),
        }
        self.proposal_routes[proposal_id] = binding
        self.register(target_role, "review_module_proposal", handler_name)
        return dict(binding)

    def lookup(self, action: str) -> List[BusRoute]:
        return [r for r in self.routes if r.action == action]

    def all_actions(self) -> List[str]:
        return sorted(set(r.action for r in self.routes))

    def record_request(self, request: ModuleRequest, kind: str = "request") -> BusMessage:
        msg = BusMessage(
            from_role=request.from_role,
            to_role=request.to_role,
            action=request.action,
            payload=dict(request.payload),
            trace_id=request.trace_id,
            priority=request.priority,
            kind=kind,
        )
        self.message_log.append(msg)
        # AUDIT FIX (v14.2.7, sec 4): persist-on-evict.
        if len(self.message_log) > 250:  # cap=200 + cushion=50; batch evict for I/O efficiency
            evict_records("module_bus_log", self.message_log[:-200])
            self.message_log = self.message_log[-200:]
        return msg

    def record_response(self, response: ModuleResponse, *, to_role: str = "") -> BusMessage:
        msg = BusMessage(
            from_role=response.from_role,
            to_role=to_role,
            action="response",
            payload={"ok": response.ok, **dict(response.payload), "error": response.error},
            trace_id=response.trace_id,
            kind="response",
        )
        self.message_log.append(msg)
        if len(self.message_log) > 250:  # cap=200 + cushion=50; batch evict for I/O efficiency
            evict_records("module_bus_log", self.message_log[:-200])
            self.message_log = self.message_log[-200:]
        return msg

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.message_log[-max(1, limit):]]


    def export_state(self) -> Dict[str, Any]:
        return {
            "routes": [asdict(r) for r in self.routes],
            "message_log": [m.to_dict() for m in self.message_log],
            "proposal_routes": {k: dict(v) for k, v in self.proposal_routes.items()},
        }

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        self.routes = []
        self.message_log = []
        self.proposal_routes = {}
        if not isinstance(data, dict):
            return
        for route in list(data.get("routes", [])):
            try:
                self.routes.append(BusRoute(**dict(route)))
            except Exception:
                pass
        for message in list(data.get("message_log", [])):
            try:
                self.message_log.append(BusMessage(**dict(message)))
            except Exception:
                pass
        self.proposal_routes = {str(k): dict(v) for k, v in dict(data.get("proposal_routes", {})).items()}

    def summary(self) -> Dict[str, Any]:
        return {
            "route_count": len(self.routes),
            "proposal_route_count": len(self.proposal_routes),
            "actions": self.all_actions(),
            "recent_messages": self.recent(10),
        }
