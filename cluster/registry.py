"""TOVAH v16 cluster/registry.py — registry of local/distributed kernel nodes."""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Iterable, List

from .node import ClusterNodeRecord


class ClusterRegistry:
    def __init__(self) -> None:
        self.nodes: Dict[str, ClusterNodeRecord] = {}
        self.events: List[Dict[str, Any]] = []

    def register_node(self, record: ClusterNodeRecord | Dict[str, Any]) -> ClusterNodeRecord:
        node = record if isinstance(record, ClusterNodeRecord) else ClusterNodeRecord(**dict(record))
        node.last_seen = time.time()
        self.nodes[node.node_id] = node
        self.events.append({"kind": "register", "node_id": node.node_id, "timestamp": node.last_seen})
        self.events = self.events[-500:]
        return node

    def upsert_node(self, node_id: str, **updates: Any) -> ClusterNodeRecord:
        node = self.nodes.get(node_id)
        if node is None:
            node = ClusterNodeRecord(node_id=node_id, kernel_id=updates.get("kernel_id", node_id), role=updates.get("role", "subkernel"))
            self.nodes[node_id] = node
        node.touch(**updates)
        self.events.append({"kind": "heartbeat", "node_id": node.node_id, "timestamp": node.last_seen, "updates": dict(updates)})
        self.events = self.events[-500:]
        return node

    def remove_node(self, node_id: str, *, reason: str = "retired") -> None:
        node = self.nodes.pop(node_id, None)
        if node is not None:
            self.events.append({"kind": "remove", "node_id": node_id, "timestamp": time.time(), "reason": reason})
            self.events = self.events[-500:]

    def get(self, node_id: str) -> ClusterNodeRecord | None:
        return self.nodes.get(node_id)

    def list_nodes(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in sorted(self.nodes.values(), key=lambda n: (n.role, n.node_id))]

    def list_by_role(self, role: str) -> List[ClusterNodeRecord]:
        return sorted([n for n in self.nodes.values() if n.role == role], key=lambda n: n.node_id)

    def eligible_nodes(self, *, role: str = "", specialization: str = "", locality: str = "") -> List[ClusterNodeRecord]:
        nodes = list(self.nodes.values())
        if role:
            nodes = [n for n in nodes if n.role == role]
        if specialization:
            nodes = [n for n in nodes if not n.specialization or n.specialization == specialization]
        if locality:
            nodes = [n for n in nodes if n.locality == locality]
        return sorted(nodes, key=lambda n: (n.role, n.specialization, n.node_id))

    def summary(self) -> Dict[str, Any]:
        import time
        nodes = list(self.nodes.values())
        local_count = sum(1 for n in nodes if n.locality == "local")
        trusted_count = sum(1 for n in nodes if n.trust_level in {"trusted", "sovereign"})
        worker_roles = {role: sum(1 for n in nodes if str(n.metadata.get("worker_role", n.role)) == role) for role in sorted({str(n.metadata.get("worker_role", n.role)) for n in nodes} | {"main", "hub", "subkernel"})}
        policy_ready_count = sum(1 for n in nodes if n.metadata.get("allowed_tool_permissions") or n.metadata.get("allowed_promotion_targets"))
        cooldown_node_count = sum(1 for n in nodes if float(n.metadata.get("cooldown_until", 0.0) or 0.0) > time.time())
        avg_maturity_bonus = (sum(float(n.metadata.get("maturity_bonus", 0.0) or 0.0) for n in nodes) / len(nodes)) if nodes else 0.0
        return {
            "node_count": len(nodes),
            "local_node_count": local_count,
            "trusted_node_count": trusted_count,
            "remote_ready_count": sum(1 for n in nodes if n.locality != "local"),
            "roles": {role: sum(1 for n in nodes if n.role == role) for role in sorted({n.role for n in nodes} | {"main", "hub", "subkernel"})},
            "worker_roles": worker_roles,
            "policy_ready_count": policy_ready_count,
            "cooldown_node_count": cooldown_node_count,
            "average_maturity_bonus": avg_maturity_bonus,
            "specializations": {spec: sum(1 for n in nodes if n.specialization == spec) for spec in sorted({n.specialization for n in nodes if n.specialization})},
            "recent_events": self.events[-10:],
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "events": copy.deepcopy(self.events[-500:]),
        }

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        self.nodes = {}
        self.events = []
        if not isinstance(data, dict):
            return
        for node_id, payload in dict(data.get("nodes", {})).items():
            try:
                node = ClusterNodeRecord.from_dict(payload)
                if node is not None:
                    self.nodes[node_id] = node
            except Exception:
                continue
        self.events = list(data.get("events", []))[-500:]
