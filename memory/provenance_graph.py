"""
TOVAH v14 memory/provenance_graph.py — Branch-memory provenance graph.

Tracks where branch memory came from, how it moved, and what packets or sync
operations were responsible. This stays additive and does not alter the core
ShadowHoTT state model.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _node_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ProvenanceNode:
    node_id: str
    kind: str
    label: str = ""
    owner_kernel_id: str = ""
    branch_id: str = ""
    created_at: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceEdge:
    source_node_id: str
    target_node_id: str
    relation: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProvenanceGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        self.branch_index: Dict[str, List[str]] = {}
        self.kernel_index: Dict[str, List[str]] = {}

    def add_node(
        self,
        kind: str,
        *,
        label: str = "",
        owner_kernel_id: str = "",
        branch_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
        node_id: str = "",
    ) -> str:
        node_id = node_id or _node_id(kind)
        node = ProvenanceNode(
            node_id=node_id,
            kind=kind,
            label=label,
            owner_kernel_id=owner_kernel_id,
            branch_id=branch_id,
            payload=dict(payload or {}),
        )
        self.nodes[node_id] = node
        if branch_id:
            self.branch_index.setdefault(branch_id, []).append(node_id)
            self.branch_index[branch_id] = self.branch_index[branch_id][-500:]
        if owner_kernel_id:
            self.kernel_index.setdefault(owner_kernel_id, []).append(node_id)
            self.kernel_index[owner_kernel_id] = self.kernel_index[owner_kernel_id][-500:]
        return node_id

    def link(self, source_node_id: str, target_node_id: str, relation: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        edge = ProvenanceEdge(source_node_id=source_node_id, target_node_id=target_node_id, relation=relation, metadata=dict(metadata or {}))
        self.edges.append(edge)
        self.edges = self.edges[-1000:]
        return edge.to_dict()

    def record_packet(self, packet: Dict[str, Any]) -> str:
        packet_id = str(packet.get("packet_id") or _node_id("packet"))
        return self.add_node(
            "packet",
            label=str(packet.get("packet_kind", "packet")),
            owner_kernel_id=str(packet.get("source_kernel_id", "")),
            branch_id=str(packet.get("source_kernel_id", "")),
            payload=dict(packet),
            node_id=packet_id,
        )

    def record_memory_entry(self, entry: Dict[str, Any], *, owner_kernel_id: str, branch_id: str = "", memory_kind: str = "") -> str:
        key = str(entry.get("key") or entry.get("summary_key") or _node_id("memory"))
        node_id = f"mem_{key}" if not key.startswith("mem_") else key
        payload = dict(entry)
        if memory_kind and "kind" not in payload:
            payload["kind"] = memory_kind
        return self.add_node(
            "memory",
            label=key,
            owner_kernel_id=owner_kernel_id,
            branch_id=branch_id or owner_kernel_id,
            payload=payload,
            node_id=node_id,
        )

    def record_sync_event(
        self,
        *,
        request_id: str,
        sync_mode: str,
        owner_kernel_id: str,
        target_kernel_id: str,
        promoted_keys: Optional[List[str]] = None,
        summary_key: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        sync_node_id = self.add_node(
            "memory_sync",
            label=request_id or sync_mode,
            owner_kernel_id=owner_kernel_id,
            branch_id=owner_kernel_id,
            payload={
                "request_id": request_id,
                "sync_mode": sync_mode,
                "target_kernel_id": target_kernel_id,
                **dict(payload or {}),
            },
            node_id=request_id or "",
        )
        for key in promoted_keys or []:
            mem_id = f"mem_{key}" if not str(key).startswith("mem_") else str(key)
            if mem_id in self.nodes:
                self.link(sync_node_id, mem_id, "promoted")
        if summary_key:
            mem_id = f"mem_{summary_key}" if not str(summary_key).startswith("mem_") else str(summary_key)
            if mem_id in self.nodes:
                self.link(sync_node_id, mem_id, "summarized")
        return sync_node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "branch_index": {k: list(v) for k, v in self.branch_index.items()},
            "kernel_index": {k: list(v) for k, v in self.kernel_index.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ProvenanceGraph":
        g = cls()
        if not isinstance(data, dict):
            return g
        for node_id, node_data in dict(data.get("nodes", {})).items():
            try:
                g.nodes[str(node_id)] = ProvenanceNode(**dict(node_data))
            except Exception:
                pass
        for edge_data in list(data.get("edges", [])):
            try:
                g.edges.append(ProvenanceEdge(**dict(edge_data)))
            except Exception:
                pass
        g.branch_index = {str(k): [str(x) for x in list(v)] for k, v in dict(data.get("branch_index", {})).items()}
        g.kernel_index = {str(k): [str(x) for x in list(v)] for k, v in dict(data.get("kernel_index", {})).items()}
        return g

    def summary(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "branches": {k: len(v) for k, v in self.branch_index.items()},
            "kernels": {k: len(v) for k, v in self.kernel_index.items()},
            "recent_edges": [e.to_dict() for e in self.edges[-10:]],
        }
