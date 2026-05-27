"""TOVAH v16 cluster/node.py — typed cluster node records.

Local-first scaffolding for future distributed subkernels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class ClusterNodeRecord:
    node_id: str
    kernel_id: str
    role: str
    parent_kernel_id: str = ""
    lifecycle: str = "born"
    mission_context: str = ""
    trust_level: str = "provisional"
    locality: str = "local"
    status: str = "active"
    specialization: str = ""
    capabilities: List[str] = field(default_factory=list)
    packet_count: int = 0
    branch_checkpoint_count: int = 0
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self, **updates: Any) -> None:
        self.last_seen = time.time()
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ClusterNodeRecord | None":
        if not isinstance(data, dict) or not data:
            return None
        return cls(**dict(data))
