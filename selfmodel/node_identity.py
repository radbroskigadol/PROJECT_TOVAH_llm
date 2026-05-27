"""TOVAH v16 selfmodel/node_identity.py — typed node identity for local/distributed kernels."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class NodeIdentity:
    node_id: str = "main_node"
    kernel_id: str = "main"
    role: str = "main"
    locality: str = "local"
    sovereign: bool = True
    parent_node_id: str = ""
    trust_level: str = "sovereign"
    mission_context: str = "global mission"
    lifecycle: str = "born"
    specialization: str = ""
    last_cluster_sync: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return asdict(self)

    def touch(self, **updates: Any) -> None:
        self.last_cluster_sync = time.time()
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "NodeIdentity":
        if not isinstance(data, dict):
            return cls()
        return cls(**dict(data))
