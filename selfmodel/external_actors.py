"""TOVAH v16 selfmodel/external_actors.py — minimal external actor records."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class ExternalActorRecord:
    actor_id: str
    role: str = "user"
    trust_level: str = "trusted"
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
