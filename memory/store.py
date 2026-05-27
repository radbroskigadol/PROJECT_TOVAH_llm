"""
TOVAH v14 memory/store.py — Memory entry and store.

SEMANTIC PRESERVATION:
  MemoryEntry shape preserves all v13 fields.
  Added: last_cycle_touched for stale cleanup.

The store manages three memory kinds: episodic, semantic, procedural.
Each store operation updates bilateral health.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.config.constants import MAX_MEMORY_PER_KIND
from tovah_v14.core.primitives import BilateralValue, bilateral_recover
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.state import ShadowState


@dataclass
class MemoryEntry:
    """Single memory entry. Shape preserved from v13 + v14 additions."""
    kind: str  # "episodic", "semantic", "procedural"
    key: str
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    goal_context: str = ""
    tags: List[str] = field(default_factory=list)
    bilateral_confidence: BilateralValue = field(default_factory=lambda: BilateralValue(0.5, 0.1))
    last_cycle_touched: int = 0


VALID_KINDS = ("episodic", "semantic", "procedural")


class MemoryStore:
    """Tripartite memory store with bilateral health tracking."""

    def __init__(self) -> None:
        self.banks: Dict[str, List[MemoryEntry]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
        }

    def store(
        self,
        kind: str,
        key: str,
        data: Dict[str, Any],
        goal_context: str = "",
        tags: Optional[List[str]] = None,
        cycle: int = 0,
        state: Optional[ShadowState] = None,
    ) -> MemoryEntry:
        """Store an entry. Updates bilateral health if state provided."""
        if kind not in VALID_KINDS:
            kind = "episodic"
        entry = MemoryEntry(
            kind=kind, key=key, data=data,
            goal_context=goal_context, tags=tags or [],
            bilateral_confidence=BilateralValue(0.6, 0.1),
            last_cycle_touched=cycle,
        )
        self.banks[kind].append(entry)
        self.banks[kind] = self.banks[kind][-MAX_MEMORY_PER_KIND:]

        if state is not None:
            state.beta["memory.consolidation_health"] = bilateral_recover(
                state.beta.get("memory.consolidation_health", BilateralValue(0.5, 0.2)),
                truth_gain=0.05, falsity_decay=0.02,
            )
            refresh_state(state)

        return entry

    def get_bank(self, kind: str) -> List[MemoryEntry]:
        return self.banks.get(kind, [])

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.banks.items()}

    def all_entries(self) -> List[MemoryEntry]:
        result = []
        for bank in self.banks.values():
            result.extend(bank)
        return result
