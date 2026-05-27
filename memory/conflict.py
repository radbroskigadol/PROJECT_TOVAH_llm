"""
TOVAH v14 memory/conflict.py — Contradiction-aware memory conflict handling.

CRITICAL RULE FROM SPEC:
  Memory conflict creates MemoryConflictRecord.
  Do NOT overwrite prior record.

When storing a new entry that contradicts an existing one,
we preserve BOTH and create a conflict record. Resolution
is a separate deliberate action, not a side effect of storage.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.memory.store import MemoryStore, MemoryEntry
from tovah_v14.core.primitives import BilateralValue


@dataclass
class MemoryConflictRecord:
    """Records a detected conflict between memory entries.

    The existing entry is NOT overwritten. Both are preserved.
    Resolution happens through explicit action, not silent overwrite.
    """
    conflict_id: str
    kind: str
    key: str
    existing_entry_key: str
    new_entry_key: str
    conflict_type: str  # "contradictory_data", "key_collision", "tag_disagreement"
    description: str = ""
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: str = ""


def _entries_conflict(existing: MemoryEntry, new_data: Dict[str, Any], new_key: str) -> Optional[str]:
    """Check if a new entry conflicts with an existing one.

    Returns conflict type string or None.
    """
    # Same key with different data
    if existing.key == new_key:
        existing_json = json.dumps(existing.data, sort_keys=True, default=str)
        new_json = json.dumps(new_data, sort_keys=True, default=str)
        if existing_json != new_json:
            return "key_collision"

    # Same key prefix with contradictory outcome
    if existing.key.split(":")[0] == new_key.split(":")[0]:
        old_outcome = existing.data.get("outcome")
        new_outcome = new_data.get("outcome")
        if old_outcome and new_outcome and old_outcome != new_outcome:
            return "contradictory_data"

    return None


def check_memory_conflict(
    store: MemoryStore,
    kind: str,
    key: str,
    data: Dict[str, Any],
) -> List[MemoryConflictRecord]:
    """Check for conflicts before storing.

    Scans the relevant bank for entries that conflict with the
    proposed new entry. Returns a list of conflict records.

    The caller decides whether to proceed with storage despite conflicts.
    """
    bank = store.get_bank(kind)
    conflicts: List[MemoryConflictRecord] = []

    for existing in bank[-100:]:  # check recent entries for efficiency
        conflict_type = _entries_conflict(existing, data, key)
        if conflict_type:
            conflicts.append(MemoryConflictRecord(
                conflict_id=f"mc_{int(time.time())}_{len(conflicts)}",
                kind=kind,
                key=key,
                existing_entry_key=existing.key,
                new_entry_key=key,
                conflict_type=conflict_type,
                description=f"{conflict_type} between '{existing.key}' and '{key}'",
            ))

    return conflicts
