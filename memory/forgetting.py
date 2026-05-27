"""
TOVAH v14 memory/forgetting.py — Forgetting and cleanup.

Implements selective forgetting:
- Entries older than max_age_hours with low confidence are removed
- Entries with bilateral confidence below threshold are decayed
- Compression: merge near-duplicate semantic entries

Without forgetting, memory becomes noise. This is critical
for retrieval quality.
"""
from __future__ import annotations

import time
from typing import Dict, List

from tovah_v14.memory.store import MemoryStore, MemoryEntry
from tovah_v14.core.primitives import BilateralValue


def forget_stale(
    store: MemoryStore,
    kind: str,
    max_age_hours: float = 48.0,
    min_confidence: float = 0.3,
) -> int:
    """Remove stale, low-confidence entries from a memory bank.

    An entry is forgotten if:
    - older than max_age_hours AND
    - bilateral_confidence.t < min_confidence AND
    - access_count <= 1

    Returns count of entries removed.
    """
    now = time.time()
    cutoff = now - max_age_hours * 3600
    bank = store.get_bank(kind)
    before = len(bank)

    keep: List[MemoryEntry] = []
    for entry in bank:
        is_old = entry.created_at < cutoff
        is_low = entry.bilateral_confidence.t < min_confidence
        is_unused = entry.access_count <= 1
        if is_old and is_low and is_unused:
            continue  # forget
        keep.append(entry)

    store.banks[kind] = keep
    return before - len(keep)


def cleanup_memory(
    store: MemoryStore,
    max_age_hours: float = 48.0,
    min_confidence: float = 0.3,
) -> Dict[str, int]:
    """Run cleanup across all memory banks.

    Returns {kind: count_removed}.
    """
    return {
        kind: forget_stale(store, kind, max_age_hours, min_confidence)
        for kind in ("episodic", "semantic", "procedural")
    }


def decay_low_confidence(
    store: MemoryStore,
    kind: str,
    decay_rate: float = 0.05,
    threshold: float = 0.4,
) -> int:
    """Decay confidence of entries below threshold.

    Entries with t < threshold get their t reduced by decay_rate.
    This gradually makes them eligible for forgetting.

    Returns count of entries decayed.
    """
    count = 0
    for entry in store.get_bank(kind):
        if entry.bilateral_confidence.t < threshold:
            entry.bilateral_confidence.t = max(0.0, entry.bilateral_confidence.t - decay_rate)
            count += 1
    return count
