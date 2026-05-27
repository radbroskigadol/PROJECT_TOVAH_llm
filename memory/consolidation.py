"""
TOVAH v14 memory/consolidation.py — Memory consolidation.

Distills episodic memory into semantic patterns.
Promotes successful workflow chains to procedural memory.

Consolidation is SELECTIVE — only high-access, high-confidence
episodes get promoted. This prevents memory bloat.
"""
from __future__ import annotations

import time
from typing import Dict, List

from tovah_v14.memory.store import MemoryStore, MemoryEntry
from tovah_v14.core.primitives import BilateralValue


def consolidate_memory(
    store: MemoryStore,
    max_age_hours: float = 1.0,
) -> Dict[str, int]:
    """Run one consolidation pass.

    1. Find recent episodic entries (within max_age_hours)
    2. Cluster by tags
    3. Promote tag clusters with >= 2 entries to semantic memory
    4. Promote successful workflow episodes to procedural memory

    Returns counts: {"semantic_created": N, "procedural_created": N}
    """
    now = time.time()
    cutoff = now - max_age_hours * 3600
    counts = {"semantic_created": 0, "procedural_created": 0}

    episodes = store.get_bank("episodic")
    recent = [e for e in episodes if e.created_at >= cutoff]

    if len(recent) < 1:
        return counts

    # 1. Tag clustering → semantic (needs >= 2)
    if len(recent) >= 2:
        tag_clusters: Dict[str, List[MemoryEntry]] = {}
        for ep in recent:
            for tag in ep.tags:
                tag_clusters.setdefault(tag, []).append(ep)

        semantic_keys = {e.key for e in store.get_bank("semantic")}
        for tag, eps in tag_clusters.items():
            if len(eps) >= 2:
                pattern_key = f"pattern:{tag}"
                if pattern_key not in semantic_keys:
                    store.store(
                        "semantic", pattern_key,
                        {
                            "observation": f"{len(eps)} episodes tagged '{tag}' in last {max_age_hours}h",
                            "episode_keys": [e.key for e in eps[:5]],
                        },
                        tags=[tag, "auto_consolidated"],
                    )
                    counts["semantic_created"] += 1
                    semantic_keys.add(pattern_key)

    # 2. Successful workflows → procedural
    procedural_keys = {e.key for e in store.get_bank("procedural")}
    for ep in recent:
        if (ep.data.get("outcome") == "success"
                and "workflow" in ep.tags
                and ep.bilateral_confidence.t > 0.5
                and ep.key not in procedural_keys):
            store.store(
                "procedural", ep.key,
                {
                    "steps": ep.data.get("steps", []),
                    "goal": ep.goal_context,
                    "success_count": 1,
                },
                goal_context=ep.goal_context,
                tags=["auto_promoted"] + ep.tags,
            )
            counts["procedural_created"] += 1
            procedural_keys.add(ep.key)

    return counts
