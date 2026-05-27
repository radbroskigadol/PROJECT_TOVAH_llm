"""
TOVAH v14 debug/failure_clusters.py — Failure clustering.

Groups related failures to detect patterns:
- repeated tool failures
- recurring patch rejections
- systematic regression areas
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FailureCluster:
    """A cluster of related failures."""
    cluster_id: str
    category: str  # "tool_failure", "patch_rejection", "regression", "timeout", "parse_error"
    entries: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    count: int = 0
    resolved: bool = False


def cluster_failures(
    error_log: List[Dict[str, Any]],
    max_clusters: int = 20,
) -> List[FailureCluster]:
    """Group errors into clusters by category and key pattern.

    Each error dict should have at minimum: {category, key, message, timestamp}
    """
    clusters: Dict[str, FailureCluster] = {}

    for entry in error_log[-200:]:
        cat = str(entry.get("category", "unknown"))
        key = str(entry.get("key", ""))
        cluster_key = f"{cat}:{key}"

        if cluster_key not in clusters:
            clusters[cluster_key] = FailureCluster(
                cluster_id=cluster_key, category=cat,
                first_seen=entry.get("timestamp", time.time()),
            )
        c = clusters[cluster_key]
        c.entries.append(entry)
        c.entries = c.entries[-20:]
        c.last_seen = entry.get("timestamp", time.time())
        c.count += 1

    result = sorted(clusters.values(), key=lambda c: c.count, reverse=True)
    return result[:max_clusters]
