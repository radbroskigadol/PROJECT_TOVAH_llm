"""
TOVAH v14 selfmodel/competence.py — Competence map.

Tracks measured competence per domain, tied to actual task outcomes.
Curriculum advancement is gated on competence measurement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from tovah_v14.core.primitives import BilateralValue, bilateral_recover


@dataclass
class CompetenceEntry:
    """Measured competence in a domain."""
    domain: str
    measured_mastery: float = 0.0
    last_tested: float = 0.0
    test_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    bilateral_confidence: BilateralValue = field(default_factory=lambda: BilateralValue(0.3, 0.3))
    linked_subsystems: List[str] = field(default_factory=list)


class CompetenceMap:
    """Maps domains to measured competence."""

    def __init__(self) -> None:
        self.entries: Dict[str, CompetenceEntry] = {}

    def record_outcome(self, domain: str, success: bool) -> CompetenceEntry:
        """Record a test/task outcome for a domain."""
        entry = self.entries.get(domain)
        if entry is None:
            entry = CompetenceEntry(domain=domain)
            self.entries[domain] = entry

        entry.test_count += 1
        entry.last_tested = time.time()
        if success:
            entry.success_count += 1
            entry.bilateral_confidence = bilateral_recover(
                entry.bilateral_confidence, truth_gain=0.12, falsity_decay=0.05,
            )
        else:
            entry.failure_count += 1
            entry.bilateral_confidence = bilateral_recover(
                entry.bilateral_confidence, truth_gain=0.0, falsity_decay=0.0,
            )
            entry.bilateral_confidence.f = min(1.0, entry.bilateral_confidence.f + 0.08)

        # Measured mastery from success rate
        total = max(1, entry.success_count + entry.failure_count)
        entry.measured_mastery = entry.success_count / total

        return entry

    def get_weakest(self, limit: int = 3) -> List[CompetenceEntry]:
        """Get domains with lowest measured mastery."""
        return sorted(self.entries.values(), key=lambda e: e.measured_mastery)[:limit]

    def get_untested(self, all_domains: List[str]) -> List[str]:
        """Get domains that have never been tested."""
        return [d for d in all_domains if d not in self.entries]
