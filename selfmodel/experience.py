"""
TOVAH v14 selfmodel/experience.py — Experience replay and outcome labeling.

Every research, patch, tool use, and plan step produces an ExperienceRecord.
Outcomes are labeled: useful/useless/contradictory/improved/regressed/neutral.

This is how TOVAH transforms her life into training data.

AUDIT FIX (P0-2, v14.1.2): record() now accepts optional independent
`truth_evidence` and `falsity_evidence` arguments (both in [0, 1]). When
both are supplied, the bilateral_assessment is set directly from them
rather than derived from the single reward signal. This makes
contradiction-class (K) examples reachable from the experience path,
which under the old scalar-reward derivation was mathematically
impossible (T + F = 1 always, so K-class threshold T ≥ 0.55 AND
F ≥ 0.55 could never be met).

When only `reward_signal` is given the v13/v14.0/v14.1 behaviour is
preserved exactly, so existing callers continue to produce A/B/G classes.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tovah_v14.config.constants import MAX_EXPERIENCE_RECORDS
from tovah_v14.core.primitives import BilateralValue


@dataclass
class ExperienceRecord:
    """Single recorded experience for replay and learning."""
    record_id: str
    action_type: str  # "research", "patch", "tool_use", "plan_step", "goal_completion"
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"  # "useful", "useless", "contradictory", "improved", "regressed", "neutral"
    reward_signal: float = 0.0  # [-1, 1]
    bilateral_assessment: BilateralValue = field(default_factory=lambda: BilateralValue(0.5, 0.5))
    subsystem_deltas: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    trace_id: str = ""
    tags: List[str] = field(default_factory=list)


class ExperienceStore:
    """Stores and retrieves experience records.

    Optional `on_record` callback fires after every successful record(),
    receiving the record-as-dict. Used by the kernel to wire continuous
    corpus export. Callback failures are swallowed and logged so the
    store's own behaviour is never perturbed by downstream subscribers.
    """

    def __init__(self, on_record: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.records: List[ExperienceRecord] = []
        self.on_record: Optional[Callable[[Dict[str, Any]], None]] = on_record

    def record(
        self,
        record_id: str,
        action_type: str,
        context: Dict[str, Any] | None = None,
        outcome: str = "unknown",
        reward_signal: float = 0.0,
        tags: List[str] | None = None,
        subsystem_deltas: Dict[str, float] | None = None,
        truth_evidence: float | None = None,
        falsity_evidence: float | None = None,
    ) -> ExperienceRecord:
        """Record an experience.

        If `truth_evidence` AND `falsity_evidence` are both given (each in
        [0, 1]), they set the bilateral assessment directly — this allows
        events where both confirming and refuting evidence were collected
        (e.g. a research finding that both supports and contradicts the
        hypothesis) to produce K-class training examples.

        If either is None, the legacy scalar-reward derivation runs and
        T + F = 1 always (yielding A/B/G classes only).
        """
        if truth_evidence is not None and falsity_evidence is not None:
            bv = BilateralValue(
                max(0.0, min(1.0, float(truth_evidence))),
                max(0.0, min(1.0, float(falsity_evidence))),
            )
        else:
            bv = BilateralValue(
                max(0.0, min(1.0, 0.5 + 0.5 * reward_signal)),
                max(0.0, min(1.0, 0.5 - 0.5 * reward_signal)),
            )
        rec = ExperienceRecord(
            record_id=record_id,
            action_type=action_type,
            context=context or {},
            outcome=outcome,
            reward_signal=reward_signal,
            bilateral_assessment=bv,
            tags=tags or [],
            subsystem_deltas=subsystem_deltas or {},
        )
        self.records.append(rec)
        self.records = self.records[-MAX_EXPERIENCE_RECORDS:]
        if self.on_record is not None:
            try:
                self.on_record(asdict(rec))
            except Exception as e:
                import logging
                logging.debug(f"ExperienceStore.on_record subscriber failed: {e}")
        return rec

    def replay(self, action_type: str = "", limit: int = 20) -> List[ExperienceRecord]:
        """Retrieve recent experiences, optionally filtered by action type."""
        if action_type:
            filtered = [r for r in self.records if r.action_type == action_type]
        else:
            filtered = list(self.records)
        return filtered[-limit:]

    def outcome_summary(self) -> Dict[str, int]:
        """Count records by outcome label."""
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r.outcome] = counts.get(r.outcome, 0) + 1
        return counts
