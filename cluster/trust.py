"""TOVAH v16 cluster/trust.py — trust ledger for node governance."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List


TRUST_LEVELS = ["untrusted", "low", "provisional", "trusted", "sovereign"]
TRUST_SCORES = {level: idx for idx, level in enumerate(TRUST_LEVELS)}


@dataclass
class TrustEvent:
    node_id: str
    trust_level: str
    reason: str
    source: str = "main"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClusterTrustLedger:
    @staticmethod
    def trust_score(level: str) -> int:
        return int(TRUST_SCORES.get(str(level or "").lower(), 0))

    @staticmethod
    def _clamp_score(score: int) -> int:
        return max(0, min(score, len(TRUST_LEVELS) - 1))

    @classmethod
    def _level_for_score(cls, score: int) -> str:
        return TRUST_LEVELS[cls._clamp_score(int(score))]

    def meets_threshold(self, node_id: str, required_level: str) -> bool:
        record = self.node_trust.get(node_id) or {}
        current = str(record.get("trust_level", "untrusted"))
        return self.trust_score(current) >= self.trust_score(required_level)

    def trust_level_for(self, node_id: str, default: str = "provisional") -> str:
        return str((self.node_trust.get(node_id) or {}).get("trust_level", default))

    def baseline_level_for(self, node_id: str, default: str = "provisional") -> str:
        return str((self.node_trust.get(node_id) or {}).get("baseline_trust_level", default))

    def __init__(self) -> None:
        self.node_trust: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []

    def _append_event(self, node_id: str, trust_level: str, *, reason: str, source: str = "main", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ev = TrustEvent(
            node_id=node_id,
            trust_level=str(trust_level),
            reason=str(reason),
            source=str(source),
            metadata=dict(metadata or {}),
        ).to_dict()
        self.events.append(ev)
        self.events = self.events[-500:]
        return ev

    def set_trust(self, node_id: str, trust_level: str, *, reason: str, source: str = "main", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        current = self.node_trust.get(node_id) or {}
        record = {
            "node_id": node_id,
            "trust_level": str(trust_level),
            "baseline_trust_level": str(current.get("baseline_trust_level", trust_level)),
            "reason": str(reason),
            "source": str(source),
            "last_updated": time.time(),
            "metadata": {**dict(current.get("metadata", {})), **dict(metadata or {})},
        }
        self.node_trust[node_id] = record
        self._append_event(node_id, str(trust_level), reason=reason, source=source, metadata=record["metadata"])
        return copy.deepcopy(record)

    def ensure_node(self, node_id: str, default_trust_level: str, *, reason: str = "registry_sync", source: str = "main", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        existing = self.node_trust.get(node_id)
        if existing is None:
            return self.set_trust(node_id, default_trust_level, reason=reason, source=source, metadata=metadata)
        existing["baseline_trust_level"] = str(default_trust_level or existing.get("baseline_trust_level", existing.get("trust_level", "provisional")))
        existing["reason"] = str(reason)
        existing["source"] = str(source)
        existing["last_updated"] = time.time()
        merged = dict(existing.get("metadata", {}))
        merged.update(dict(metadata or {}))
        existing["metadata"] = merged
        self.node_trust[node_id] = existing
        self._append_event(
            node_id,
            str(existing.get("trust_level", default_trust_level)),
            reason=f"{reason}:baseline_refresh",
            source=source,
            metadata={"baseline_trust_level": existing["baseline_trust_level"], **dict(metadata or {})},
        )
        return copy.deepcopy(existing)

    def note_packet(self, packet: Any) -> None:
        node_id = str(getattr(packet, "source_kernel_id", "") or "")
        if not node_id:
            return
        current = self.node_trust.get(node_id, {"trust_level": str(getattr(packet, "trust_level", "provisional") or "provisional")})
        self.set_trust(
            node_id,
            str(getattr(packet, "trust_level", current.get("trust_level", "provisional"))),
            reason=f"packet:{getattr(packet, 'packet_kind', 'unknown')}",
            source=str(getattr(packet, "target_kernel_id", "main") or "main"),
            metadata={"risk_class": str(getattr(packet, "risk_class", "")), "packet_id": str(getattr(packet, "packet_id", ""))},
        )

    def record_outcome(
        self,
        node_id: str,
        outcome: str,
        *,
        success: bool,
        severity: str = "normal",
        source: str = "main",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        outcome = str(outcome or "outcome")
        severity = str(severity or "normal")
        base = self.node_trust.get(node_id) or self.ensure_node(node_id, "provisional", reason="outcome_bootstrap", source=source)
        current_score = self.trust_score(base.get("trust_level", "provisional"))
        baseline_score = self.trust_score(base.get("baseline_trust_level", base.get("trust_level", "provisional")))

        delta = 1 if success else -1
        if severity in {"high", "severe", "critical"}:
            delta *= 2
        elif severity in {"low", "minor"}:
            delta = 1 if success else 0

        new_score = self._clamp_score(current_score + delta)
        if success and new_score < baseline_score:
            new_score = min(baseline_score, new_score + 1)

        new_level = self._level_for_score(new_score)
        merged = dict(base.get("metadata", {}))
        merged.update(dict(metadata or {}))
        merged["last_outcome"] = outcome
        merged["success_count"] = int(merged.get("success_count", 0)) + (1 if success else 0)
        merged["failure_count"] = int(merged.get("failure_count", 0)) + (0 if success else 1)
        merged["outcome_count"] = int(merged.get("outcome_count", 0)) + 1
        merged["outcome_success_rate"] = merged["success_count"] / max(1, merged["outcome_count"])

        record = {
            "node_id": node_id,
            "trust_level": new_level,
            "baseline_trust_level": base.get("baseline_trust_level", new_level),
            "reason": f"outcome:{outcome}",
            "source": str(source),
            "last_updated": time.time(),
            "metadata": merged,
        }
        self.node_trust[node_id] = record
        self._append_event(
            node_id,
            new_level,
            reason=f"outcome:{outcome}",
            source=source,
            metadata={"success": bool(success), "severity": severity, "delta": delta, **dict(metadata or {})},
        )
        return copy.deepcopy(record)

    def recover_toward_baseline(self, node_id: str, *, amount: int = 1, source: str = "main", reason: str = "recovery", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        record = self.node_trust.get(node_id) or self.ensure_node(node_id, "provisional", reason="recovery_bootstrap", source=source)
        current_score = self.trust_score(record.get("trust_level", "provisional"))
        baseline_score = self.trust_score(record.get("baseline_trust_level", record.get("trust_level", "provisional")))
        if current_score >= baseline_score:
            return copy.deepcopy(record)
        new_score = min(baseline_score, current_score + max(1, int(amount)))
        new_level = self._level_for_score(new_score)
        record["trust_level"] = new_level
        record["reason"] = str(reason)
        record["source"] = str(source)
        record["last_updated"] = time.time()
        merged = dict(record.get("metadata", {}))
        merged.update(dict(metadata or {}))
        record["metadata"] = merged
        self.node_trust[node_id] = record
        self._append_event(node_id, new_level, reason=reason, source=source, metadata={"recovery_to_baseline": True, **dict(metadata or {})})
        return copy.deepcopy(record)

    def gate_report(self, node_id: str, *, required_level: str, reason: str = "") -> Dict[str, Any]:
        current = self.trust_level_for(node_id, default="untrusted")
        return {
            "node_id": node_id,
            "required_level": required_level,
            "current_level": current,
            "current_score": self.trust_score(current),
            "required_score": self.trust_score(required_level),
            "allowed": self.meets_threshold(node_id, required_level),
            "reason": reason or "threshold_check",
        }

    def get_node_report(self, node_id: str) -> Dict[str, Any]:
        rec = copy.deepcopy(self.node_trust.get(node_id, {}))
        rec["dynamic_delta"] = self.trust_score(rec.get("trust_level", "untrusted")) - self.trust_score(rec.get("baseline_trust_level", rec.get("trust_level", "untrusted")))
        return {
            "node": rec,
            "recent_events": [ev for ev in self.events if ev.get("node_id") == node_id][-10:],
        }

    def summary(self) -> Dict[str, Any]:
        levels: Dict[str, int] = {}
        total_score = 0
        dynamic_nodes = 0
        success_total = 0
        failure_total = 0
        for record in self.node_trust.values():
            level = record.get("trust_level", "unknown")
            levels[level] = levels.get(level, 0) + 1
            total_score += self.trust_score(level)
            if record.get("baseline_trust_level", level) != level:
                dynamic_nodes += 1
            meta = dict(record.get("metadata", {}))
            success_total += int(meta.get("success_count", 0))
            failure_total += int(meta.get("failure_count", 0))
        tracked = len(self.node_trust)
        total_outcomes = success_total + failure_total
        return {
            "tracked_nodes": tracked,
            "levels": levels,
            "dynamic_nodes": dynamic_nodes,
            "average_trust_score": (total_score / tracked) if tracked else 0.0,
            "outcome_success_rate": (success_total / total_outcomes) if total_outcomes else 0.0,
            "recent_events": self.events[-10:],
        }

    def export_state(self) -> Dict[str, Any]:
        return {"node_trust": copy.deepcopy(self.node_trust), "events": copy.deepcopy(self.events[-500:])}

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        self.node_trust = {}
        self.events = []
        if not isinstance(data, dict):
            return
        self.node_trust = dict(data.get("node_trust", {}))
        self.events = list(data.get("events", []))[-500:]
