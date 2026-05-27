"""
TOVAH v14 mutation/mutation_log.py — Mutation logger.

Every patch lifecycle event is recorded:
  proposed → analyzed → staged → sandboxed → regression-tested
  → shadow-deployed → promoted → quarantined → reverted → archived

No silent transitions. Every event has a timestamp and reason.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, List


class MutationLogger:
    """Append-only mutation log.

    Writes to a .py file for human readability (preserving v13 format)
    and maintains an in-memory event list for structured access.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.events: List[Dict[str, Any]] = []

    def record(
        self,
        event_type: str,
        patch_name: str,
        target: str,
        code: str = "",
        reason: str = "",
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Record a mutation lifecycle event."""
        event = {
            "event_type": event_type,
            "patch_name": patch_name,
            "target": target,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
        }
        if details:
            event["details"] = details
        self.events.append(event)
        self.events = self.events[-500:]

        # Append to log file (v13-compatible .py format)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"\n# === {event_type} {patch_name} -> {target} @ {event['timestamp']} ===\n")
                if reason:
                    f.write(f"# reason: {reason}\n")
                if code:
                    f.write(f"{code}\n")
        except Exception as e:
            logging.warning(f"mutation log write failed: {e}")

    def record_stage(self, patch_name: str, target: str, source: str = "") -> None:
        self.record("STAGED", patch_name, target, reason=f"source={source}")

    def record_apply(self, patch_name: str, target: str, code: str = "") -> None:
        self.record("APPLIED", patch_name, target, code=code)

    def record_revert(self, patch_name: str, target: str) -> None:
        self.record("REVERTED", patch_name, target, code="# REVERTED\n")

    def record_quarantine(self, patch_name: str, target: str, reason: str = "") -> None:
        self.record("QUARANTINED", patch_name, target, reason=reason)

    def record_promote(self, patch_name: str, target: str, stage: str = "") -> None:
        self.record("PROMOTED", patch_name, target, reason=f"stage={stage}")

    def record_reject(self, patch_name: str, target: str, reason: str = "") -> None:
        self.record("REJECTED", patch_name, target, reason=reason)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent events."""
        return self.events[-limit:]
