"""
TOVAH v14 mutation/quarantine.py — Patch quarantine.

Quarantined patches are isolated from the promotion pipeline.
They cannot be promoted until explicitly reviewed and released.

Quarantine triggers:
- Contract violation
- Regression after shadow deployment
- Destabilizing contradiction detected
- Manual quarantine by David
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QuarantineRecord:
    """Record of a quarantined patch."""
    patch_name: str
    target: str
    reason: str
    quarantined_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    source: str = ""
    errors: List[str] = field(default_factory=list)
    released: bool = False
    released_at: str = ""


class QuarantineManager:
    """Manages quarantined patches."""

    def __init__(self) -> None:
        self.records: Dict[str, QuarantineRecord] = {}

    def quarantine(
        self,
        patch_name: str,
        target: str,
        reason: str,
        source: str = "",
        errors: List[str] | None = None,
    ) -> QuarantineRecord:
        """Quarantine a patch. It cannot be promoted until released."""
        rec = QuarantineRecord(
            patch_name=patch_name,
            target=target,
            reason=reason,
            source=source,
            errors=errors or [],
        )
        self.records[patch_name] = rec
        logging.info(f"QUARANTINED: {patch_name} -> {target} | {reason}")
        return rec

    def release(self, patch_name: str) -> bool:
        """Release a patch from quarantine. Returns True if found and released."""
        rec = self.records.get(patch_name)
        if rec is None:
            return False
        rec.released = True
        rec.released_at = dt.datetime.now().isoformat(timespec="seconds")
        logging.info(f"QUARANTINE RELEASED: {patch_name}")
        return True

    def is_quarantined(self, patch_name: str) -> bool:
        """Check if a patch is currently quarantined (not released)."""
        rec = self.records.get(patch_name)
        return rec is not None and not rec.released

    def list_active(self) -> List[QuarantineRecord]:
        """List all currently quarantined (not released) patches."""
        return [r for r in self.records.values() if not r.released]


def quarantine_patch(
    patch_name: str,
    target: str,
    reason: str,
    staged_patches: Dict[str, Dict[str, Any]],
    manager: QuarantineManager | None = None,
) -> QuarantineRecord:
    """Quarantine a patch and update its staged status.

    Convenience function that updates the staged_patches dict
    and registers with the QuarantineManager.
    """
    if manager is None:
        manager = QuarantineManager()

    rec = manager.quarantine(patch_name, target, reason)

    # Update staged patch status
    if patch_name in staged_patches:
        staged_patches[patch_name]["status"] = "quarantined"
        staged_patches[patch_name]["quarantine_reason"] = reason

    return rec
