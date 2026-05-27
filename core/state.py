"""
TOVAH v14 core/state.py — State dataclasses.

SEMANTIC PRESERVATION:
  CarrierState, ProvenanceState, ShadowState shapes are preserved from v13.
  All existing fields keep their types and defaults.
  v14 additions are additive only (new fields with defaults).

MIGRATION NOTE:
  Old v13 state files will not have `degraded` on CarrierState.
  The default is False, so missing fields are safe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from tovah_v14.core.primitives import BilateralValue


@dataclass
class CarrierState:
    """Kernel carrier state: operational mode and tracking."""
    active_goal: str = ""
    last_tool: str = ""
    last_action: str = ""
    cycle: int = 0
    mode: str = "local"
    paused: bool = False
    degraded: bool = False  # v14: true when regression rate below threshold


@dataclass
class ProvenanceState:
    """Provenance tracking for state transitions."""
    step: int = 0
    tags: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    refresh_count: int = 0


@dataclass
class ShadowState:
    """Complete ShadowHoTT kernel state.

    c: carrier state (operational)
    beta: bilateral evidence map (the core)
    nu: gamma cache (four-valued classification, derived from beta)
    pi: provenance tracking
    """
    c: CarrierState
    beta: Dict[str, BilateralValue]
    nu: Dict[str, str]  # gamma cache: key -> "T" | "F" | "B" | "G"
    pi: ProvenanceState

    def snapshot(self) -> Dict[str, Any]:
        """Serialize to plain dict for persistence.
        This is the canonical serialization shape that v13 state files use.
        """
        return {
            "c": asdict(self.c),
            "beta": {k: asdict(v) for k, v in self.beta.items()},
            "nu": dict(self.nu),
            "pi": asdict(self.pi),
        }
