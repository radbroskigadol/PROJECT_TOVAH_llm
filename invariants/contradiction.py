"""
TOVAH v14 invariants/contradiction.py — Contradiction governance / glut hygiene.

Mandatory at every mutable subsystem. Detects glut, gap, and coherence
defects. Distinguishes informative contradictions from destabilizing ones.

Actions per subsystem:
  preserve / dampen / quarantine / escalate / bridge-to-classical / archive

This module provides diagnostic tools. The kernel orchestrates policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState


@dataclass
class ContradictionDiagnostic:
    """Diagnostic for a single contradicted belief key."""
    key: str
    glut: float
    gap: float
    delta: float
    classification: str  # "informative", "destabilizing", "transient", "structural"
    recommended_action: str  # "preserve", "dampen", "quarantine", "escalate"
    subsystem: str = ""


@dataclass
class GlutHygieneReport:
    """Summary of contradiction state across the entire kernel."""
    total_keys: int = 0
    glut_keys: int = 0
    gap_keys: int = 0
    informative_contradictions: int = 0
    destabilizing_contradictions: int = 0
    diagnostics: List[ContradictionDiagnostic] = field(default_factory=list)
    recommended_dampening: List[str] = field(default_factory=list)
    recommended_quarantine: List[str] = field(default_factory=list)
    recommended_escalation: List[str] = field(default_factory=list)


def classify_contradiction(key: str, v: BilateralValue) -> str:
    """Classify a contradiction as informative or destabilizing.

    Informative: glut is moderate, delta is not extreme — genuine uncertainty.
    Destabilizing: very high glut with near-zero delta — cancellation.
    Transient: low glut that will likely resolve with more evidence.
    Structural: very high glut for extended periods (tracked externally).
    """
    if v.glut < 0.15:
        return "transient"
    if v.glut >= 0.45 and abs(v.delta) < 0.1:
        return "destabilizing"
    if v.glut >= 0.25:
        return "informative"
    return "transient"


def recommend_action(classification: str, key: str) -> str:
    """Recommend a contradiction management action."""
    if classification == "destabilizing":
        # High-value kernel keys get escalation, others get dampening
        if any(prefix in key for prefix in ["runtime.", "state.", "regression.", "boot."]):
            return "escalate"
        return "dampen"
    if classification == "informative":
        return "preserve"
    return "preserve"  # transient: let evidence accumulate


def diagnose_contradictions(
    s: ShadowState,
    glut_threshold: float = 0.25,
    gap_threshold: float = 0.30,
) -> List[ContradictionDiagnostic]:
    """Diagnose all contradicted/gapped beliefs in the state.

    Returns sorted by glut descending (most contradicted first).
    """
    diagnostics: List[ContradictionDiagnostic] = []
    for key, v in s.beta.items():
        if v.glut >= glut_threshold or v.gap >= gap_threshold:
            cls = classify_contradiction(key, v)
            action = recommend_action(cls, key)
            # Infer subsystem from key prefix
            subsystem = key.split(".")[0] if "." in key else "unknown"
            diagnostics.append(ContradictionDiagnostic(
                key=key, glut=v.glut, gap=v.gap, delta=v.delta,
                classification=cls, recommended_action=action,
                subsystem=subsystem,
            ))
    diagnostics.sort(key=lambda d: d.glut, reverse=True)
    return diagnostics


def build_hygiene_report(s: ShadowState) -> GlutHygieneReport:
    """Build a full glut hygiene report for the current state."""
    diagnostics = diagnose_contradictions(s)
    report = GlutHygieneReport(
        total_keys=len(s.beta),
        glut_keys=sum(1 for v in s.beta.values() if v.glut >= 0.25),
        gap_keys=sum(1 for v in s.beta.values() if v.gap >= 0.30),
        informative_contradictions=sum(1 for d in diagnostics if d.classification == "informative"),
        destabilizing_contradictions=sum(1 for d in diagnostics if d.classification == "destabilizing"),
        diagnostics=diagnostics[:30],
    )
    for d in diagnostics:
        if d.recommended_action == "dampen":
            report.recommended_dampening.append(d.key)
        elif d.recommended_action == "quarantine":
            report.recommended_quarantine.append(d.key)
        elif d.recommended_action == "escalate":
            report.recommended_escalation.append(d.key)
    return report
