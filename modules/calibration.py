"""
TOVAH v14.2.7 modules/calibration.py — Named calibration profiles.

AUDIT FIX (v14.2.7, sec 2 — Phase 1): the v14.2.6 feedback and gate code
contained ~30 numeric literals (0.55, 0.45, 0.80, 0.60, 0.40, ...) embedded
inline. This module lifts every one of them into named dataclass fields,
with defaults that EXACTLY match the v14.2.6 values. This is the
"refactor without behavior change" pass — a precondition for the bilateral
binding pass that follows.

Two profiles are defined:

  ModuleFeedbackCalibration   — coefficients for ModuleRegistry feedback,
                                family carry-over, and quality scoring.
  AdaptiveGateCalibration     — thresholds for the PromotionLadder
                                adaptive gates (evidence floors, success
                                floors, budget caps, failure caps).

Both are passed through their respective classes as optional kwargs.
Callers that don't supply them get the v14.2.6 defaults. Tests in
`tests/test_calibration_phase1.py` lock the defaults to v14.2.6 numerics
so any future drift is caught explicitly.

PHILOSOPHICAL NOTE: lifting constants into named fields does NOT solve the
"magic numbers" problem on its own — it only makes the surface auditable
and the constants tunable. Binding fields to module-family bilateral state
is a separate, deliberate next step. See AUDIT.md §2 for the phased plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Module-level feedback calibration (consumed by ModuleRegistry)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModuleFeedbackCalibration:
    """All numeric constants used by ModuleRegistry feedback math.

    Field names match the role each constant plays in the original code.
    Defaults match v14.2.6 exactly — see commit history for provenance.
    """

    # --- Evidence-quality scoring (_evidence_quality_total) ---------------
    # Penalty applied to stale-weak evidence per (1-quality)*(1-decay) unit.
    stale_weak_penalty_weight: float = 0.55
    # Credit applied to fresh-strong evidence; base + freshness scale.
    fresh_strong_credit_base: float = 0.45
    fresh_strong_credit_scale: float = 0.55
    # Cap on penalty cancellation as a fraction of strong credit.
    cancellation_fraction_of_credit: float = 0.80

    # --- Severity weights (apply_module_feedback) -------------------------
    severity_weights: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.5, "minor": 0.5, "normal": 1.0,
        "high": 1.5, "severe": 2.0, "critical": 2.5,
    })
    # Quality multiplier on rework/evidence-gather feedback: 0.75 + 0.55*q,
    # capped at 2.0.
    rework_quality_base: float = 0.75
    rework_quality_scale: float = 0.55
    rework_quality_cap: float = 2.0

    # --- Local feedback decay (apply_module_feedback) ---------------------
    success_decay_factor: float = 0.40   # fail weight decay on success
    success_maturity_gain: float = 0.20  # maturity bonus gain on success
    rework_maturity_bonus: float = 0.16  # extra maturity for proposal_rework
    rework_fail_relief: float = 0.15     # extra fail relief for proposal_rework
    evidence_gather_credit_gain: float = 0.35
    fail_maturity_penalty: float = 0.16
    fail_evidence_credit_loss: float = 0.20

    # --- Cooldown after failure (apply_module_feedback) -------------------
    cooldown_seconds_main: float = 600.0
    cooldown_seconds_default: float = 120.0

    # --- Maturity-bonus caps ---------------------------------------------
    maturity_bonus_cap_normal: float = 2.5
    maturity_bonus_cap_rework: float = 2.8
    maturity_bonus_floor: float = -1.5
    success_weight_cap: float = 8.0
    failure_weight_cap: float = 8.0
    evidence_reentry_cap: float = 2.5

    # --- Family-level feedback (multipliers and gains) -------------------
    fam_multiplier_evidence_success: float = 0.50
    fam_multiplier_evidence_default: float = 0.35
    fam_multiplier_rework_success: float = 0.62
    fam_multiplier_rework_fail: float = 0.40
    fam_multiplier_review_wave_success: float = 0.60
    fam_multiplier_review_wave_fail: float = 0.55
    fam_success_fail_decay: float = 0.20
    fam_maturity_success_gain: float = 0.10
    fam_review_wave_bonus: float = 0.08
    fam_review_wave_resolution_bonus: float = 0.10
    fam_rework_bonus: float = 0.12
    fam_rework_fail_relief: float = 0.18
    fam_maturity_cap_normal: float = 2.0
    fam_maturity_cap_review_wave: float = 2.5
    fam_maturity_cap_resolution: float = 2.6
    fam_maturity_cap_rework: float = 2.8
    fam_maturity_floor: float = -1.2
    fam_fail_penalty: float = 0.10
    fam_success_weight_cap: float = 8.0
    fam_failure_weight_cap: float = 8.0
    fam_cooldown_seconds_review_wave: float = 240.0
    fam_cooldown_seconds_default: float = 180.0

    # --- Family bonus carry-over (module_operational_metrics) ------------
    family_bonus_cap_base: float = 0.55
    family_bonus_cap_decay_per_fail: float = 0.18
    family_bonus_carry_fraction: float = 0.35
    family_rel_carry_cap_base: float = 0.12
    family_rel_carry_cap_decay_per_fail: float = 0.03
    family_rel_carry_scale: float = 0.20
    family_fail_carry_fraction: float = 0.30
    family_fail_carry_cap: float = 1.25
    family_cooldown_carry_fraction: float = 0.35
    family_cooldown_carry_cap: float = 180.0

    # --- Decay windows ----------------------------------------------------
    default_feedback_decay_window: float = 21600.0
    default_evidence_decay_window: float = 86400.0


# ---------------------------------------------------------------------------
# Adaptive gate calibration (consumed by PromotionLadder)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateStageThresholds:
    """Per-stage adaptive thresholds. v14.2.6 inline defaults."""
    required_evidence: int
    min_success_rate: float
    max_budget_pressure: float
    max_recent_failure_weight: float


@dataclass(frozen=True)
class AdaptiveGateCalibration:
    """All numeric constants used by PromotionLadder adaptive gates.

    Field names mirror the role each plays in `_adaptive_gate_checks`.
    Defaults match v14.2.6 exactly.
    """

    # Baseline thresholds when desired_stage doesn't match a named bucket.
    baseline: GateStageThresholds = GateStageThresholds(
        required_evidence=1,
        min_success_rate=0.0,
        max_budget_pressure=1.0,
        max_recent_failure_weight=99.0,
    )

    # shadow_deployed bucket.
    shadow_deployed: GateStageThresholds = GateStageThresholds(
        required_evidence=2,
        min_success_rate=0.45,
        max_budget_pressure=0.95,
        max_recent_failure_weight=3.0,
    )

    # live_promoted / revertable / main bucket.
    live_promoted: GateStageThresholds = GateStageThresholds(
        required_evidence=3,
        min_success_rate=0.60,
        max_budget_pressure=0.80,
        max_recent_failure_weight=1.75,
    )

    # promotable bucket.
    promotable: GateStageThresholds = GateStageThresholds(
        required_evidence=1,
        min_success_rate=0.25,
        max_budget_pressure=1.0,
        max_recent_failure_weight=5.0,
    )

    # Mature sources get a one-evidence discount on harder stages.
    maturity_bonus_threshold: float = 1.0
    maturity_evidence_discount: int = 1

    # Dynamic delta floor for live promotion.
    live_dynamic_delta_floor: float = -1.0

    def thresholds_for(self, desired_stage: str) -> GateStageThresholds:
        """Return the stage-specific thresholds bucket."""
        if desired_stage in {"shadow_deployed"}:
            return self.shadow_deployed
        if desired_stage in {"live_promoted", "revertable", "main"}:
            return self.live_promoted
        if desired_stage in {"promotable"}:
            return self.promotable
        return self.baseline


# Module-level singletons (the v14.2.6 numerics).
DEFAULT_FEEDBACK_CALIBRATION = ModuleFeedbackCalibration()
DEFAULT_GATE_CALIBRATION = AdaptiveGateCalibration()


__all__ = [
    "ModuleFeedbackCalibration",
    "AdaptiveGateCalibration",
    "GateStageThresholds",
    "DEFAULT_FEEDBACK_CALIBRATION",
    "DEFAULT_GATE_CALIBRATION",
]
