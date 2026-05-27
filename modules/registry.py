"""
TOVAH v14 modules/registry.py — Module registry.

Manages module manifests, branch-local proposals, and health status.
Future distribution: this becomes the service registry.
Currently: typed container with health queries and governed proposal intake.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.state import ShadowState
from tovah_v14.kernel.action_model import ModuleProposal
from tovah_v14.debug.trace_writer import evict_records
from tovah_v14.modules.calibration import (
    DEFAULT_FEEDBACK_CALIBRATION,
    ModuleFeedbackCalibration,
)
from tovah_v14.modules.manifests import MODULE_MANIFESTS, ModuleManifest
from tovah_v14.tasks.worker_roles import evaluate_promotion_target, profile_for


@dataclass
class ModuleProposalRecord:
    """Governed record for a module proposal or branch-local promotion."""

    proposal_id: str
    proposer_kernel_id: str
    module_name: str
    module_kind: str
    target_role: str = "hub"
    rationale: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    risk_class: str = "medium"
    promotion_target: str = "hub"
    requires_approval: bool = True
    status: str = "proposed"
    branch_local: bool = True
    packet_id: str = ""
    packet_kind: str = ""
    source_node_id: str = ""
    source_role: str = "subkernel"
    trust_level: str = "provisional"
    created_at: float = field(default_factory=time.time)
    reviewed_at: Optional[float] = None
    approved_by: str = ""
    notes: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModuleRegistry:
    @staticmethod
    def _trust_score(level: str) -> int:
        return {"untrusted": 0, "low": 1, "provisional": 2, "trusted": 3, "sovereign": 4}.get(str(level or "").lower(), 0)

    @staticmethod
    def _evidence_quality_total(
        items: List[Dict[str, Any]],
        *,
        now: float | None = None,
        decay_window: float = 86400.0,
        calibration: Optional[ModuleFeedbackCalibration] = None,
    ) -> float:
        # AUDIT FIX (v14.2.7, sec 2 — Phase 1): the 0.55 / 0.45 / 0.55 / 0.80
        # weights are now named fields on ModuleFeedbackCalibration. Defaults
        # are bit-identical to v14.2.6; see modules/calibration.py.
        cal = calibration if calibration is not None else DEFAULT_FEEDBACK_CALIBRATION
        total = 0.0
        stale_weak_penalty = 0.0
        fresh_strong_credit = 0.0
        now = float(now or time.time())
        window = max(1.0, float(decay_window or cal.default_evidence_decay_window))
        for item in list(items or []):
            try:
                quality = max(0.1, float(item.get("evidence_quality", item.get("quality", 1.0)) or 1.0))
            except Exception:
                quality = 1.0
            ts = item.get("time", item.get("created_at", item.get("timestamp", now)))
            try:
                age = max(0.0, now - float(ts or now))
            except Exception:
                age = 0.0
            decay = 0.5 ** (age / window)
            freshness = max(0.0, 1.0 - (age / window))
            weighted = quality * decay
            total += weighted
            if quality < 1.0:
                stale_weak_penalty += (
                    (1.0 - quality)
                    * max(0.0, 1.0 - decay)
                    * cal.stale_weak_penalty_weight
                )
            elif quality > 1.0:
                fresh_strong_credit += (
                    (quality - 1.0)
                    * (
                        cal.fresh_strong_credit_base
                        + cal.fresh_strong_credit_scale * max(decay, freshness)
                    )
                )
        cancellation = min(
            stale_weak_penalty,
            fresh_strong_credit * cal.cancellation_fraction_of_credit,
        )
        return max(0.0, total - max(0.0, stale_weak_penalty - cancellation))

    @staticmethod
    def required_evidence_for_target(target: str) -> int:
        target = str(target or "hub")
        if target in {"main", "live_promoted", "revertable"}:
            return 2
        return 1

    def __init__(self, feedback_calibration: Optional[ModuleFeedbackCalibration] = None) -> None:
        self.manifests: Dict[str, ModuleManifest] = dict(MODULE_MANIFESTS)
        self.experimental_manifests: Dict[str, ModuleManifest] = {}
        self.proposals: Dict[str, ModuleProposalRecord] = {}
        self.proposal_history: List[Dict[str, Any]] = []
        self.branch_local_modules: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.module_feedback: Dict[str, Dict[str, Any]] = {}
        self.module_family_feedback: Dict[str, Dict[str, Any]] = {}
        # AUDIT FIX (v14.2.7, sec 2): feedback coefficients are now lifted
        # into a named profile. Phase 1 is consumed by _evidence_quality_total
        # only; the remaining apply_module_feedback / module_operational_metrics
        # constants are exposed on the profile and will be consumed in a
        # follow-up commit. The Phase 1 surface change is API-only — there
        # is no numerical drift vs v14.2.6.
        self.feedback_calibration: ModuleFeedbackCalibration = (
            feedback_calibration
            if feedback_calibration is not None
            else DEFAULT_FEEDBACK_CALIBRATION
        )

    @staticmethod
    def family_key_for(module_name: str = "", module_kind: str = "") -> str:
        kind = str(module_kind or "").strip().lower()
        if kind:
            return f"kind:{kind}"
        name = str(module_name or "").strip().lower()
        if not name:
            return "family:unknown"
        for sep in ("_", ".", "-"):
            if sep in name:
                return f"name:{name.split(sep, 1)[0]}"
        return f"name:{name[:12]}"

    def _family_feedback_state(self, family_key: str) -> Dict[str, Any]:
        current = dict(self.module_family_feedback.get(str(family_key or "family:unknown"), {}))
        now = time.time()
        last = float(current.get("feedback_last_at", now) or now)
        decay_window = float(current.get("feedback_decay_window", 21600.0) or 21600.0)
        elapsed = max(0.0, now - last)
        decay = 0.5 ** (elapsed / max(1.0, decay_window))
        maturity_bonus = float(current.get("maturity_bonus", 0.0) or 0.0) * decay
        recent_failure_weight = float(current.get("recent_failure_weight", 0.0) or 0.0) * decay
        recent_success_weight = float(current.get("recent_success_weight", 0.0) or 0.0) * decay
        evidence_reentry_credit = float(current.get("evidence_reentry_credit", 0.0) or 0.0) * decay
        cooldown_until = float(current.get("cooldown_until", 0.0) or 0.0)
        cooldown_remaining = max(0.0, cooldown_until - now)
        reliability_score = (recent_success_weight + 1.0) / max(2.0, recent_success_weight + recent_failure_weight + 2.0)
        return {
            **current,
            "feedback_last_at": last,
            "feedback_decay_window": decay_window,
            "elapsed": elapsed,
            "decay": decay,
            "maturity_bonus": maturity_bonus,
            "recent_failure_weight": recent_failure_weight,
            "recent_success_weight": recent_success_weight,
            "evidence_reentry_credit": evidence_reentry_credit,
            "cooldown_until": cooldown_until,
            "cooldown_remaining": cooldown_remaining,
            "reliability_score": reliability_score,
        }

    def _feedback_state(self, module_name: str) -> Dict[str, Any]:
        current = dict(self.module_feedback.get(str(module_name or ""), {}))
        now = time.time()
        last = float(current.get("feedback_last_at", now) or now)
        decay_window = float(current.get("feedback_decay_window", 21600.0) or 21600.0)
        elapsed = max(0.0, now - last)
        decay = 0.5 ** (elapsed / max(1.0, decay_window))
        maturity_bonus = float(current.get("maturity_bonus", 0.0) or 0.0) * decay
        recent_failure_weight = float(current.get("recent_failure_weight", 0.0) or 0.0) * decay
        recent_success_weight = float(current.get("recent_success_weight", 0.0) or 0.0) * decay
        cooldown_until = float(current.get("cooldown_until", 0.0) or 0.0)
        cooldown_remaining = max(0.0, cooldown_until - now)
        reliability_score = (recent_success_weight + 1.0) / max(2.0, recent_success_weight + recent_failure_weight + 2.0)
        return {
            **current,
            "feedback_last_at": last,
            "feedback_decay_window": decay_window,
            "elapsed": elapsed,
            "decay": decay,
            "maturity_bonus": maturity_bonus,
            "recent_failure_weight": recent_failure_weight,
            "recent_success_weight": recent_success_weight,
            "cooldown_until": cooldown_until,
            "cooldown_remaining": cooldown_remaining,
            "reliability_score": reliability_score,
        }

    def apply_module_feedback(self, module_name: str, *, success: bool, severity: str = "normal", kind: str = "module_review", target: str = "", metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        module_name = str(module_name or "")
        severity = str(severity or "normal")
        meta = dict(metadata or {})
        family_key = self.family_key_for(module_name, str(meta.get("module_kind", "") or ""))
        weight = {"low": 0.5, "minor": 0.5, "normal": 1.0, "high": 1.5, "severe": 2.0, "critical": 2.5}.get(severity, 1.0)
        quality = max(0.1, float(meta.get("rework_quality", meta.get("evidence_quality", 1.0)) or 1.0))
        if kind in {"proposal_rework", "evidence_gather", "review_wave_resolution", "review_wave_auto_close"}:
            weight *= min(2.0, 0.75 + 0.55 * quality)
        current = self._feedback_state(module_name)
        maturity_bonus = float(current.get("maturity_bonus", 0.0))
        recent_failure_weight = float(current.get("recent_failure_weight", 0.0))
        recent_success_weight = float(current.get("recent_success_weight", 0.0))
        evidence_reentry_credit = float(current.get("evidence_reentry_credit", 0.0) or 0.0)
        cooldown_until = float(current.get("cooldown_until", 0.0))
        now = time.time()
        if success:
            recent_success_weight = min(8.0, recent_success_weight + weight)
            recent_failure_weight = max(0.0, recent_failure_weight - (0.40 * weight))
            maturity_bonus = min(2.5, maturity_bonus + (0.20 * weight))
            if kind == "proposal_rework":
                maturity_bonus = min(2.8, maturity_bonus + (0.16 * weight))
                recent_failure_weight = max(0.0, recent_failure_weight - (0.15 * weight))
            if kind == "evidence_gather":
                evidence_reentry_credit = min(2.5, evidence_reentry_credit + (0.35 * quality))
            if cooldown_until > now and kind in {"module_review", "module_promotion_request", "review_wave_auto_close", "review_wave_resolution", "proposal_rework"}:
                cooldown_until = now
        else:
            recent_failure_weight = min(8.0, recent_failure_weight + weight)
            maturity_bonus = max(-1.5, maturity_bonus - (0.16 * weight))
            evidence_reentry_credit = max(0.0, evidence_reentry_credit - (0.20 * weight))
            if kind in {"module_review", "module_promotion_request"} or str(target) in {"main", "live_promoted", "revertable"}:
                base_seconds = 600.0 if str(target) in {"main", "live_promoted", "revertable"} else 120.0
                cooldown_until = max(cooldown_until, now + (base_seconds * weight))
        reliability_score = (recent_success_weight + 1.0) / max(2.0, recent_success_weight + recent_failure_weight + 2.0)
        update = {
            "feedback_last_at": now,
            "feedback_decay_window": float(current.get("feedback_decay_window", 21600.0) or 21600.0),
            "maturity_bonus": maturity_bonus,
            "recent_failure_weight": recent_failure_weight,
            "recent_success_weight": recent_success_weight,
            "evidence_reentry_credit": evidence_reentry_credit,
            "cooldown_until": cooldown_until,
            "reliability_score": reliability_score,
            "module_name": module_name,
            "family_key": family_key,
            "recent_rework_quality": quality if kind == "proposal_rework" else float(current.get("recent_rework_quality", 0.0) or 0.0),
            "recent_evidence_quality": quality if kind == "evidence_gather" else float(current.get("recent_evidence_quality", 0.0) or 0.0),
        }
        if meta:
            update.update(meta)
        self.module_feedback[module_name] = update
        fam = self._family_feedback_state(family_key)
        fam_multiplier = 0.50 if (success and kind == "evidence_gather") else 0.35
        if kind == "proposal_rework":
            fam_multiplier = 0.62 if success else 0.40
        if kind == "review_wave":
            fam_multiplier = 0.60 if success else 0.55
        fam_weight = fam_multiplier * weight
        fam_success = float(fam.get("recent_success_weight", 0.0))
        fam_fail = float(fam.get("recent_failure_weight", 0.0))
        fam_bonus = float(fam.get("maturity_bonus", 0.0))
        fam_cooldown = float(fam.get("cooldown_until", 0.0))
        if success:
            fam_success = min(8.0, fam_success + fam_weight)
            fam_fail = max(0.0, fam_fail - (0.20 * fam_weight))
            fam_bonus = min(2.0, fam_bonus + (0.10 * fam_weight))
            if kind == "review_wave":
                fam_bonus = min(2.5, fam_bonus + (0.08 * fam_weight))
            if kind in {"review_wave_auto_close", "review_wave_resolution"}:
                fam_bonus = min(2.6, fam_bonus + (0.10 * fam_weight))
                if fam_cooldown > now and fam_success >= 1.0:
                    fam_cooldown = now
            if kind == "proposal_rework":
                fam_bonus = min(2.8, fam_bonus + (0.12 * fam_weight))
                fam_fail = max(0.0, fam_fail - (0.18 * fam_weight))
                if fam_cooldown > now and fam_success >= 0.8:
                    fam_cooldown = now
        else:
            fam_fail = min(8.0, fam_fail + fam_weight)
            fam_bonus = max(-1.2, fam_bonus - (0.10 * fam_weight))
            if str(target) in {"main", "live_promoted", "revertable"} or kind == "review_wave":
                fam_cooldown = max(fam_cooldown, now + ((240.0 if kind == "review_wave" else 180.0) * fam_weight))
        self.module_family_feedback[family_key] = {
            "family_key": family_key,
            "feedback_last_at": now,
            "feedback_decay_window": float(fam.get("feedback_decay_window", 21600.0) or 21600.0),
            "maturity_bonus": fam_bonus,
            "recent_failure_weight": fam_fail,
            "recent_success_weight": fam_success,
            "cooldown_until": fam_cooldown,
            "reliability_score": (fam_success + 1.0) / max(2.0, fam_success + fam_fail + 2.0),
        }
        self.proposal_history.append({"event": "module_feedback", "module_name": module_name, "family_key": family_key, "time": now, "success": bool(success), "kind": kind, "target": target})
        if len(self.proposal_history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("module_proposal_history", self.proposal_history[:-500])
            self.proposal_history = self.proposal_history[-500:]
        return dict(update)

    def module_operational_metrics(self, module_name: str) -> Dict[str, Any]:
        state = self._feedback_state(module_name)
        family_key = str(state.get("family_key") or self.family_key_for(module_name, str(state.get("module_kind", "") or "")))
        fam = self._family_feedback_state(family_key)
        local_bonus = float(state.get("maturity_bonus", 0.0))
        local_fail = float(state.get("recent_failure_weight", 0.0))
        local_success = float(state.get("recent_success_weight", 0.0))
        fam_bonus = float(fam.get("maturity_bonus", 0.0))
        fam_fail = float(fam.get("recent_failure_weight", 0.0))
        fam_success = float(fam.get("recent_success_weight", 0.0))
        local_rel = float(state.get("reliability_score", 0.5))
        fam_rel = float(fam.get("reliability_score", 0.5))
        family_bonus_cap = max(0.0, 0.55 - 0.18 * local_fail)
        family_bonus_carry = min(0.35 * fam_bonus, family_bonus_cap)
        family_rel_carry_cap = max(0.0, 0.12 - 0.03 * local_fail)
        family_rel_carry = min(family_rel_carry_cap, max(0.0, 0.20 * (fam_rel - 0.5)))
        effective_rel = min(1.0, max(0.0, local_rel + family_rel_carry))
        effective_cooldown = max(float(state.get("cooldown_remaining", 0.0)), min(180.0, 0.35 * float(fam.get("cooldown_remaining", 0.0))))
        effective_failure_weight = local_fail + min(1.25, 0.30 * fam_fail)
        return {
            "module_name": str(module_name or ""),
            "family_key": family_key,
            "recent_rework_quality": float(state.get("recent_rework_quality", 0.0) or 0.0),
            "recent_evidence_quality": float(state.get("recent_evidence_quality", 0.0) or 0.0),
            "maturity_bonus": local_bonus,
            "family_maturity_bonus": fam_bonus,
            "family_bonus_cap": family_bonus_cap,
            "family_bonus_carry": family_bonus_carry,
            "effective_maturity_bonus": local_bonus + family_bonus_carry,
            "recent_failure_weight": local_fail,
            "family_recent_failure_weight": fam_fail,
            "effective_failure_weight": effective_failure_weight,
            "recent_success_weight": local_success,
            "family_recent_success_weight": fam_success,
            "evidence_reentry_credit": float(state.get("evidence_reentry_credit", 0.0) or 0.0),
            "cooldown_until": float(state.get("cooldown_until", 0.0)),
            "cooldown_remaining": float(state.get("cooldown_remaining", 0.0)),
            "family_cooldown_remaining": float(fam.get("cooldown_remaining", 0.0)),
            "effective_cooldown_remaining": effective_cooldown,
            "reliability_score": local_rel,
            "family_reliability_score": fam_rel,
            "family_reliability_carry": family_rel_carry,
            "effective_reliability_score": effective_rel,
        }

    def proposal_priority(self, proposal_id: str, *, target: str = "") -> Dict[str, Any]:
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return {"proposal_id": proposal_id, "score": -999.0, "reason": "unknown_proposal"}
        maturity = self.maturity_report(proposal_id, target=target or rec.promotion_target)
        metrics = self.module_operational_metrics(rec.module_name)
        cooldown_remaining = float(metrics.get("effective_cooldown_remaining", 0.0))
        score = 4.0 * float(maturity.get("maturity_score", 0.0)) + 1.5 * float(metrics.get("effective_reliability_score", 0.5)) + 0.6 * float(metrics.get("effective_maturity_bonus", 0.0)) - 0.9 * float(metrics.get("effective_failure_weight", 0.0)) - (10.0 if cooldown_remaining > 0.0 else 0.0)
        return {"proposal_id": proposal_id, "module_name": rec.module_name, "target": target or rec.promotion_target, "status": rec.status, "score": score, "cooldown_remaining": cooldown_remaining, "maturity": maturity, "metrics": metrics}

    def prioritized_proposals(self, status: str = "") -> List[Dict[str, Any]]:
        items = []
        for rec in self.proposals.values():
            if status and rec.status != status:
                continue
            items.append(self.proposal_priority(rec.proposal_id, target=rec.promotion_target))
        items.sort(key=lambda item: (item.get("score", -999.0), item.get("maturity", {}).get("maturity_score", 0.0), item.get("module_name", "")), reverse=True)
        return items

    def family_readiness_summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        family_keys = set(self.module_family_feedback.keys())
        family_keys.update(self.family_key_for(p.module_name, p.module_kind) for p in self.proposals.values())
        for family_key in sorted(family_keys):
            fam = self._family_feedback_state(family_key)
            score = 1.4 * float(fam.get("reliability_score", 0.5)) + 0.5 * float(fam.get("maturity_bonus", 0.0)) - 0.8 * float(fam.get("recent_failure_weight", 0.0)) - (8.0 if float(fam.get("cooldown_remaining", 0.0)) > 0.0 else 0.0)
            rows.append({
                "family_key": family_key,
                "score": score,
                "maturity_bonus": float(fam.get("maturity_bonus", 0.0)),
                "recent_failure_weight": float(fam.get("recent_failure_weight", 0.0)),
                "recent_success_weight": float(fam.get("recent_success_weight", 0.0)),
                "cooldown_remaining": float(fam.get("cooldown_remaining", 0.0)),
                "reliability_score": float(fam.get("reliability_score", 0.5)),
            })
        rows.sort(key=lambda item: (item.get("score", -999.0), item.get("family_key", "")), reverse=True)
        return rows

    def list_modules(self, include_experimental: bool = True) -> List[str]:
        names = set(self.manifests.keys())
        if include_experimental:
            names.update(self.experimental_manifests.keys())
        return sorted(names)

    def describe(self, role: str) -> Dict[str, Any]:
        m = self.manifests.get(role) or self.experimental_manifests.get(role)
        if m is None:
            return {"error": f"unknown module: {role}"}
        proposal = next((p for p in self.proposals.values() if p.module_name == role), None)
        return {"role": m.role, "health_key": m.health_key, "version": m.version, "methods": m.methods, "depends_on": m.depends_on, "interface_inputs": m.interface_inputs, "interface_outputs": m.interface_outputs, "status": m.status, "experimental": role in self.experimental_manifests, "proposal_id": proposal.proposal_id if proposal else "", "module_metrics": self.module_operational_metrics(role)}

    def health_summary(self, state: ShadowState) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        all_manifests = {**self.manifests, **self.experimental_manifests}
        for role, manifest in all_manifests.items():
            bv = state.beta.get(manifest.health_key, BilateralValue(0.5, 0.2))
            summary[role] = {"t": bv.t, "f": bv.f, "delta": bv.delta, "glut": bv.glut}
        return summary

    def weakest(self, state: ShadowState, limit: int = 3) -> List[str]:
        all_manifests = {**self.manifests, **self.experimental_manifests}
        scored = []
        for role, manifest in all_manifests.items():
            bv = state.beta.get(manifest.health_key, BilateralValue(0.5, 0.2))
            scored.append((bv.t - bv.f, role))
        scored.sort()
        return [role for _, role in scored[:max(1, limit)]]

    def dependency_graph(self) -> Dict[str, List[str]]:
        """Return {role: [depends_on_role, ...]} for every registered manifest.

        Combines core and experimental manifests. Used by health/diagnostic
        tooling to walk the module dependency DAG.
        """
        all_manifests = {**self.manifests, **self.experimental_manifests}
        return {role: list(m.depends_on) for role, m in all_manifests.items()}

    def propose(self, proposal: ModuleProposal | Dict[str, Any], *, source_kernel_id: str = "", packet_id: str = "", packet_kind: str = "", trust_level: str = "provisional", branch_local: bool = True, source_node_id: str = "", source_role: str = "subkernel", evidence: Optional[List[Dict[str, Any]]] = None) -> ModuleProposalRecord:
        if isinstance(proposal, dict):
            proposal = ModuleProposal(**proposal)
        proposer = proposal.proposer_kernel_id or source_kernel_id or "unknown"
        rec = ModuleProposalRecord(proposal_id=proposal.proposal_id, proposer_kernel_id=proposer, module_name=proposal.module_name, module_kind=proposal.module_kind, target_role=proposal.target_role, rationale=proposal.rationale, capabilities=list(proposal.capabilities), dependencies=list(proposal.dependencies), risk_class=proposal.risk_class, promotion_target=proposal.promotion_target, requires_approval=proposal.requires_approval, branch_local=branch_local, packet_id=packet_id, packet_kind=packet_kind, source_node_id=source_node_id or (f"node_{proposer}" if proposer else ""), source_role=source_role or ("hub" if proposer == "hub" else ("main" if proposer == "main" else "subkernel")), trust_level=trust_level, evidence=list(evidence or []))
        self.proposals[rec.proposal_id] = rec
        self.proposal_history.append({"event": "proposed", "proposal_id": rec.proposal_id, "time": rec.created_at, "module_name": rec.module_name})
        if len(self.proposal_history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("module_proposal_history", self.proposal_history[:-500])
            self.proposal_history = self.proposal_history[-500:]
        if branch_local:
            self.branch_local_modules.setdefault(proposer, {})[rec.module_name] = {"status": rec.status, "proposal_id": rec.proposal_id, "module_kind": rec.module_kind, "promotion_target": rec.promotion_target}
        family_key = self.family_key_for(rec.module_name, rec.module_kind)
        self.module_feedback.setdefault(rec.module_name, {"module_name": rec.module_name, "family_key": family_key, "feedback_last_at": rec.created_at, "feedback_decay_window": 21600.0, "maturity_bonus": 0.0, "recent_failure_weight": 0.0, "recent_success_weight": 0.0, "cooldown_until": 0.0, "reliability_score": 0.5})
        self.module_family_feedback.setdefault(family_key, {"family_key": family_key, "feedback_last_at": rec.created_at, "feedback_decay_window": 21600.0, "maturity_bonus": 0.0, "recent_failure_weight": 0.0, "recent_success_weight": 0.0, "cooldown_until": 0.0, "reliability_score": 0.5})
        return rec

    def assess_promotion_gate(self, proposal_id: str, *, trust_level: str = "provisional", locality: str = "local", target: str = "") -> Dict[str, Any]:
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return {"allowed": False, "reason": "unknown_proposal", "proposal_id": proposal_id}
        profile = profile_for(rec.source_role or "subkernel", specialization=rec.module_kind, locality=locality)
        target = target or rec.promotion_target or "hub"
        decision = evaluate_promotion_target(target=target, profile=profile, trust_level=trust_level or rec.trust_level, risk_class=rec.risk_class, locality=locality, artifact_kind="module")
        metrics = self.module_operational_metrics(rec.module_name)
        if float(metrics.get("effective_cooldown_remaining", 0.0)) > 0.0:
            decision["allowed"] = False
            decision["reason"] = "module_on_cooldown"
        elif target in {"main", "live_promoted", "revertable"} and float(metrics.get("effective_failure_weight", 0.0)) >= 2.0:
            decision["allowed"] = False
            decision["reason"] = "module_recent_failures"
        decision.update({"proposal_id": proposal_id, "module_name": rec.module_name, "source_node_id": rec.source_node_id, "source_role": rec.source_role, "required_evidence": self.required_evidence_for_target(target), "current_evidence": len(rec.evidence), "module_metrics": metrics})
        rec.evidence.append({"kind": "policy_gate", **decision})
        rec.evidence = rec.evidence[-100:]
        self.proposal_history.append({"event": "policy_gate", "proposal_id": proposal_id, "time": time.time(), "allowed": decision.get("allowed", False), "reason": decision.get("reason", "")})
        if len(self.proposal_history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("module_proposal_history", self.proposal_history[:-500])
            self.proposal_history = self.proposal_history[-500:]
        return decision

    def maturity_report(self, proposal_id: str, *, target: str = "") -> Dict[str, Any]:
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return {"proposal_id": proposal_id, "ready": False, "reason": "unknown_proposal"}
        target = target or rec.promotion_target or "hub"
        required_evidence = self.required_evidence_for_target(target)
        feedback = self.module_operational_metrics(rec.module_name)
        evidence_discount = min(1, int(max(0.0, float(feedback.get("effective_maturity_bonus", 0.0))) // 1.0))
        reentry_discount = min(1, int(max(0.0, float(feedback.get("evidence_reentry_credit", 0.0))) // 1.0))
        effective_required = max(1, required_evidence - evidence_discount - reentry_discount)
        evidence_count = len(rec.evidence)
        evidence_quality_total = self._evidence_quality_total(
            rec.evidence, calibration=self.feedback_calibration
        )
        tscore = self._trust_score(rec.trust_level)
        maturity_score = min(1.0, (evidence_quality_total + tscore + max(0.0, float(feedback.get("effective_maturity_bonus", 0.0)))) / max(1.0, float(effective_required + 3)))
        return {"proposal_id": proposal_id, "module_name": rec.module_name, "target": target, "evidence_count": evidence_count, "evidence_quality_total": evidence_quality_total, "required_evidence": required_evidence, "effective_required_evidence": effective_required, "evidence_discount": evidence_discount, "reentry_discount": reentry_discount, "trust_level": rec.trust_level, "maturity_score": maturity_score, "ready": evidence_quality_total >= effective_required and float(feedback.get("effective_cooldown_remaining", 0.0)) <= 0.0, "module_metrics": feedback}

    def governed_review(self, proposal_id: str, *, reviewer: str = "main", trust_level: str = "provisional", locality: str = "local", target: str = "", notes: str = "", auto_promote: bool = False) -> Dict[str, Any]:
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return {"proposal_id": proposal_id, "status": "missing", "allowed": False, "reason": "unknown_proposal"}
        target = target or rec.promotion_target
        gate = self.assess_promotion_gate(proposal_id, trust_level=trust_level or rec.trust_level, locality=locality, target=target)
        maturity = self.maturity_report(proposal_id, target=target)
        final_status = rec.status
        reason = gate.get("reason", "")
        if not gate.get("allowed", False):
            hard_block = reason in {"target_not_allowed_for_role", "main_promotion_requires_trusted", "risk_exceeds_role_budget", "trust_below_role_floor", "main_promotion_requires_locality", "module_on_cooldown", "module_recent_failures"}
            final_status = "rejected" if hard_block else "review_pending"
            self.review(proposal_id, status=final_status, reviewer=reviewer, notes=notes or reason)
        elif not maturity.get("ready", False):
            final_status = "review_pending"
            self.review(proposal_id, status=final_status, reviewer=reviewer, notes=notes or "awaiting_more_evidence")
        elif auto_promote or target == "main":
            promoted = self.promote(proposal_id, reviewer=reviewer, notes=notes or "governed promotion")
            final_status = str((promoted or {}).get("status", rec.status or "promoted"))
        else:
            self.review(proposal_id, status="approved", reviewer=reviewer, notes=notes or "governed approval")
            final_status = "approved"
        severity = "high" if target in {"main", "live_promoted", "revertable"} else "normal"
        self.apply_module_feedback(rec.module_name, success=final_status in {"approved", "promoted"}, severity=severity if final_status == "rejected" else ("low" if final_status == "approved" else "normal"), kind="module_review" if final_status != "promoted" else "module_promotion_request", target=target, metadata={"last_status": final_status, "last_target": target, "module_kind": rec.module_kind})
        outcome = {"proposal_id": proposal_id, "module_name": rec.module_name, "status": final_status, "gate": gate, "maturity": maturity, "target": target}
        rec.evidence.append({"kind": "governed_review", "status": final_status, "reviewer": reviewer, "target": target})
        rec.evidence = rec.evidence[-100:]
        self.proposal_history.append({"event": "governed_review", "proposal_id": proposal_id, "time": time.time(), "status": final_status, "target": target})
        if len(self.proposal_history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("module_proposal_history", self.proposal_history[:-500])
            self.proposal_history = self.proposal_history[-500:]
        return outcome

    def attach_evidence(self, proposal_id: str, evidence: Dict[str, Any]) -> bool:
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return False
        payload = dict(evidence)
        payload.setdefault("time", time.time())
        payload["evidence_quality"] = max(0.1, float(payload.get("evidence_quality", payload.get("quality", 1.0)) or 1.0))
        rec.evidence.append(payload)
        rec.evidence = rec.evidence[-100:]
        self.proposal_history.append({"event": "evidence", "proposal_id": proposal_id, "time": time.time(), "kind": payload.get("kind", "generic"), "evidence_quality": payload.get("evidence_quality", 1.0)})
        if len(self.proposal_history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("module_proposal_history", self.proposal_history[:-500])
            self.proposal_history = self.proposal_history[-500:]
        return True

    def review(self, proposal_id: str, *, status: str, reviewer: str = "main", notes: str = "") -> bool:
        if status not in {"review_pending", "approved", "rejected", "promoted"}:
            raise ValueError(f"unsupported proposal review status: {status}")
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return False
        rec.status = status
        rec.reviewed_at = time.time()
        rec.approved_by = reviewer
        rec.notes = notes
        self.proposal_history.append({"event": status, "proposal_id": proposal_id, "time": rec.reviewed_at, "reviewer": reviewer})
        if len(self.proposal_history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("module_proposal_history", self.proposal_history[:-500])
            self.proposal_history = self.proposal_history[-500:]
        branch = self.branch_local_modules.setdefault(rec.proposer_kernel_id, {})
        branch.setdefault(rec.module_name, {})["status"] = status
        return True

    def promote(self, proposal_id: str, *, reviewer: str = "main", notes: str = "") -> Dict[str, Any] | None:
        rec = self.proposals.get(proposal_id)
        if rec is None:
            return None
        self.review(proposal_id, status="promoted", reviewer=reviewer, notes=notes)
        manifest = ModuleManifest(role=rec.module_name, health_key=f"module.{rec.module_name}_health", version="16.0.0-candidate", methods=list(rec.capabilities), depends_on=list(rec.dependencies), interface_inputs=["packet_payload", "module_request"], interface_outputs=["module_response", "promotion_evidence"], status="candidate")
        self.experimental_manifests[rec.module_name] = manifest
        self.branch_local_modules.setdefault(rec.proposer_kernel_id, {})[rec.module_name] = {"status": "promoted", "proposal_id": rec.proposal_id, "module_kind": rec.module_kind, "promotion_target": rec.promotion_target, "reviewer": reviewer}
        return {"proposal_id": rec.proposal_id, "module_name": rec.module_name, "reviewer": reviewer, "status": rec.status, "promotion_target": rec.promotion_target, "module_metrics": self.module_operational_metrics(rec.module_name)}

    def list_proposals(self, status: str = "") -> List[Dict[str, Any]]:
        items = [p.to_dict() for p in self.proposals.values()]
        if status:
            items = [p for p in items if p.get("status") == status]
        items.sort(key=lambda p: (p.get("created_at", 0.0), p.get("proposal_id", "")))
        return items

    def export_state(self) -> Dict[str, Any]:
        return {"experimental_manifests": {k: asdict(v) for k, v in self.experimental_manifests.items()}, "proposals": {k: v.to_dict() for k, v in self.proposals.items()}, "proposal_history": list(self.proposal_history), "branch_local_modules": {k: dict(v) for k, v in self.branch_local_modules.items()}, "module_feedback": {k: dict(v) for k, v in self.module_feedback.items()}, "module_family_feedback": {k: dict(v) for k, v in self.module_family_feedback.items()}}

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        self.experimental_manifests = {}
        self.proposals = {}
        self.proposal_history = []
        self.branch_local_modules = {}
        self.module_feedback = {}
        self.module_family_feedback = {}
        if not isinstance(data, dict):
            return
        for role, payload in dict(data.get("experimental_manifests", {})).items():
            try:
                self.experimental_manifests[str(role)] = ModuleManifest(**dict(payload))
            except Exception:
                pass
        for proposal_id, payload in dict(data.get("proposals", {})).items():
            try:
                self.proposals[str(proposal_id)] = ModuleProposalRecord(**dict(payload))
            except Exception:
                pass
        self.proposal_history = list(data.get("proposal_history", []))[-500:]
        self.branch_local_modules = {str(k): dict(v) for k, v in dict(data.get("branch_local_modules", {})).items()}
        self.module_feedback = {str(k): dict(v) for k, v in dict(data.get("module_feedback", {})).items()}
        self.module_family_feedback = {str(k): dict(v) for k, v in dict(data.get("module_family_feedback", {})).items()}

    def summary(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {}
        for rec in self.proposals.values():
            status_counts[rec.status] = status_counts.get(rec.status, 0) + 1
        feedback = {name: self.module_operational_metrics(name) for name in sorted(self.module_feedback)}
        avg_bonus = (sum(v["effective_maturity_bonus"] for v in feedback.values()) / max(1, len(feedback))) if feedback else 0.0
        cooldowns = sum(1 for v in feedback.values() if float(v.get("effective_cooldown_remaining", 0.0)) > 0.0)
        family_rows = self.family_readiness_summary()
        avg_family = (sum(float(r.get("score", 0.0)) for r in family_rows) / max(1, len(family_rows))) if family_rows else 0.0
        return {"canonical_modules": len(self.manifests), "experimental_modules": len(self.experimental_manifests), "proposal_count": len(self.proposals), "status_counts": status_counts, "branches": {k: sorted(v.keys()) for k, v in self.branch_local_modules.items()}, "family_count": len(self.module_family_feedback), "cooldown_modules": cooldowns, "average_maturity_bonus": avg_bonus, "average_family_readiness": avg_family, "top_priorities": self.prioritized_proposals()[:10], "family_priorities": family_rows[:10], "recent_history": self.proposal_history[-20:]}
