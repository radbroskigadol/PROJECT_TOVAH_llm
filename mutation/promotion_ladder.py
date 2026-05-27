"""
TOVAH v14 mutation/promotion_ladder.py — Promotion ladder.

THE ONLY PATH TO LIVE DEPLOYMENT.

Stages: proposed → static_approved → sandbox_passed → regression_passed
        → shadow_deployed → live_promoted → revertable

Each stage has an explicit gate. No stage can be skipped.
Every transition is logged. Quarantine on failure.

The kernel calls advance() to move a patch one stage forward.
The kernel calls apply_live() ONLY when a patch reaches live_promoted stage.
"""
from __future__ import annotations

import datetime as dt
import inspect
import logging
import time
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tovah_v14.config.constants import PROMOTION_STAGES
from tovah_v14.core.contracts import ALLOWED_TARGETS_UNIFIED, PROTECTED_METHODS, CONTRACT_REGISTRY
from tovah_v14.debug.trace_writer import evict_records
from tovah_v14.modules.calibration import (
    AdaptiveGateCalibration,
    DEFAULT_GATE_CALIBRATION,
)
from tovah_v14.mutation.analysis import analyze_patch_code, analyze_patch_with_contract
from tovah_v14.tasks.worker_roles import evaluate_promotion_target, profile_for


@dataclass
class PromotionRecord:
    """Records a promotion stage transition. Fully auditable."""
    patch_name: str
    from_stage: str
    to_stage: str
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    gate_result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class PromotionLadder:
    """Manages patch promotion through the staged pipeline.

    HARD RULES:
    - No stage can be skipped
    - Every transition is recorded
    - Failure at any stage → patch stays at current stage or gets quarantined
    - apply_live is the ONLY way to bind a function to the kernel class
    """

    def __init__(self, gate_calibration: Optional["AdaptiveGateCalibration"] = None) -> None:
        self.state: Dict[str, str] = {}  # patch_name -> current stage
        self.history: List[PromotionRecord] = []
        self.evidence_log: Dict[str, List[Dict[str, Any]]] = {}
        self.source_metadata: Dict[str, Dict[str, Any]] = {}
        self.gate_log: List[Dict[str, Any]] = []
        # AUDIT FIX (v14.2.7, sec 2): adaptive-gate thresholds now come from
        # an AdaptiveGateCalibration profile. Default = v14.2.6 numerics.
        self.gate_calibration: "AdaptiveGateCalibration" = (
            gate_calibration if gate_calibration is not None else DEFAULT_GATE_CALIBRATION
        )
        # Optional kernel-side hook for continuous corpus export.
        # Receives (decision_dict, patch_name) for every gate-log append.
        # Failures swallowed by ladder; never affects gate behaviour.
        self.on_gate_decision: Optional[Callable[[Dict[str, Any], str], None]] = None

    def _emit_gate_decision(self, decision: Dict[str, Any], patch_name: str) -> None:
        """Best-effort dispatch to optional gate-decision subscriber."""
        if self.on_gate_decision is None:
            return
        try:
            self.on_gate_decision(decision, patch_name)
        except Exception as e:
            logging.debug(f"PromotionLadder.on_gate_decision subscriber failed: {e}")

    def current_stage(self, patch_name: str) -> str:
        return self.state.get(patch_name, "proposed")

    def _record(self, patch_name: str, from_stage: str, to_stage: str,
                gate_result: str = "", details: Dict[str, Any] | None = None) -> None:
        rec = PromotionRecord(
            patch_name=patch_name, from_stage=from_stage, to_stage=to_stage,
            gate_result=gate_result, details=details or {},
        )
        self.history.append(rec)
        # AUDIT FIX (v14.2.7, sec 4): persist-on-evict before truncation.
        if len(self.history) > 600:  # cap=500 + cushion=100; batch evict for I/O efficiency
            evict_records("promotion_history", self.history[:-500])
            self.history = self.history[-500:]

    def record_evidence(
        self,
        patch_name: str,
        kind: str,
        *,
        source_kernel_id: str = "",
        trust_level: str = "",
        risk_class: str = "",
        details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        entry = {
            "kind": kind,
            "source_kernel_id": source_kernel_id,
            "trust_level": trust_level,
            "risk_class": risk_class,
            "details": dict(details or {}),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.evidence_log.setdefault(patch_name, []).append(entry)
        # AUDIT FIX (v14.2.7, sec 4): persist-on-evict. Tag each evicted
        # entry with its patch_name so the on-disk trace remains queryable.
        if len(self.evidence_log[patch_name]) > 150:  # cap=100 + cushion=50; batch evict for I/O efficiency
            overflow = self.evidence_log[patch_name][:-100]
            evict_records(
                "promotion_evidence",
                [{"patch_name": patch_name, **e} for e in overflow],
            )
            self.evidence_log[patch_name] = self.evidence_log[patch_name][-100:]
        return entry

    def set_source_metadata(self, patch_name: str, **metadata: Any) -> Dict[str, Any]:
        current = self.source_metadata.setdefault(patch_name, {})
        current.update({k: v for k, v in metadata.items() if v not in (None, "")})
        return dict(current)

    def evidence_for(self, patch_name: str) -> List[Dict[str, Any]]:
        return list(self.evidence_log.get(patch_name, []))

    def summary(self, patch_name: str) -> Dict[str, Any]:
        return {
            "patch_name": patch_name,
            "stage": self.current_stage(patch_name),
            "evidence_count": len(self.evidence_log.get(patch_name, [])),
            "source_metadata": dict(self.source_metadata.get(patch_name, {})),
        }

    def _source_context(self, patch_name: str) -> Dict[str, Any]:
        meta = dict(self.source_metadata.get(patch_name, {}))
        cooldown_until = float(meta.get("cooldown_until", 0.0) or 0.0)
        now = time.time()
        # AUDIT FIX (v14.2.7, sec 1 / RC-1 hardening):
        # Previously, when no source_metadata was registered, the patch was
        # treated as sovereign-main with risk=low, granting full bypass of
        # the adaptive evidence/budget/failure gates. This was fail-OPEN:
        # any code path that staged a patch name without first calling
        # `set_source_metadata` got silent immunity.
        #
        # The fix inverts the default: empty metadata → provisional trust,
        # medium risk, subkernel role. All legitimate sovereign paths
        # (`direct_inject_method`, `_stage_patch_proposal`) explicitly set
        # sovereign metadata BEFORE the gate, so they are unaffected. Any
        # unaccounted path now defaults to the strictest provisional track
        # and gets a warning log to surface the missing registration.
        has_meta = bool(meta)
        if not has_meta:
            logging.warning(
                "PromotionLadder: patch_name=%r reached _source_context with "
                "no source_metadata; defaulting to provisional/subkernel/medium. "
                "If this patch should be sovereign-main, call "
                "set_source_metadata() before the gate.",
                patch_name,
            )
        default_role = "subkernel"
        default_trust = "provisional"
        default_risk = "medium"
        return {
            "source_role": str(meta.get("source_role") or default_role),
            "trust_level": str(meta.get("trust_level") or default_trust),
            "locality": str(meta.get("source_locality", "local") or "local"),
            "risk_class": str(meta.get("risk_level", meta.get("risk_class")) or default_risk),
            "source_kernel_id": str(meta.get("source_kernel_id", "") or ""),
            "outcome_success_rate": float(meta.get("outcome_success_rate", 1.0) or 0.0),
            "budget_pressure": float(meta.get("budget_pressure", 0.0) or 0.0),
            "dynamic_delta": float(meta.get("dynamic_delta", 0.0) or 0.0),
            "recent_failure_weight": float(meta.get("recent_failure_weight", 0.0) or 0.0),
            "maturity_bonus": float(meta.get("maturity_bonus", 0.0) or 0.0),
            "cooldown_until": cooldown_until,
            "cooldown_remaining": max(0.0, cooldown_until - now),
            "_has_metadata": has_meta,
        }

    def _adaptive_gate_checks(self, patch_name: str, *, desired_stage: str, ctx: Dict[str, Any], base_allowed: bool, base_reason: str) -> Dict[str, Any]:
        evidence_count = len(self.evidence_for(patch_name))
        success_rate = float(ctx.get("outcome_success_rate", 1.0) or 0.0)
        budget_pressure = float(ctx.get("budget_pressure", 0.0) or 0.0)
        dynamic_delta = float(ctx.get("dynamic_delta", 0.0) or 0.0)
        recent_failure_weight = float(ctx.get("recent_failure_weight", 0.0) or 0.0)
        cooldown_remaining = float(ctx.get("cooldown_remaining", 0.0) or 0.0)
        maturity_bonus = float(ctx.get("maturity_bonus", 0.0) or 0.0)
        # Sovereign-main is the authoritative source: it self-attests and is
        # not subject to the adaptive evidence/budget/failure-rate checks
        # that exist to police external (subkernel/hub) proposals. The base
        # policy gate (evaluate_promotion_target) still applies.
        is_sovereign_main = (
            str(ctx.get("source_role", "")).lower() == "main"
            and str(ctx.get("trust_level", "")).lower() == "sovereign"
        )
        allowed = bool(base_allowed)
        reason = str(base_reason or "policy_ok")

        # AUDIT FIX (v14.2.7, sec 2 — Phase 1): per-stage thresholds now
        # come from a named AdaptiveGateCalibration profile instead of
        # inline literals. The numerics are bit-identical to v14.2.6;
        # see modules/calibration.py for provenance.
        thr = self.gate_calibration.thresholds_for(desired_stage)
        required_evidence = thr.required_evidence
        min_success_rate = thr.min_success_rate
        max_budget_pressure = thr.max_budget_pressure
        max_recent_failure_weight = thr.max_recent_failure_weight

        # Mature sources get a small evidence discount, but never below 1.
        maturity_discount = (
            self.gate_calibration.maturity_evidence_discount
            if maturity_bonus >= self.gate_calibration.maturity_bonus_threshold
            and desired_stage in {"shadow_deployed", "live_promoted", "revertable", "main"}
            else 0
        )
        effective_required_evidence = max(1, required_evidence - maturity_discount)
        if is_sovereign_main:
            # Sovereign authority bypasses the adaptive checks entirely.
            effective_required_evidence = 0

        if allowed and not is_sovereign_main and desired_stage in {"shadow_deployed", "live_promoted", "revertable", "main"} and cooldown_remaining > 0.0:
            allowed = False
            reason = "source_on_cooldown"
        if allowed and evidence_count < effective_required_evidence:
            allowed = False
            reason = "insufficient_evidence"
        if allowed and not is_sovereign_main and success_rate < min_success_rate:
            allowed = False
            reason = "low_outcome_success_rate"
        if allowed and not is_sovereign_main and budget_pressure > max_budget_pressure:
            allowed = False
            reason = "source_pressure_too_high"
        if allowed and not is_sovereign_main and recent_failure_weight > max_recent_failure_weight:
            allowed = False
            reason = "recent_failures_too_heavy"
        if allowed and not is_sovereign_main and desired_stage in {"live_promoted", "revertable", "main"} and dynamic_delta < self.gate_calibration.live_dynamic_delta_floor:
            allowed = False
            reason = "trust_degraded_below_baseline"

        return {
            "allowed": allowed,
            "reason": reason,
            "evidence_count": evidence_count,
            "required_evidence": required_evidence,
            "effective_required_evidence": effective_required_evidence,
            "outcome_success_rate": success_rate,
            "min_success_rate": min_success_rate,
            "budget_pressure": budget_pressure,
            "max_budget_pressure": max_budget_pressure,
            "dynamic_delta": dynamic_delta,
            "recent_failure_weight": recent_failure_weight,
            "max_recent_failure_weight": max_recent_failure_weight,
            "cooldown_remaining": cooldown_remaining,
            "maturity_bonus": maturity_bonus,
        }

    def assess_stage_transition_gate(self, patch_name: str, *, to_stage: str, target: str = "") -> Dict[str, Any]:
        ctx = self._source_context(patch_name)
        artifact_target = target or ("main" if to_stage in {"live_promoted", "revertable"} else "hub")
        profile = profile_for(ctx["source_role"], locality=ctx["locality"])
        decision = evaluate_promotion_target(
            target=artifact_target,
            profile=profile,
            trust_level=ctx["trust_level"],
            risk_class=ctx["risk_class"],
            locality=ctx["locality"],
            artifact_kind="patch",
        )
        adaptive = self._adaptive_gate_checks(
            patch_name,
            desired_stage=artifact_target if artifact_target == "main" else to_stage,
            ctx=ctx,
            base_allowed=bool(decision.get("allowed", False)),
            base_reason=str(decision.get("reason", "policy_ok")),
        )
        decision.update(adaptive)
        decision.update({
            "patch_name": patch_name,
            "to_stage": to_stage,
            "source_kernel_id": ctx["source_kernel_id"],
        })
        self.gate_log.append(dict(decision))
        if len(self.gate_log) > 250:  # cap=200 + cushion=50; batch evict for I/O efficiency
            evict_records("promotion_gate_log", self.gate_log[:-200])
            self.gate_log = self.gate_log[-200:]
        self._emit_gate_decision(dict(decision), patch_name)
        return decision


    def assess_request_gate(
        self,
        patch_name: str,
        *,
        source_kernel_id: str = "",
        trust_level: str = "untrusted",
        desired_stage: str = "promotable",
        risk_class: str = "medium",
        evidence: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        evidence = list(evidence or [])
        trust_scores = {"untrusted": 0, "low": 1, "provisional": 2, "trusted": 3, "sovereign": 4}
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        tscore = int(trust_scores.get(str(trust_level or "").lower(), 0))
        desired_stage = str(desired_stage or "promotable")
        risk_class = str(risk_class or "medium")
        required_trust = "provisional"
        if desired_stage in {"live_promoted", "revertable", "main", "shadow_deployed"}:
            required_trust = "trusted"
        allowed = tscore >= trust_scores.get(required_trust, 0)
        reason = "policy_ok" if allowed else "trust_below_required_stage"
        if allowed and risk_scores.get(risk_class, 2) >= 3 and tscore < trust_scores.get("trusted", 3):
            allowed = False
            reason = "high_risk_requires_trusted"
        ctx = self._source_context(patch_name)
        if source_kernel_id and not ctx.get("source_kernel_id"):
            ctx["source_kernel_id"] = source_kernel_id
        if trust_level:
            ctx["trust_level"] = trust_level
        if risk_class:
            ctx["risk_class"] = risk_class
        if evidence:
            # temporary combine request evidence with existing evidence
            self.evidence_log.setdefault(patch_name, []).extend([
                {
                    "kind": "request_evidence",
                    "source_kernel_id": source_kernel_id,
                    "trust_level": trust_level,
                    "risk_class": risk_class,
                    "details": {"value": item},
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                }
                for item in evidence
            ])
            self.evidence_log[patch_name] = self.evidence_log[patch_name][-100:]
        adaptive = self._adaptive_gate_checks(
            patch_name,
            desired_stage=desired_stage,
            ctx=ctx,
            base_allowed=allowed,
            base_reason=reason,
        )
        report = {
            "patch_name": patch_name,
            "source_kernel_id": source_kernel_id,
            "trust_level": trust_level,
            "desired_stage": desired_stage,
            "risk_class": risk_class,
            "required_trust": required_trust,
            "allowed": adaptive["allowed"],
            "reason": adaptive["reason"],
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "evidence_count": adaptive["evidence_count"],
            "required_evidence": adaptive["required_evidence"],
            "outcome_success_rate": adaptive["outcome_success_rate"],
            "min_success_rate": adaptive["min_success_rate"],
            "budget_pressure": adaptive["budget_pressure"],
            "max_budget_pressure": adaptive["max_budget_pressure"],
            "dynamic_delta": adaptive["dynamic_delta"],
        }
        self.gate_log.append(report)
        if len(self.gate_log) > 250:  # cap=200 + cushion=50; batch evict for I/O efficiency
            evict_records("promotion_gate_log", self.gate_log[:-200])
            self.gate_log = self.gate_log[-200:]
        self._emit_gate_decision(dict(report), patch_name)
        return report

    def export_state(self) -> Dict[str, Any]:
        return {
            "state": dict(self.state),
            "history": [r.__dict__ for r in self.history],
            "evidence_log": {k: list(v) for k, v in self.evidence_log.items()},
            "source_metadata": {k: dict(v) for k, v in self.source_metadata.items()},
            "gate_log": list(self.gate_log[-200:]),
        }

    def restore_state(self, data: Dict[str, Any] | None) -> None:
        self.state = {}
        self.history = []
        self.evidence_log = {}
        self.source_metadata = {}
        self.gate_log = []
        if not isinstance(data, dict):
            return
        self.state = {str(k): str(v) for k, v in dict(data.get("state", {})).items()}
        for rec in list(data.get("history", [])):
            try:
                self.history.append(PromotionRecord(**dict(rec)))
            except Exception:
                pass
        self.evidence_log = {str(k): list(v) for k, v in dict(data.get("evidence_log", {})).items()}
        self.source_metadata = {str(k): dict(v) for k, v in dict(data.get("source_metadata", {})).items()}
        self.gate_log = list(data.get("gate_log", []))[-200:]

    def advance(
        self,
        patch_name: str,
        staged_patches: Dict[str, Dict[str, Any]],
        *,
        sandbox_runner: Optional[Callable[[str], Tuple[bool, str]]] = None,
        regression_runner: Optional[Callable[[], Tuple[int, int, Dict[str, bool]]]] = None,
        # HoTT integration (v14.2.0): optional kernel-state provider for
        # generating PatchCertificates at the regression_passed →
        # shadow_deployed transition. If None, the gate runs as before
        # (policy-only). If callable, it must return a (pre_state, post_state)
        # pair; we'll run certify_patch() against the protected invariants.
        kernel_state_provider: Optional[Callable[[], Tuple[Any, Any]]] = None,
        hott_probes: Optional[List[Any]] = None,
    ) -> Tuple[str, str]:
        """Advance a patch one stage through the promotion ladder.

        Returns (new_stage, message).

        Gates:
        - proposed → static_approved: static analysis + contract validation pass
        - static_approved → sandbox_passed: sandbox execution succeeds
        - sandbox_passed → regression_passed: regression suite passes
        - regression_passed → shadow_deployed: policy + optional HoTT
            patch certificate (v14.2.0). When kernel_state_provider is
            given, we model the transition as a Path between kernel states
            and demand that protected invariants transport across it.
            Verdicts:
              - "pass" / "warn" → advance to shadow_deployed
              - "block_refuted" → blocked, protected invariant refuted
              - "block_paradox" → blocked, K-class contradiction on invariant
        - shadow_deployed → live_promoted: requires explicit apply_live call
        - live_promoted → revertable: marking only
        """
        rec = staged_patches.get(patch_name)
        if not rec:
            return self.current_stage(patch_name), f"no staged patch: {patch_name}"

        current = self.current_stage(patch_name)
        target = str(rec.get("target", "")).strip()
        code = str(rec.get("code", "")).strip()

        if current == "proposed":
            # Gate: static analysis + contract
            overall_ok, fn_names, errors, contract_ok = analyze_patch_with_contract(target, code)
            if overall_ok and target in fn_names:
                self.state[patch_name] = "static_approved"
                self._record(patch_name, "proposed", "static_approved", "passed")
                return "static_approved", "passed static + contract analysis"
            self._record(patch_name, "proposed", "proposed", "failed", {"errors": errors})
            return "proposed", f"static/contract failed: {'; '.join(errors[:3])}"

        elif current == "static_approved":
            # Gate: sandbox execution
            if sandbox_runner is not None:
                ok, msg = sandbox_runner(code)
                if ok:
                    self.state[patch_name] = "sandbox_passed"
                    self._record(patch_name, "static_approved", "sandbox_passed", "passed")
                    return "sandbox_passed", f"sandbox ok: {msg}"
                self._record(patch_name, "static_approved", "static_approved", "failed", {"error": msg})
                return "static_approved", f"sandbox failed: {msg}"
            # No sandbox runner → BLOCK
            self._record(patch_name, "static_approved", "static_approved", "blocked (no sandbox runner)")
            return "static_approved", "blocked: no sandbox runner"

        elif current == "sandbox_passed":
            # Gate: regression suite
            if regression_runner is not None:
                passed, total, details = regression_runner()
                if passed >= total - 1:  # allow 1 failure tolerance
                    self.state[patch_name] = "regression_passed"
                    self._record(patch_name, "sandbox_passed", "regression_passed", "passed",
                                {"passed": passed, "total": total})
                    return "regression_passed", f"regression: {passed}/{total}"
                self._record(patch_name, "sandbox_passed", "sandbox_passed", "failed",
                            {"passed": passed, "total": total})
                return "sandbox_passed", f"regression failed: {passed}/{total}"
            # No runner → BLOCK
            self._record(patch_name, "sandbox_passed", "sandbox_passed", "blocked (no regression runner)")
            return "sandbox_passed", "blocked: no regression runner"

        elif current == "regression_passed":
            gate = self.assess_stage_transition_gate(patch_name, to_stage="shadow_deployed")
            if not gate.get("allowed", False):
                self._record(patch_name, "regression_passed", "regression_passed", "blocked-policy",
                             {"reason": gate.get("reason", "")})
                return "regression_passed", f"blocked by policy: {gate.get('reason', 'policy_gate')}"

            # v14.2.0: optional HoTT patch certificate.
            hott_cert_summary: Optional[Dict[str, Any]] = None
            if kernel_state_provider is not None:
                try:
                    pre_state, post_state = kernel_state_provider()
                    from tovah_v14.hott import (
                        Patch as _HPatch,
                        certify_patch as _certify,
                        default_probes as _defaults,
                    )
                    from tovah_v14.core.primitives import BilateralValue as _BV
                    probes = hott_probes if hott_probes is not None else _defaults()
                    patch = _HPatch(
                        name=patch_name,
                        source_state=pre_state,
                        target_state=post_state,
                        diff_witness={"target": target, "code_head": code[:200]},
                        bilateral=_BV(0.8, 0.1),  # passed cheap gates → moderate T
                    )
                    cert = _certify(patch, probes)
                    hott_cert_summary = cert.to_dict()
                    if cert.verdict == "block_refuted":
                        self._record(
                            patch_name, "regression_passed", "regression_passed",
                            "blocked-hott-refuted",
                            {"hott_certificate": hott_cert_summary},
                        )
                        return ("regression_passed",
                                f"blocked by HoTT: {cert.verdict_reason}")
                    if cert.verdict == "block_paradox":
                        self._record(
                            patch_name, "regression_passed", "regression_passed",
                            "blocked-hott-paradox",
                            {"hott_certificate": hott_cert_summary},
                        )
                        return ("regression_passed",
                                f"blocked by HoTT (K-class): {cert.verdict_reason}")
                    # cert.verdict in {"pass", "warn"}: proceed, log warn.
                except Exception as e:
                    import logging
                    logging.error(
                        "HoTT certification raised on %s: %s — blocking promotion",
                        patch_name, e,
                    )
                    hott_cert_summary = {"error": str(e), "fail_closed": True}
                    self._record(
                        patch_name, "regression_passed", "regression_passed",
                        "blocked-hott-error",
                        {"hott_certificate": hott_cert_summary},
                    )
                    return ("regression_passed",
                            f"blocked by HoTT certification error: {e}")

            self.state[patch_name] = "shadow_deployed"
            extra: Dict[str, Any] = {"gate_reason": gate.get("reason", "policy_ok")}
            if hott_cert_summary is not None:
                extra["hott_certificate"] = hott_cert_summary
            self._record(patch_name, "regression_passed", "shadow_deployed", "auto", extra)
            return "shadow_deployed", "shadow deployed"

        elif current == "shadow_deployed":
            # Cannot advance further without explicit apply_live call
            return "shadow_deployed", "awaiting apply_live call"

        elif current == "live_promoted":
            self.state[patch_name] = "revertable"
            self._record(patch_name, "live_promoted", "revertable", "auto")
            return "revertable", "marked revertable"

        return current, "terminal stage"

    def apply_live(
        self,
        patch_name: str,
        staged_patches: Dict[str, Dict[str, Any]],
        kernel_class: type,
        original_methods: Dict[str, Any],
        evolved_method_names: set,
    ) -> Tuple[bool, str]:
        """Apply a shadow_deployed patch to the live kernel class.

        THIS IS THE ONLY WAY TO BIND A FUNCTION TO THE KERNEL.
        Only callable when current stage is shadow_deployed.

        Returns (ok, message).
        """
        current = self.current_stage(patch_name)
        if current != "shadow_deployed":
            return False, f"cannot apply: stage is {current}, need shadow_deployed"

        rec = staged_patches.get(patch_name)
        if not rec:
            return False, f"no staged patch: {patch_name}"

        gate = self.assess_stage_transition_gate(patch_name, to_stage="live_promoted", target="main")
        if not gate.get("allowed", False):
            self._record(patch_name, "shadow_deployed", "shadow_deployed", "blocked-policy", {"reason": gate.get("reason", "")})
            return False, f"blocked by policy: {gate.get('reason', 'policy_gate')}"

        target = str(rec.get("target", "")).strip()
        code = str(rec.get("code", "")).strip()

        if target in PROTECTED_METHODS:
            self._record(patch_name, "shadow_deployed", "shadow_deployed", "blocked-protected")
            return False, f"protected: {target}"

        if target not in ALLOWED_TARGETS_UNIFIED:
            self._record(patch_name, "shadow_deployed", "shadow_deployed", "blocked-not-allowed")
            return False, f"not in ALLOWED_TARGETS_UNIFIED: {target}"

        # Final static check
        ok, fn_names, errs = analyze_patch_code(code)
        if not ok or target not in fn_names:
            self._record(patch_name, "shadow_deployed", "shadow_deployed", "blocked-analysis",
                        {"errors": errs})
            return False, f"final analysis failed: {'; '.join(errs[:3])}"

        # Build exec environment
        import json, time, math, re as re_mod, hashlib, random, traceback, os, requests
        from tovah_v14.core.primitives import BilateralValue, bilateral_or, bilateral_recover
        from tovah_v14.core.cache import refresh_state, is_cache_coherent, gamma_cache
        from tovah_v14.core.state import ShadowState
        from tovah_v14.core.lanes import lane_project
        from tovah_v14.core.contracts import PROTECTED_METHODS as PM, ALLOWED_PATCH_TARGETS as APT
        from tovah_v14.tools.result import ToolResult
        from pathlib import Path
        from typing import List, Dict, Any, Optional, Tuple

        env = {
            "json": json, "time": time, "math": math, "re": re_mod, "logging": logging,
            "requests": requests, "traceback": traceback, "os": os, "hashlib": hashlib,
            "Path": Path, "ToolResult": ToolResult,
            "BilateralValue": BilateralValue, "bilateral_or": bilateral_or,
            "bilateral_recover": bilateral_recover, "refresh_state": refresh_state,
            "is_cache_coherent": is_cache_coherent, "analyze_patch_code": analyze_patch_code,
            "PROTECTED_METHODS": PM, "MAX_RESEARCH_RESULTS_STORED": 300,
            "ALLOWED_PATCH_TARGETS": APT,
            "List": List, "Dict": Dict, "Any": Any, "Optional": Optional, "Tuple": Tuple,
            "lane_project": lane_project, "gamma_cache": gamma_cache,
            "ShadowState": ShadowState, "random": random,
        }
        local_ns: Dict[str, Any] = {}

        try:
            exec(compile(code, f"<patch:{patch_name}>", "exec"), env, local_ns)
        except Exception as e:
            self._record(patch_name, "shadow_deployed", "shadow_deployed", "exec-failed",
                        {"error": str(e)})
            return False, f"exec failed: {e}"

        fn = local_ns.get(target)
        if not isinstance(fn, types.FunctionType):
            self._record(patch_name, "shadow_deployed", "shadow_deployed", "not-a-function")
            return False, "not a function"

        # Signature compatibility check
        old_fn = getattr(kernel_class, target, None)
        if callable(old_fn):
            try:
                old_req = [
                    p for p in inspect.signature(old_fn).parameters.values()
                    if p.name != "self" and p.default is inspect._empty
                    and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                new_req = [
                    p for p in inspect.signature(fn).parameters.values()
                    if p.name != "self" and p.default is inspect._empty
                    and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(new_req) > len(old_req):
                    self._record(patch_name, "shadow_deployed", "shadow_deployed", "sig-incompatible")
                    return False, f"sig incompatible: {len(new_req)}>{len(old_req)}"
            except Exception as e:
                self._record(patch_name, "shadow_deployed", "shadow_deployed", "sig-check-error",
                            {"error": str(e)})
                return False, f"sig check: {e}"

        # Save original
        if target not in original_methods:
            orig = getattr(kernel_class, target, None)
            if orig:
                original_methods[target] = orig

        # Bind to class
        setattr(kernel_class, target, fn)
        evolved_method_names.add(target)

        # Mark as live
        self.state[patch_name] = "live_promoted"
        rec["status"] = "applied"
        rec["applied_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self._record(patch_name, "shadow_deployed", "live_promoted", "applied")

        logging.info(f"LIVE PROMOTED: {patch_name} -> {target}")
        return True, f"applied {patch_name}"

    def revert(
        self,
        target: str,
        kernel_class: type,
        original_methods: Dict[str, Any],
        evolved_method_names: set,
        staged_patches: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Revert a live-promoted method to its original implementation."""
        if target not in original_methods:
            return False, f"no original for: {target}"

        setattr(kernel_class, target, original_methods[target])
        evolved_method_names.discard(target)

        # Update staged patch records
        for pn, rec in staged_patches.items():
            if rec.get("target") == target and rec.get("status") == "applied":
                rec["status"] = "reverted"
                if pn in self.state:
                    self.state[pn] = "reverted"

        self._record(f"REVERT_{target}", "live_promoted", "reverted", "reverted")
        logging.info(f"REVERTED: {target}")
        return True, f"reverted {target}"
