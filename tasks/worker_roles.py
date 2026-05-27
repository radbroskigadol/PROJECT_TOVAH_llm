"""TOVAH v16 tasks/worker_roles.py — typed worker role policies for delegation and governance."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


RISK_SCORES: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}
TRUST_SCORES: Dict[str, int] = {"untrusted": 0, "low": 1, "provisional": 2, "trusted": 3, "sovereign": 4}
PERMISSION_SCORES: Dict[str, int] = {
    "safe_autonomous": 1,
    "safe_logged": 2,
    "sandbox_only": 3,
    "approval_required": 4,
    "forbidden": 99,
}


@dataclass
class WorkerRoleProfile:
    role: str
    specialization: str = ""
    locality: str = "local"
    trust_floor: str = "low"
    max_active_leases: int = 3
    capabilities: List[str] = field(default_factory=list)
    allowed_permission_levels: List[str] = field(default_factory=list)
    allowed_promotion_targets: List[str] = field(default_factory=list)
    max_risk_class: str = "medium"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_WORKER_PROFILES: Dict[str, WorkerRoleProfile] = {
    "main": WorkerRoleProfile(
        role="main",
        trust_floor="sovereign",
        max_active_leases=1,
        capabilities=["sovereign_coordination", "determinization", "patch_governance"],
        allowed_permission_levels=["safe_autonomous", "safe_logged", "sandbox_only", "approval_required"],
        allowed_promotion_targets=["main", "hub", "subkernel"],
        max_risk_class="high",
        notes="Authoritative coordinator; not a general delegation target.",
    ),
    "hub": WorkerRoleProfile(
        role="hub",
        trust_floor="provisional",
        max_active_leases=4,
        capabilities=["branch_experimentation", "proposal_incubation", "work_rehearsal"],
        allowed_permission_levels=["safe_autonomous", "safe_logged"],
        allowed_promotion_targets=["hub", "main"],
        max_risk_class="medium",
        notes="Experimental branch may use safe logged tools locally and propose upward.",
    ),
    "subkernel": WorkerRoleProfile(
        role="subkernel",
        trust_floor="low",
        max_active_leases=3,
        capabilities=["delegated_execution", "specialized_tasking"],
        allowed_permission_levels=["safe_autonomous"],
        allowed_promotion_targets=["hub"],
        max_risk_class="low",
        notes="Specialized child kernels may execute narrow work and propose upward to hub.",
    ),
}


def trust_score(level: str) -> int:
    return int(TRUST_SCORES.get(str(level or "").lower(), 0))


def risk_score(level: str) -> int:
    return int(RISK_SCORES.get(str(level or "").lower(), 0))


def permission_score(level: str) -> int:
    return int(PERMISSION_SCORES.get(str(level or "").lower(), 0))


def profile_for(role: str, *, specialization: str = "", locality: str = "local") -> WorkerRoleProfile:
    base = DEFAULT_WORKER_PROFILES.get(role, DEFAULT_WORKER_PROFILES["subkernel"])
    return WorkerRoleProfile(
        role=role,
        specialization=specialization,
        locality=locality,
        trust_floor=base.trust_floor,
        max_active_leases=base.max_active_leases,
        capabilities=list(base.capabilities),
        allowed_permission_levels=list(base.allowed_permission_levels),
        allowed_promotion_targets=list(base.allowed_promotion_targets),
        max_risk_class=base.max_risk_class,
        notes=base.notes,
    )


def _effective_max_risk(profile: WorkerRoleProfile, trust_level: str) -> str:
    current = risk_score(profile.max_risk_class)
    tscore = trust_score(trust_level)
    if profile.role == "hub" and tscore >= trust_score("trusted"):
        current = max(current, risk_score("high"))
    return next((label for label, score in RISK_SCORES.items() if score == current), profile.max_risk_class)


def evaluate_tool_access(
    tool_name: str,
    *,
    required_permission: str,
    profile: WorkerRoleProfile,
    trust_level: str,
    locality: str = "local",
) -> Dict[str, Any]:
    required_permission = str(required_permission or "safe_autonomous")
    decision = {
        "tool_name": tool_name,
        "role": profile.role,
        "specialization": profile.specialization,
        "locality": locality,
        "required_permission": required_permission,
        "trust_level": str(trust_level),
        "trust_floor": profile.trust_floor,
        "allowed": False,
        "reason": "blocked",
        "allowed_permissions": list(profile.allowed_permission_levels),
    }
    if required_permission == "forbidden":
        decision["reason"] = "forbidden_tool_permission"
        return decision
    if trust_score(trust_level) < trust_score(profile.trust_floor):
        decision["reason"] = "trust_below_role_floor"
        return decision
    if required_permission not in profile.allowed_permission_levels:
        if required_permission == "safe_logged" and profile.role in {"hub", "subkernel"} and trust_score(trust_level) >= trust_score("trusted") and locality == "local":
            decision["allowed"] = True
            decision["reason"] = "trusted_local_override"
            return decision
        decision["reason"] = "permission_not_allowed_for_role"
        return decision
    if required_permission == "safe_logged" and locality != "local" and profile.role != "main":
        decision["reason"] = "logged_tool_requires_locality"
        return decision
    if required_permission in {"sandbox_only", "approval_required"} and profile.role != "main":
        decision["reason"] = "reserved_for_sovereign_main"
        return decision
    decision["allowed"] = True
    decision["reason"] = "policy_ok"
    return decision


def evaluate_promotion_target(
    *,
    target: str,
    profile: WorkerRoleProfile,
    trust_level: str,
    risk_class: str = "medium",
    locality: str = "local",
    artifact_kind: str = "module",
) -> Dict[str, Any]:
    target = str(target or "hub")
    risk_class = str(risk_class or "medium")
    decision = {
        "artifact_kind": artifact_kind,
        "target": target,
        "role": profile.role,
        "specialization": profile.specialization,
        "locality": locality,
        "trust_level": str(trust_level),
        "trust_floor": profile.trust_floor,
        "risk_class": risk_class,
        "max_risk_class": _effective_max_risk(profile, trust_level),
        "allowed": False,
        "reason": "blocked",
        "allowed_targets": list(profile.allowed_promotion_targets),
    }
    if trust_score(trust_level) < trust_score(profile.trust_floor):
        decision["reason"] = "trust_below_role_floor"
        return decision
    if target not in profile.allowed_promotion_targets:
        if not (profile.role == "hub" and target == "main" and trust_score(trust_level) >= trust_score("trusted")):
            decision["reason"] = "target_not_allowed_for_role"
            return decision
    if target == "main" and profile.role != "main" and trust_score(trust_level) < trust_score("trusted"):
        decision["reason"] = "main_promotion_requires_trusted"
        return decision
    if target == "main" and locality != "local" and profile.role != "main":
        decision["reason"] = "main_promotion_requires_locality"
        return decision
    if risk_score(risk_class) > risk_score(decision["max_risk_class"]):
        decision["reason"] = "risk_exceeds_role_budget"
        return decision
    decision["allowed"] = True
    decision["reason"] = "policy_ok"
    return decision


def summarize_profiles() -> Dict[str, Dict[str, Any]]:
    return {name: profile.to_dict() for name, profile in DEFAULT_WORKER_PROFILES.items()}
