"""
TOVAH v14 selfmodel/model.py — Kernel self-model.

Structured, state-linked self-model that integrates bilateral diagnostics,
competence, budgets, module health, blocked growth, failures, and capabilities.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from tovah_v14.core.primitives import BilateralValue
from tovah_v14.core.cache import refresh_state
from tovah_v14.core.state import ShadowState

@dataclass
class SelfModel:
    version: str = "14.2.6"
    known_weaknesses: List[str] = field(default_factory=list)
    pending_rewrites: List[str] = field(default_factory=list)
    subsystem_reliability: Dict[str, float] = field(default_factory=dict)
    improvement_priorities: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    degraded_mode: bool = False
    boot_valid: bool = True
    competence_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    budget_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    budget_pressure: List[str] = field(default_factory=list)
    module_health_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blocked_growth: List[Dict[str, str]] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    active_tools: List[str] = field(default_factory=list)
    active_services: List[str] = field(default_factory=list)
    patch_pipeline_ready: bool = True
    pending_growth_opportunities: int = 0
    capability_count: int = 0
    node_identity_summary: Dict[str, Any] = field(default_factory=dict)
    cluster_summary: Dict[str, Any] = field(default_factory=dict)
    trust_summary: Dict[str, Any] = field(default_factory=dict)
    distributed_queue_summary: Dict[str, Any] = field(default_factory=dict)
    delegation_confidence: float = 0.0
    delegation_success_rate: float = 0.0
    promotion_readiness: float = 0.0
    module_priority_summary: List[Dict[str, Any]] = field(default_factory=list)
    hub_review_summary: List[Dict[str, Any]] = field(default_factory=list)
    module_readiness: float = 0.0
    family_module_readiness: float = 0.0
    cooldown_pressure: float = 0.0
    growth_priority_summary: List[Dict[str, Any]] = field(default_factory=list)
    hub_wave_summary: List[Dict[str, Any]] = field(default_factory=list)


def update_self_model(
    sm: SelfModel, state: ShadowState,
    rewrite_queue: List[str] | None = None,
    competence_map: Any = None, budget_manager: Any = None,
    module_health: Any = None, blocked_growth_log: List[Dict[str, str]] | None = None,
    runtime_error_counts: Dict[str, int] | None = None,
    active_lab_tools: Dict[str, Any] | None = None,
    free_services: List[Dict[str, Any]] | None = None,
    staged_patches: Dict[str, Any] | None = None,
    promotion_ladder: Any = None,
    node_identity: Any = None, cluster_registry: Any = None, trust_ledger: Any = None,
    distributed_queue: Any = None, cluster_model: Any = None,
    module_registry: Any = None, hub_review_state: Any = None,
) -> SelfModel:
    reliability_keys = {
        "planner": "planning.confidence", "browser": "browser.reachability",
        "patcher": "patch.pipeline.health", "search": "tool.search_efficacy",
        "memory": "memory.consolidation_health", "tasks": "task.queue_health",
        "persistence": "state.coherent",
    }
    sm.subsystem_reliability = {}
    for name, beta_key in reliability_keys.items():
        bv = state.beta.get(beta_key, BilateralValue(0.5, 0.3))
        sm.subsystem_reliability[name] = float(bv.t)
    weaknesses: List[str] = []
    for key, bv in state.beta.items():
        if bv.f > 0.6:
            weaknesses.append(f"high falsity: {key} (f={bv.f:.2f})")
        if bv.glut > 0.4:
            weaknesses.append(f"high glut: {key} (glut={bv.glut:.2f})")
    sm.known_weaknesses = weaknesses[:20]
    sm.pending_rewrites = list(rewrite_queue or [])
    sm.improvement_priorities = [
        k for k, _ in sorted(sm.subsystem_reliability.items(), key=lambda kv: kv[1])
    ][:5]
    if competence_map is not None:
        sm.competence_summary = {}
        for domain, entry in getattr(competence_map, "entries", {}).items():
            sm.competence_summary[domain] = {
                "mastery": entry.measured_mastery, "tests": entry.test_count,
                "confidence_t": entry.bilateral_confidence.t, "confidence_f": entry.bilateral_confidence.f,
            }
    if budget_manager is not None:
        sm.budget_summary = {}
        sm.budget_pressure = []
        for resource, info in getattr(budget_manager, "budgets", {}).items():
            used, limit = info.get("used", 0), info.get("limit", 1)
            sm.budget_summary[resource] = {"used": used, "limit": limit}
            if limit > 0 and used / limit > 0.8:
                sm.budget_pressure.append(f"{resource}: {used}/{limit}")
    if module_health is not None:
        sm.module_health_summary = module_health.get_health_summary(state)
    sm.blocked_growth = [{"patch": bg.get("patch",""), "stage": bg.get("stage",""),
                          "reason": bg.get("reason","")} for bg in (blocked_growth_log or [])[-10:]]
    if runtime_error_counts:
        sm.recent_failures = [f"{k}: {v}" for k, v in sorted(
            runtime_error_counts.items(), key=lambda kv: -kv[1])][:10]
    sm.active_tools = sorted(active_lab_tools.keys()) if active_lab_tools else []
    sm.active_services = [s.get("name", "?") for s in (free_services or [])
                          if s.get("status") == "active"][:20]
    if promotion_ladder is not None:
        blocked = [pn for pn, stage in promotion_ladder.state.items()
                   if stage in ("static_approved", "sandbox_passed")]
        sm.patch_pipeline_ready = len(blocked) == 0
    if staged_patches:
        sm.pending_growth_opportunities = sum(
            1 for r in staged_patches.values() if r.get("status") == "staged")
    if node_identity is not None:
        sm.node_identity_summary = node_identity.summary() if hasattr(node_identity, "summary") else dict(node_identity)
    if cluster_registry is not None:
        sm.cluster_summary = cluster_registry.summary() if hasattr(cluster_registry, "summary") else dict(cluster_registry)
    if trust_ledger is not None:
        sm.trust_summary = trust_ledger.summary() if hasattr(trust_ledger, "summary") else dict(trust_ledger)
    if distributed_queue is not None:
        sm.distributed_queue_summary = distributed_queue.summary() if hasattr(distributed_queue, "summary") else dict(distributed_queue)
    if module_registry is not None:
        sm.module_priority_summary = list(module_registry.prioritized_proposals()[:10]) if hasattr(module_registry, "prioritized_proposals") else []
        if hasattr(module_registry, 'summary'):
            ms = module_registry.summary()
            top = sm.module_priority_summary
            fam = list(ms.get('family_priorities', [])[:10])
            sm.module_readiness = sum(float(item.get('score', 0.0)) for item in top) / max(1, len(top)) if top else 0.0
            sm.family_module_readiness = sum(float(item.get('score', 0.0)) for item in fam) / max(1, len(fam)) if fam else 0.0
            sm.cooldown_pressure = float(ms.get('cooldown_modules', 0)) / max(1, int(ms.get('proposal_count', 0) or 1))
            growth = []
            for item in top[:5]:
                qbonus = 0.40 * float(item.get('metrics', {}).get('recent_rework_quality', 0.0) or 0.0) + 0.25 * float(item.get('metrics', {}).get('recent_evidence_quality', 0.0) or 0.0)
                growth.append({'kind': 'module', 'name': str(item.get('module_name', '')), 'score': float(item.get('score', 0.0)) + qbonus, 'target': str(item.get('target', ''))})
            for fam_item in fam[:3]:
                growth.append({'kind': 'family', 'name': str(fam_item.get('family_key', '')), 'score': float(fam_item.get('score', 0.0)), 'target': 'family'})
            growth.sort(key=lambda d: (d.get('score', -999.0), d.get('name', '')), reverse=True)
            sm.growth_priority_summary = growth[:8]
    if hub_review_state is not None:
        if isinstance(hub_review_state, dict):
            sm.hub_review_summary = list(hub_review_state.get("queue", [])[:10])
            sm.hub_wave_summary = list(hub_review_state.get("waves", [])[:10])
            for wave in sm.hub_wave_summary[:3]:
                if float(wave.get("success_rate", 0.0) or 0.0) > 0.0:
                    sm.growth_priority_summary.append({
                        "kind": "review_wave",
                        "name": str(wave.get("wave_id", "")),
                        "score": float(wave.get("success_rate", 0.0)) + 0.1 * float(wave.get("outcome_summary", {}).get("item_count", 0)),
                        "target": "wave",
                    })
            for wavep in list(hub_review_state.get("wave_priorities", [])[:3]):
                sm.growth_priority_summary.append({
                    "kind": "open_wave",
                    "name": str(wavep.get("wave_id", "")),
                    "score": float(wavep.get("score", 0.0)),
                    "target": "wave_resolution",
                })
            for entry in list(hub_review_state.get("resolution_history", [])[:8]):
                for target in list(entry.get("targets", [])[:3]):
                    sm.growth_priority_summary.append({
                        "kind": str(target.get("kind", "")),
                        "name": str(target.get("name", "")),
                        "score": float(target.get("score", entry.get("score", 0.0))),
                        "target": str(entry.get("outcome", "wave_resolution")),
                    })
            for entry in list(hub_review_state.get("escalation_history", [])[:8]):
                for route in list(entry.get("routes", [])[:3]):
                    sm.growth_priority_summary.append({
                        "kind": str(route.get("kind", "")),
                        "name": str(route.get("name", route.get("proposal_id", ""))),
                        "score": float(entry.get("confidence", 0.0)) + 0.5,
                        "target": str(entry.get("outcome", "wave_escalation")),
                    })
            for entry in list(hub_review_state.get("proposal_rework_history", [])[:8]):
                sm.growth_priority_summary.append({
                    "kind": "proposal_rework",
                    "name": str(entry.get("module_name", entry.get("proposal_id", ""))),
                    "score": 1.15,
                    "target": str(entry.get("family_key", "proposal_rework")),
                })
            for entry in list(hub_review_state.get("blocked_growth_followup_history", [])[:8]):
                sm.growth_priority_summary.append({
                    "kind": "blocked_growth",
                    "name": str(entry.get("artifact_name", entry.get("wave_id", ""))),
                    "score": 0.85,
                    "target": str(entry.get("outcome", "blocked_growth_followup")),
                })
            sm.growth_priority_summary.sort(key=lambda d: (d.get("score", -999.0), d.get("name", "")), reverse=True)
            sm.growth_priority_summary = sm.growth_priority_summary[:10]
        else:
            sm.hub_review_summary = list(hub_review_state[:10])
    if cluster_model is not None:
        sm.delegation_confidence = float(getattr(cluster_model, "average_trust_score", 0.0)) / 4.0
        sm.delegation_success_rate = float(getattr(cluster_model, "delegation_success_rate", 0.0))
        base_prom = float(getattr(cluster_model, "promotion_readiness", 0.0))
        family_adj = max(0.0, min(1.0, 0.5 + 0.2 * sm.family_module_readiness - 0.4 * sm.cooldown_pressure)) if sm.family_module_readiness or sm.cooldown_pressure else 0.5
        sm.promotion_readiness = max(0.0, min(1.0, 0.75 * base_prom + 0.25 * family_adj))
    elif sm.trust_summary:
        sm.delegation_confidence = float(sm.trust_summary.get("average_trust_score", 0.0)) / 4.0
    sm.last_updated = time.time()
    state.beta["self_model.accuracy"] = BilateralValue(
        min(1.0, 0.5 + 0.05 * len(sm.subsystem_reliability)), 0.1).clamp()
    refresh_state(state)
    return sm
