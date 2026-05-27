"""TOVAH v16 selfmodel/cluster_model.py — summary model of kernel ecology readiness."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClusterSelfModel:
    node_count: int = 0
    trusted_node_count: int = 0
    local_node_count: int = 0
    subkernel_count: int = 0
    hub_present: bool = False
    active_nodes: List[str] = field(default_factory=list)
    trust_alerts: List[str] = field(default_factory=list)
    average_trust_score: float = 0.0
    dynamic_trust_nodes: int = 0
    delegation_pressure: float = 0.0
    delegation_capacity: int = 0
    delegation_success_rate: float = 0.0
    cooldown_node_count: int = 0
    average_maturity_bonus: float = 0.0
    promotion_readiness: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def update_from(self, cluster_summary: Dict[str, Any], trust_summary: Dict[str, Any], *, hub_present: bool, subkernel_count: int, delegation_summary: Dict[str, Any] | None = None, promotion_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.node_count = int(cluster_summary.get("node_count", 0))
        self.trusted_node_count = int(cluster_summary.get("trusted_node_count", 0))
        self.local_node_count = int(cluster_summary.get("local_node_count", 0))
        self.subkernel_count = int(subkernel_count)
        self.hub_present = bool(hub_present)
        self.active_nodes = [evt.get("node_id", "") for evt in list(cluster_summary.get("recent_events", [])) if evt.get("node_id")]
        self.average_trust_score = float(trust_summary.get("average_trust_score", 0.0))
        self.dynamic_trust_nodes = int(trust_summary.get("dynamic_nodes", 0))
        self.cooldown_node_count = int(cluster_summary.get("cooldown_node_count", 0))
        self.average_maturity_bonus = float(cluster_summary.get("average_maturity_bonus", 0.0))
        levels = dict(trust_summary.get("levels", {}))
        delegation_summary = delegation_summary or {}
        active_delegations = int(delegation_summary.get("active_count", 0))
        self.delegation_capacity = max(0, self.subkernel_count + (1 if hub_present else 0))
        self.delegation_pressure = (active_delegations / max(1, self.delegation_capacity))
        self.delegation_success_rate = float(delegation_summary.get("success_rate", 0.0))
        promotion_summary = promotion_summary or {}
        pending_promotions = int(promotion_summary.get("pending_count", 0))
        tracked_promotions = int(promotion_summary.get("tracked_patch_count", 0))
        self.promotion_readiness = 1.0 if tracked_promotions == 0 else max(0.0, 1.0 - (pending_promotions / max(1, tracked_promotions)))
        alerts: List[str] = []
        if levels.get("untrusted", 0):
            alerts.append(f"{levels.get('untrusted', 0)} untrusted nodes")
        if self.average_trust_score < 1.5 and self.node_count:
            alerts.append("cluster trust below nominal floor")
        if self.delegation_pressure > 0.9:
            alerts.append("delegation capacity saturated")
        if self.cooldown_node_count:
            alerts.append(f"{self.cooldown_node_count} nodes cooling down")
        if self.delegation_success_rate < 0.5 and (int(delegation_summary.get("completed_count", 0)) + int(delegation_summary.get("failed_count", 0))) >= 2:
            alerts.append("delegation success degraded")
        self.trust_alerts = alerts[:10]
        self.last_updated = time.time()
        return {
            "node_count": self.node_count,
            "trusted_node_count": self.trusted_node_count,
            "dynamic_trust_nodes": self.dynamic_trust_nodes,
            "delegation_pressure": self.delegation_pressure,
            "delegation_success_rate": self.delegation_success_rate,
            "cooldown_node_count": self.cooldown_node_count,
            "average_maturity_bonus": self.average_maturity_bonus,
            "promotion_readiness": self.promotion_readiness,
            "trust_alerts": list(self.trust_alerts),
        }
