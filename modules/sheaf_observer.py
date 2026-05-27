"""
TOVAH v14.2.7 modules/sheaf_observer.py — Cellular-sheaf observer.

AUDIT FIX (v14.2.7, sec 5, paradigm 3): a READ-ONLY observer that models
the ModuleRegistry dependency graph as a cellular sheaf and surfaces
localized inconsistencies (Gluts / K-mass concentrations) without
influencing any promotion or gate decisions.

STRUCTURE
---------
Vertex set V         — one node per module name from MODULE_MANIFESTS.
Edge set E ⊆ V × V   — directed edge (u, v) if v ∈ u.depends_on.
Stalk F(v)           — a triple (t, f, q) ∈ [0,1]^3 derived from module
                       operational metrics: t = effective_maturity_bonus
                       normalized, f = recent_failure_weight normalized,
                       q = evidence_quality proxy.
Restriction maps     — for each edge (u, v), a pair (ρ_uv, ρ_vu): linear
                       maps from the source stalk's value into the edge
                       stalk. The pair encodes the bilateral handshake
                       across the dependency boundary.
Edge stalk F(e_{uv}) — same ambient space; the restriction maps land here.

LOCAL CONSISTENCY
-----------------
For each edge (u, v), the sheaf laplacian's contribution at e is

    Δ(e) = || ρ_uv(F(v)) - ρ_vu(F(u)) ||²

The total disagreement (sheaf 0-cohomology obstruction) is the sum over
edges. High contributions on a small set of edges → localized Glut; flat
distribution → global drift.

OBSERVER ROLE
-------------
The observer subscribes to ladder gate decisions and registry feedback.
At each call to `assess()`, it recomputes the global obstruction and
classifies the result as ok / drift / glut. Findings go to the trace
writer ("sheaf_observer_findings"). It does NOT call set_source_metadata,
record_evidence, or any gate; it is strictly diagnostic.

LITERATURE PROVENANCE
---------------------
The "neural sheaf diffusion" line (Bodnar et al., 2022; Hansen & Gebhart,
2020) gives the formalism we lean on. The TOVAH-specific specialization
is: stalks are bilateral pairs (t, f) rather than abstract feature
vectors, and restriction maps are derived from interface_inputs /
interface_outputs overlaps rather than learned. This is the "observer"
prototype — a learned-restriction-map version is future work, scoped in
AUDIT.md §5.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tovah_v14.debug.trace_writer import TraceWriter
from tovah_v14.modules.manifests import MODULE_MANIFESTS, ModuleManifest


# ---------------------------------------------------------------------------
# Sheaf data structures
# ---------------------------------------------------------------------------

# A stalk value is a 3-vector (t, f, q) ∈ [0,1]^3.
StalkValue = Tuple[float, float, float]


@dataclass(frozen=True)
class SheafEdge:
    """Directed edge u -> v with paired restriction maps.

    Each restriction map is a 3x3 row-stochastic matrix represented as
    nine floats in row-major order. Identity by default; specialization
    happens via `restriction_for_interface_overlap`.
    """
    u: str  # source module
    v: str  # target module (dependency)
    rho_uv: Tuple[float, ...]  # 9 floats, map applied to F(v)
    rho_vu: Tuple[float, ...]  # 9 floats, map applied to F(u)
    weight: float = 1.0        # relative importance of this edge


def _identity_map() -> Tuple[float, ...]:
    return (1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0)


def _apply(m: Tuple[float, ...], v: StalkValue) -> StalkValue:
    a, b, c, d, e, f, g, h, i = m
    x, y, z = v
    return (a * x + b * y + c * z,
            d * x + e * y + f * z,
            g * x + h * y + i * z)


def _l2_sq(a: StalkValue, b: StalkValue) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def restriction_for_interface_overlap(
    u_outputs: List[str], v_inputs: List[str]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Build restriction maps from interface contract overlap.

    Heuristic (Phase 1, no learned weights):

      overlap_ratio = |outputs ∩ inputs| / max(1, |outputs ∪ inputs|)

    High overlap → restriction is close to identity (consistent handshake
    expected). Low overlap → restriction damps the t/f channels, leaving
    q (evidence quality) at higher weight — this reflects that loose
    interfaces should be judged primarily by their evidence track record.
    """
    out_set = set(s for s in (u_outputs or []) if s)
    in_set = set(s for s in (v_inputs or []) if s)
    if not out_set and not in_set:
        # No declared interface either side; assume identity.
        return _identity_map(), _identity_map()
    union = out_set | in_set
    inter = out_set & in_set
    overlap = (len(inter) / max(1, len(union))) if union else 0.0
    # alpha ∈ [0.4, 1.0]: damps t and f channels at low overlap.
    alpha = 0.4 + 0.6 * overlap
    m = (
        alpha, 0.0,   0.0,
        0.0,   alpha, 0.0,
        0.0,   0.0,   1.0,
    )
    # Both directions use the same damping by default. Asymmetric maps
    # are reserved for future learned versions.
    return m, m


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

@dataclass
class SheafFinding:
    """A single assessment snapshot."""
    timestamp: float
    obstruction_total: float
    classification: str  # "ok" | "drift" | "glut"
    hottest_edges: List[Dict[str, Any]] = field(default_factory=list)
    nonabelian_twists: List[Dict[str, Any]] = field(default_factory=list)
    n_nodes: int = 0
    n_edges: int = 0


class SheafObserver:
    """Read-only sheaf observer over the module dependency graph.

    Typical use:

        observer = SheafObserver()
        observer.bind(registry, ladder)   # attaches hooks; non-mutating
        finding = observer.assess()       # returns latest SheafFinding

    The observer is safe to call from any thread; its only mutating
    state is `self._last_findings`, which is bounded.
    """

    # Classification thresholds. Values chosen so that an all-identity
    # graph with stalks at zero produces "ok"; substantial divergence
    # produces "drift"; concentrated divergence on few edges produces
    # "glut". These are not gate thresholds — only labels for logging.
    DRIFT_OBSTRUCTION_FLOOR: float = 0.25
    GLUT_CONCENTRATION_RATIO: float = 0.50  # fraction of total in top-k edges
    GLUT_TOP_K: int = 3
    HISTORY_CAP: int = 64

    def __init__(self) -> None:
        self._registry: Any = None
        self._ladder: Any = None
        self._edges: List[SheafEdge] = []
        self._nodes: List[str] = []
        self._last_findings: List[SheafFinding] = []
        self._trace = TraceWriter.get("sheaf_observer_findings")

    # --- topology -------------------------------------------------------

    def _rebuild_topology(self, manifests: Dict[str, ModuleManifest]) -> None:
        """(Re)derive node/edge set from manifests. Idempotent."""
        nodes: List[str] = sorted(manifests.keys())
        edges: List[SheafEdge] = []
        for name in nodes:
            m = manifests[name]
            for dep in (m.depends_on or []):
                if dep not in manifests:
                    continue
                dep_m = manifests[dep]
                rho_uv, rho_vu = restriction_for_interface_overlap(
                    m.interface_outputs, dep_m.interface_inputs,
                )
                edges.append(SheafEdge(
                    u=name, v=dep, rho_uv=rho_uv, rho_vu=rho_vu, weight=1.0,
                ))
        self._nodes = nodes
        self._edges = edges

    # --- binding --------------------------------------------------------

    def bind(self, registry: Any, ladder: Optional[Any] = None) -> None:
        """Attach to a registry (required) and ladder (optional).

        The observer reads from these objects; it does not write to them.
        If a ladder is provided, its `on_gate_decision` hook is wrapped
        (not replaced) so the observer can opportunistically reassess
        after each gate decision.
        """
        self._registry = registry
        self._ladder = ladder
        manifests = getattr(registry, "manifests", None) or {}
        if manifests:
            self._rebuild_topology(manifests)
        if ladder is not None:
            prev = getattr(ladder, "on_gate_decision", None)
            def _wrapped(decision: Dict[str, Any], patch_name: str) -> None:
                if callable(prev):
                    try:
                        prev(decision, patch_name)
                    except Exception:
                        logging.debug("SheafObserver: prior gate hook raised; continuing.")
                try:
                    self.assess(reason=f"gate:{patch_name}")
                except Exception as e:
                    logging.debug("SheafObserver: assess failed inside ladder hook (%s).", e)
            ladder.on_gate_decision = _wrapped

    # --- stalk extraction ----------------------------------------------

    def _stalk_for(self, module_name: str) -> StalkValue:
        """Derive (t, f, q) ∈ [0,1]^3 from module operational metrics.

        t — effective_maturity_bonus, squashed via tanh to [0, 1].
        f — recent_failure_weight, squashed via tanh to [0, 1].
        q — evidence_credit normalized, squashed via tanh to [0, 1].

        Falls back to a neutral mid-stalk (0.5, 0.5, 0.5) if metrics are
        unavailable or raise.
        """
        if self._registry is None:
            return (0.5, 0.5, 0.5)
        try:
            metrics = self._registry.module_operational_metrics(module_name)
        except Exception:
            return (0.5, 0.5, 0.5)
        if not isinstance(metrics, dict):
            return (0.5, 0.5, 0.5)
        # Squash each channel through tanh(x / scale). Scales chosen so
        # typical operational values land in [0.2, 0.9].
        def _sq(x: float, scale: float) -> float:
            try:
                return 0.5 * (1.0 + math.tanh(float(x) / max(1e-6, scale)))
            except Exception:
                return 0.5
        t = _sq(float(metrics.get("effective_maturity_bonus", 0.0) or 0.0), 1.0)
        f = _sq(float(metrics.get("recent_failure_weight", 0.0) or 0.0), 2.0)
        q = _sq(float(metrics.get("evidence_credit", 0.0) or 0.0), 2.0)
        return (t, f, q)

    # --- main assessment -----------------------------------------------

    def assess(self, *, reason: str = "manual") -> SheafFinding:
        """Compute the global obstruction and classify the result.

        Returns a SheafFinding and appends it to the bounded history /
        on-disk trace. Safe to call frequently; cost is O(|E|).
        """
        if not self._edges:
            # Try to lazily rebuild from registry, if bound.
            if self._registry is not None:
                self._rebuild_topology(getattr(self._registry, "manifests", {}) or {})
        if not self._edges:
            finding = SheafFinding(
                timestamp=time.time(),
                obstruction_total=0.0,
                classification="ok",
                hottest_edges=[],
                nonabelian_twists=[],
                n_nodes=len(self._nodes),
                n_edges=0,
            )
            return self._record(finding, reason=reason)

        # Precompute stalk values once per assessment.
        stalks: Dict[str, StalkValue] = {n: self._stalk_for(n) for n in self._nodes}
        per_edge: List[Tuple[float, SheafEdge]] = []
        total = 0.0
        for e in self._edges:
            fu = stalks.get(e.u, (0.5, 0.5, 0.5))
            fv = stalks.get(e.v, (0.5, 0.5, 0.5))
            disagreement = _l2_sq(_apply(e.rho_uv, fv), _apply(e.rho_vu, fu))
            contribution = e.weight * disagreement
            total += contribution
            per_edge.append((contribution, e))

        per_edge.sort(key=lambda x: -x[0])
        top_k = per_edge[: self.GLUT_TOP_K]
        top_k_sum = sum(c for c, _ in top_k)
        concentration = (top_k_sum / total) if total > 0 else 0.0

        if total < self.DRIFT_OBSTRUCTION_FLOOR:
            classification = "ok"
        elif concentration >= self.GLUT_CONCENTRATION_RATIO:
            classification = "glut"
        else:
            classification = "drift"

        hottest = [
            {
                "u": e.u, "v": e.v,
                "contribution": round(c, 6),
                "weight": e.weight,
            }
            for c, e in top_k
        ]
        twists = self._triple_twists(stalks)
        if twists and classification == "ok":
            classification = "twist"
        finding = SheafFinding(
            timestamp=time.time(),
            obstruction_total=round(total, 6),
            classification=classification,
            hottest_edges=hottest,
            nonabelian_twists=twists,
            n_nodes=len(self._nodes),
            n_edges=len(self._edges),
        )
        return self._record(finding, reason=reason)

    def _triple_twists(self, stalks: Dict[str, StalkValue]) -> List[Dict[str, Any]]:
        """Localized triple-overlap twist diagnostic.

        This is a lightweight non-abelian Čech proxy: for dependency triples
        u→v→w with u→w also present, compare the composed restriction path with
        the direct path.  It surfaces pairwise-consistent but globally twisted
        module designs without mutating gates.
        """
        edges = {(e.u, e.v): e for e in self._edges}
        out: List[Dict[str, Any]] = []
        for (u, v), e_uv in edges.items():
            for (v2, w), e_vw in edges.items():
                if v2 != v or (u, w) not in edges:
                    continue
                e_uw = edges[(u, w)]
                fu = stalks.get(u, (0.5, 0.5, 0.5))
                fw = stalks.get(w, (0.5, 0.5, 0.5))
                via = _apply(e_uv.rho_vu, _apply(e_vw.rho_vu, fw))
                direct = _apply(e_uw.rho_vu, fu)
                twist = _l2_sq(via, direct)
                if twist > 0.10:
                    out.append({"u": u, "v": v, "w": w, "twist": round(twist, 6)})
        out.sort(key=lambda d: -d["twist"])
        return out[: self.GLUT_TOP_K]

    def _record(self, finding: SheafFinding, *, reason: str) -> SheafFinding:
        self._last_findings.append(finding)
        if len(self._last_findings) > self.HISTORY_CAP:
            self._last_findings = self._last_findings[-self.HISTORY_CAP:]
        # Best-effort persist; failures are swallowed by TraceWriter.
        self._trace.append({
            "timestamp": finding.timestamp,
            "reason": reason,
            "obstruction_total": finding.obstruction_total,
            "classification": finding.classification,
            "hottest_edges": finding.hottest_edges,
            "nonabelian_twists": finding.nonabelian_twists,
            "n_nodes": finding.n_nodes,
            "n_edges": finding.n_edges,
        })
        if finding.classification == "glut":
            logging.warning(
                "SheafObserver: localized inconsistency (glut) detected at edges %s "
                "(total obstruction=%.4f). This is diagnostic only — no gate impact.",
                [(d["u"], d["v"]) for d in finding.hottest_edges],
                finding.obstruction_total,
            )
        return finding

    # --- introspection --------------------------------------------------

    def history(self) -> List[SheafFinding]:
        """Return a copy of the bounded findings history."""
        return list(self._last_findings)

    def topology_summary(self) -> Dict[str, Any]:
        """Return a serializable description of the current topology."""
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "nodes": list(self._nodes),
            "edges": [
                {"u": e.u, "v": e.v, "weight": e.weight}
                for e in self._edges
            ],
        }


__all__ = [
    "SheafObserver",
    "SheafEdge",
    "SheafFinding",
    "restriction_for_interface_overlap",
]
