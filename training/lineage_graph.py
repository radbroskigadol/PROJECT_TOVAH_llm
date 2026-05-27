"""
TOVAH v14 training/lineage_graph.py — Provenance DAG over TrainingExamples.

Each TrainingExample's `provenance_chain` lists upstream lineage_ids.
This module turns that into a DAG that downstream trainers can consult
to condition on genealogy. Half-built already in
`memory/provenance_graph.py`; this file is the training-data exporter.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from tovah_v14.training.corpus_builder import TrainingExample


@dataclass
class LineageGraph:
    """A simple DAG over training-example lineage ids."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges_forward: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    edges_reverse: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_example(self, ex: TrainingExample) -> None:
        """Add a node for `ex` and edges from each provenance entry to ex."""
        self.nodes[ex.lineage_id] = {
            "kind": ex.kind,
            "outcome_label": ex.outcome_label,
            "quality_score": ex.quality_score,
            "paraconsistent_class": ex.paraconsistent_class,
            "time": ex.time,
        }
        for upstream in ex.provenance_chain:
            self.edges_forward[upstream].add(ex.lineage_id)
            self.edges_reverse[ex.lineage_id].add(upstream)

    def upstream(self, lineage_id: str, *, max_depth: int = 16) -> List[str]:
        """BFS upstream from a lineage id, depth-limited."""
        seen: Set[str] = set()
        out: List[str] = []
        q: deque = deque([(lineage_id, 0)])
        while q:
            node, d = q.popleft()
            if d > max_depth or node in seen:
                continue
            seen.add(node)
            for parent in self.edges_reverse.get(node, ()):
                if parent not in seen:
                    out.append(parent)
                    q.append((parent, d + 1))
        return out

    def downstream(self, lineage_id: str, *, max_depth: int = 16) -> List[str]:
        """BFS downstream from a lineage id, depth-limited."""
        seen: Set[str] = set()
        out: List[str] = []
        q: deque = deque([(lineage_id, 0)])
        while q:
            node, d = q.popleft()
            if d > max_depth or node in seen:
                continue
            seen.add(node)
            for child in self.edges_forward.get(node, ()):
                if child not in seen:
                    out.append(child)
                    q.append((child, d + 1))
        return out

    def stats(self) -> Dict[str, Any]:
        n_nodes = len(self.nodes)
        n_edges = sum(len(v) for v in self.edges_forward.values())
        chain_lens = []
        for lid in self.nodes:
            chain_lens.append(len(self.upstream(lid, max_depth=64)))
        avg_chain = (sum(chain_lens) / len(chain_lens)) if chain_lens else 0.0
        max_chain = max(chain_lens) if chain_lens else 0
        roots = [lid for lid in self.nodes if not self.edges_reverse.get(lid)]
        leaves = [lid for lid in self.nodes if not self.edges_forward.get(lid)]
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_roots": len(roots),
            "n_leaves": len(leaves),
            "avg_chain_depth": avg_chain,
            "max_chain_depth": max_chain,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": dict(self.nodes),
            "edges_forward": {k: sorted(v) for k, v in self.edges_forward.items()},
            "edges_reverse": {k: sorted(v) for k, v in self.edges_reverse.items()},
            "stats": self.stats(),
        }


def build_lineage_graph(examples: List[TrainingExample]) -> LineageGraph:
    g = LineageGraph()
    for ex in examples:
        g.add_example(ex)
    return g
