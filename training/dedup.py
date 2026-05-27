"""
TOVAH v14 training/dedup.py — Deduplicate TrainingExamples.

Implements three strategies:
  KEEP_BEST_QUALITY      — for each lineage_id keep the highest quality_score
  KEEP_MOST_RECENT       — for each lineage_id keep the highest .time
  MERGE_WITH_PROVENANCE  — collapse into one canonical example whose provenance
                           chain accumulates all merged ids; quality_score is
                           averaged, bilateral T/F are unioned (max of T,
                           max of F — contradictions become K-mass)

The default strategy is MERGE_WITH_PROVENANCE because that matches the
David-flavour: contradictions and reworks contribute *to* the canonical
example rather than being thrown out. This is the bilateral-Belnap angle.
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple

from tovah_v14.training.corpus_builder import TrainingExample


class DedupStrategy(str, Enum):
    KEEP_BEST_QUALITY = "keep_best_quality"
    KEEP_MOST_RECENT = "keep_most_recent"
    MERGE_WITH_PROVENANCE = "merge_with_provenance"


def deduplicate(examples: List[TrainingExample],
                *, strategy: DedupStrategy = DedupStrategy.MERGE_WITH_PROVENANCE
                ) -> Tuple[List[TrainingExample], Dict[str, int]]:
    """Remove duplicates per `strategy`. Returns (unique_examples, stats)."""
    by_lineage: Dict[str, List[TrainingExample]] = defaultdict(list)
    for ex in examples:
        by_lineage[ex.lineage_id].append(ex)

    duplicates_collapsed = 0
    unique: List[TrainingExample] = []

    for lineage, group in by_lineage.items():
        if len(group) == 1:
            unique.append(group[0])
            continue
        duplicates_collapsed += len(group) - 1

        if strategy == DedupStrategy.KEEP_BEST_QUALITY:
            best = max(group, key=lambda e: e.quality_score)
            unique.append(best)
        elif strategy == DedupStrategy.KEEP_MOST_RECENT:
            best = max(group, key=lambda e: e.time)
            unique.append(best)
        else:  # MERGE_WITH_PROVENANCE
            # Pick the highest-quality member as the canonical text/kind base.
            base = max(group, key=lambda e: e.quality_score)
            merged_chain = list(base.provenance_chain)
            for sib in group:
                if sib is base:
                    continue
                merged_chain.append(sib.lineage_id + "@merged")
                merged_chain.extend(sib.provenance_chain)
            # Bilateral merge: T = max(T_i), F = max(F_i). When two
            # observations of the same artefact disagree, that's by design
            # contradiction-mass (paraconsistent K).
            t = max(e.bilateral_t for e in group)
            f = max(e.bilateral_f for e in group)
            avg_q = sum(e.quality_score for e in group) / len(group)
            merged = TrainingExample(
                lineage_id=base.lineage_id,
                kind=base.kind,
                text=base.text,
                time=max(e.time for e in group),
                source_kernel_id=base.source_kernel_id,
                mission_context=base.mission_context,
                provenance_chain=sorted(set(merged_chain)),
                outcome_label=base.outcome_label,
                bilateral_t=t,
                bilateral_f=f,
                quality_score=avg_q,
                paraconsistent_class="",
                metadata={**base.metadata,
                          "merged_count": len(group),
                          "merged_outcomes": sorted(set(e.outcome_label for e in group))},
            )
            unique.append(merged)

    stats = {
        "input_total": len(examples),
        "unique": len(unique),
        "duplicates_collapsed": duplicates_collapsed,
        "dedup_ratio": float(len(unique)) / max(1, len(examples)),
    }
    return unique, stats
