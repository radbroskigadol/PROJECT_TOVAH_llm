"""
TOVAH v14 training/quality_filter.py — Paraconsistent A/B/K/G routing.

Each TrainingExample carries (T, F) bilateral mass. The Belnap four-valued
semantics partition the unit square into four classes:

    A  (true / agreed)         T high, F low
    B  (false / refuted)       T low,  F high
    K  (contradiction)         T high AND F high
    G  (gap / unknown)         T low  AND F low

This module assigns one class per example and lets you split a corpus
into per-class shards. Most LLM training uses A only; TOVAH's distinctive
training signal is in K (contradictions) and B (labelled negatives).

The thresholds match the kernel's own GAMMA_THETA_T/F constants (0.55)
so the corpus's class boundaries are aligned with the runtime's gamma
coherence checker.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List

from tovah_v14.config.constants import GAMMA_THETA_T, GAMMA_THETA_F
from tovah_v14.training.corpus_builder import TrainingExample


class ParaconsistentClass(str, Enum):
    A = "A"  # agreed/true
    B = "B"  # refuted/false
    K = "K"  # contradiction
    G = "G"  # gap/unknown


def classify_one(ex: TrainingExample,
                 *, theta_t: float = GAMMA_THETA_T,
                 theta_f: float = GAMMA_THETA_F) -> ParaconsistentClass:
    """Return the ABKG class for a single example."""
    t_high = ex.bilateral_t >= theta_t
    f_high = ex.bilateral_f >= theta_f
    if t_high and f_high:
        return ParaconsistentClass.K
    if t_high and not f_high:
        return ParaconsistentClass.A
    if not t_high and f_high:
        return ParaconsistentClass.B
    return ParaconsistentClass.G


def classify_examples(examples: List[TrainingExample],
                      *, theta_t: float = GAMMA_THETA_T,
                      theta_f: float = GAMMA_THETA_F) -> List[TrainingExample]:
    """Return a new list with `paraconsistent_class` filled in for each."""
    result: List[TrainingExample] = []
    for ex in examples:
        cls = classify_one(ex, theta_t=theta_t, theta_f=theta_f)
        # Build a fresh dataclass via dataclasses.replace pattern.
        ex.paraconsistent_class = cls.value
        result.append(ex)
    return result


def class_counts(examples: List[TrainingExample]) -> Dict[str, int]:
    """Count examples by paraconsistent class."""
    counts = {c.value: 0 for c in ParaconsistentClass}
    for ex in examples:
        if ex.paraconsistent_class:
            counts[ex.paraconsistent_class] = counts.get(ex.paraconsistent_class, 0) + 1
        else:
            cls = classify_one(ex)
            counts[cls.value] += 1
    return counts


def split_by_class(examples: List[TrainingExample]
                   ) -> Dict[str, List[TrainingExample]]:
    """Partition examples into {class_letter: [examples_in_that_class]}."""
    out: Dict[str, List[TrainingExample]] = {c.value: [] for c in ParaconsistentClass}
    for ex in examples:
        cls = ex.paraconsistent_class or classify_one(ex).value
        out.setdefault(cls, []).append(ex)
    return out
