"""
TOVAH v14 memory/retrieval.py — Memory retrieval with TF-IDF-like scoring.

v14 IMPROVEMENT:
  v13 used raw word-set overlap for retrieval scoring.
  v14 uses term frequency * inverse document frequency style scoring:
  - Rare terms in the query get more weight
  - Recency and bilateral confidence contribute to ranking
  - Access count is updated on retrieval

This is the foundation of retrieval competence:
returning the right thing at the right time.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Dict, List, Optional

from tovah_v14.memory.store import MemoryEntry, MemoryStore


def _tokenize(text: str) -> List[str]:
    """Extract lowercased tokens of length >= 3."""
    return re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())


def _entry_text(entry: MemoryEntry) -> str:
    """Extract searchable text from a memory entry."""
    parts = [entry.key, " ".join(entry.tags)]
    try:
        parts.append(json.dumps(entry.data, default=str)[:500])
    except Exception:
        pass
    parts.append(entry.goal_context or "")
    return " ".join(parts)


def _compute_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency for each term."""
    n_docs = max(1, len(corpus_tokens))
    doc_freq: Dict[str, int] = {}
    for tokens in corpus_tokens:
        for t in set(tokens):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return {t: math.log(n_docs / max(1, df)) for t, df in doc_freq.items()}


def memory_query(
    store: MemoryStore,
    kind: str,
    query: str,
    limit: int = 10,
    recency_weight: float = 0.2,
    confidence_weight: float = 0.1,
) -> List[MemoryEntry]:
    """Search memory bank with TF-IDF-like scoring.

    Scoring = tfidf_overlap * 0.7 + recency * recency_weight + confidence * confidence_weight

    Updates access_count and accessed_at for returned entries.
    """
    bank = store.get_bank(kind)
    if not bank:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        # No query tokens — return most recent
        sorted_bank = sorted(bank, key=lambda e: e.created_at, reverse=True)
        for e in sorted_bank[:limit]:
            e.accessed_at = time.time()
            e.access_count += 1
        return sorted_bank[:limit]

    # Build corpus tokens for IDF
    corpus_tokens = [_tokenize(_entry_text(e)) for e in bank]
    idf = _compute_idf(corpus_tokens)

    # Score each entry
    now = time.time()
    scored: List[tuple] = []
    q_set = set(q_tokens)
    q_idf = {t: idf.get(t, 1.0) for t in q_set}
    q_idf_total = sum(q_idf.values()) or 1.0

    for entry, e_tokens in zip(bank, corpus_tokens):
        e_set = set(e_tokens)
        # TF-IDF overlap: sum of IDF for matching terms, normalized
        overlap_score = sum(q_idf.get(t, 0.0) for t in q_set & e_set) / q_idf_total

        # Recency: decays over hours
        hours_old = (now - entry.created_at) / 3600.0
        recency = 1.0 / (1.0 + hours_old)

        # Bilateral confidence
        conf = entry.bilateral_confidence.t - entry.bilateral_confidence.f * 0.5

        score = overlap_score * 0.7 + recency * recency_weight + conf * confidence_weight
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _, entry in scored[:limit]:
        entry.accessed_at = time.time()
        entry.access_count += 1
        results.append(entry)

    return results
