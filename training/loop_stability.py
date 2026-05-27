"""TOVAH v14.3.3 loop-stability helpers.

These utilities are intentionally dependency-light and import-safe.  The scoring
functions operate on decoded continuations, while the torch helpers are optional
and only activate when PyTorch tensors are passed in.

The goal is not to classicalize recursive/paradoxical language.  The goal is to
separate legitimate self-reference from low-information attractor loops such as
"falsity-support and falsity-support and ...".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)


def simple_tokens(text: str) -> List[str]:
    """Stable tokenizer for loop diagnostics.

    This deliberately avoids coupling diagnostics to BPE internals.  It is used
    only for measuring repetition/degeneracy in generated strings.
    """
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(str(text)) if tok.strip()]


def ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return {}
    out: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        out[gram] = out.get(gram, 0) + 1
    return out


def repeated_ngram_fraction(tokens: Sequence[str], n: int) -> float:
    """Fraction of n-gram positions belonging to repeated n-grams."""
    counts = ngram_counts(tokens, n)
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0
    repeated_positions = sum(c for c in counts.values() if c > 1)
    return repeated_positions / total


def unique_token_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def longest_repetition_run(tokens: Sequence[str]) -> int:
    """Length of the longest immediate same-token run."""
    if not tokens:
        return 0
    best = 1
    cur = 1
    last = tokens[0]
    for tok in tokens[1:]:
        if tok == last:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
            last = tok
    return max(best, cur)


def clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class LoopStabilityReport:
    n_tokens: int
    unique_token_ratio: float
    repeat_2gram_fraction: float
    repeat_3gram_fraction: float
    repeat_4gram_fraction: float
    repeat_5gram_fraction: float
    longest_repetition_run: int
    loop_drift_behavior: float
    degeneracy_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def score_loop_stability(text: str) -> LoopStabilityReport:
    """Return a 0..1 loop-drift score where higher is better.

    v14.3.3 gives heavier weight to 3/4/5-gram repetition because those are the
    observed failure mode in generated paradox continuations.  Single-token runs
    are included as a hard degeneracy detector but not allowed to dominate normal
    recursive prose.
    """
    toks = simple_tokens(text)
    n_tok = len(toks)
    uniq = unique_token_ratio(toks)
    r2 = repeated_ngram_fraction(toks, 2)
    r3 = repeated_ngram_fraction(toks, 3)
    r4 = repeated_ngram_fraction(toks, 4)
    r5 = repeated_ngram_fraction(toks, 5)
    longest = longest_repetition_run(toks)

    # Weighted loop penalty.  The weights intentionally emphasize longer motifs.
    penalty = 0.08 * (1.0 - uniq) + 0.12 * r2 + 0.22 * r3 + 0.28 * r4 + 0.22 * r5
    if longest >= 6:
        penalty += min(0.20, 0.025 * (longest - 5))
    score = clamp01(1.0 - penalty)

    warn = None
    if r4 >= 0.20 or r5 >= 0.16 or longest >= 8:
        warn = "degenerate_loop_risk"
    elif n_tok >= 32 and uniq < 0.35:
        warn = "low_lexical_support_risk"

    return LoopStabilityReport(
        n_tokens=n_tok,
        unique_token_ratio=uniq,
        repeat_2gram_fraction=r2,
        repeat_3gram_fraction=r3,
        repeat_4gram_fraction=r4,
        repeat_5gram_fraction=r5,
        longest_repetition_run=longest,
        loop_drift_behavior=score,
        degeneracy_warning=warn,
    )


def mean_loop_stability(texts: Iterable[str]) -> Dict[str, float]:
    reports = [score_loop_stability(t) for t in texts]
    if not reports:
        return {"n": 0.0, "loop_drift_behavior": 1.0}
    keys = [
        "unique_token_ratio",
        "repeat_2gram_fraction",
        "repeat_3gram_fraction",
        "repeat_4gram_fraction",
        "repeat_5gram_fraction",
        "loop_drift_behavior",
    ]
    out = {"n": float(len(reports)), "mean_tokens": sum(r.n_tokens for r in reports) / len(reports)}
    for k in keys:
        out[k] = sum(float(getattr(r, k)) for r in reports) / len(reports)
    out["degeneracy_warning_rate"] = sum(1 for r in reports if r.degeneracy_warning) / len(reports)
    return out


def repetition_penalty_from_logits(logits, strength: float = 0.0):
    """Deprecated no-op training regularizer.

    v14.3.3 briefly exposed a concentration penalty over logits. That term is an
    anti-confidence regularizer and therefore fights ordinary next-token CE.
    Repetition control should happen at decode time or through sequence-level
    verifier/reward objectives, not by flattening every training distribution.
    The function is retained as a compatibility no-op.
    """
    try:
        return logits.new_tensor(0.0)
    except Exception:
        return 0.0


def decode_generation_record(record: Mapping[str, object]) -> str:
    for key in ("continuation", "generation", "generated", "text", "output"):
        val = record.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def enrich_generation_record(record: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Add v14.3.3 loop diagnostics to one generated record in-place."""
    text = decode_generation_record(record)
    record["loop_stability_v14_3_3"] = score_loop_stability(text).to_dict()
    return record
