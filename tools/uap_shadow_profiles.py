"""TOVAH v14.3.2 UAP / ShadowHoTT token-profile labels and metrics.

This module deliberately treats AdamW-style next-token prediction as the
classicalized floor.  It adds a richer token ontology layer for paraconsistent
training/evaluation: truth-support, falsity-support, gluts, gaps, obstruction
residue, collapse pressure, and classicalization depth.

The implementation is dependency-light so it can be used during corpus
generation, validation, and smoke tests before a full model head is wired in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9_'-]+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

POSITIVE_TERMS = {
    "affirmed", "affirm", "asserted", "true", "truth", "supports", "supported",
    "proved", "proof", "verified", "valid", "accepted", "witness", "positive",
    "designated", "holds", "yes", "entails", "entailed",
}
NEGATIVE_TERMS = {
    "rejected", "reject", "false", "falsity", "refuted", "refutation", "opposes",
    "opposed", "invalid", "denied", "negative", "counterexample", "contradicts",
    "fails", "failure", "no", "anti", "objection",
}
GLUT_TERMS = {
    "both", "glut", "glutty", "contradiction", "contradictory", "inconsistent",
    "paradox", "simultaneously", "co-present", "coherent-incoherent", "overlap",
    "bilateral", "paraconsistent", "hot", "dual", "two-sided",
}
GAP_TERMS = {
    "gap", "gappy", "unknown", "undetermined", "underdetermined", "neither",
    "missing", "silent", "suspended", "unresolved", "opaque", "absence",
    "incomplete", "unproven", "no-evidence", "insufficient",
}
OBSTRUCTION_TERMS = {
    "obstruction", "obstructed", "residue", "residual", "local-global", "local",
    "global", "gluing", "unglued", "noncollapse", "non-collapsing", "backlift",
    "backliftability", "lift", "lifting", "cocycle", "torsor", "fragment", "fragmented",
    "screen", "boundary", "failure-to-globalize", "does-not-globalize",
}
COLLAPSE_TERMS = {
    "collapse", "classicalize", "classicalized", "resolve", "resolved", "decide",
    "determine", "determinize", "force", "reduce", "single", "winner", "erase",
}
PRESERVATION_TERMS = {
    "preserve", "retain", "retention", "do-not-collapse", "do", "not", "keep",
    "maintain", "carry", "route", "without-collapsing", "noncollapse", "non-collapse",
}


PROFILE_KEYS = (
    "t_support", "f_support", "glut_mass", "gap_mass", "obstruction_residue",
    "collapse_pressure", "classicalization_depth",
)

FAMILY_HINTS = {
    "liar": {"liar", "this sentence", "says of itself"},
    "sorites": {"heap", "grain", "borderline", "vague"},
    "curry": {"if this claim is true", "curry"},
    "russell": {"set of all", "russell"},
    "local_global_obstruction": {"local", "global", "gluing", "obstruction", "cocycle"},
    "gap_case": {"unknown", "underdetermined", "neither", "gap"},
    "evidence_conflict": {"supports", "opposes", "evidence", "ledger"},
    "de_morgan": {"de morgan", "swap", "involution", "truth", "refutation"},
    "backliftability": {"backlift", "lift", "lifting"},
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def tokenize_words(text: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(text or "")]


def _canonical_token(token: str) -> str:
    return token.lower().strip().replace("_", "-")


def _term_density(tokens: Sequence[str], terms: Iterable[str]) -> float:
    if not tokens:
        return 0.0
    termset = {_canonical_token(t) for t in terms}
    hits = sum(1 for t in tokens if _canonical_token(t) in termset)
    return clamp(hits / max(3.0, math.sqrt(len(tokens)) * 2.0))


def _phrase_hits(text: str, phrases: Iterable[str]) -> float:
    lowered = (text or "").lower()
    hits = sum(1 for p in phrases if " " in p and p in lowered)
    return clamp(hits / 3.0)


def infer_family(text: str, default: str = "synthetic_paradox") -> str:
    lowered = (text or "").lower()
    best_name = default
    best_score = 0
    for name, hints in FAMILY_HINTS.items():
        score = sum(1 for h in hints if h in lowered)
        if score > best_score:
            best_name, best_score = name, score
    return best_name


@dataclass(frozen=True)
class UAPTokenProfile:
    surface_token: str
    local_context: str
    t_support: float
    f_support: float
    glut_mass: float
    gap_mass: float
    obstruction_residue: float
    collapse_pressure: float
    classicalization_depth: float
    family: str = "synthetic_paradox"
    probe_type: str = "profile"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowDepthScores:
    contradiction_retention: float
    noncollapse_under_gluts: float
    gap_recognition: float
    truth_falsity_calibration: float
    local_global_obstruction_preservation: float
    unseen_paradox_family_transfer: float
    stabilization_depth: float
    loop_drift_behavior: float
    backliftability_diagnostics: float
    collapse_pressure: float
    support_profile_consistency: float
    residue_preservation: float

    @property
    def shadow_depth_mean(self) -> float:
        positive = [
            self.contradiction_retention,
            self.noncollapse_under_gluts,
            self.gap_recognition,
            self.truth_falsity_calibration,
            self.local_global_obstruction_preservation,
            self.unseen_paradox_family_transfer,
            self.stabilization_depth,
            self.loop_drift_behavior,
            self.backliftability_diagnostics,
            1.0 - self.collapse_pressure,
            self.support_profile_consistency,
            self.residue_preservation,
        ]
        return sum(clamp(x) for x in positive) / len(positive)

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["shadow_depth_mean"] = self.shadow_depth_mean
        return d


def profile_from_text(
    text: str,
    *,
    surface_token: str = "<span>",
    local_context: Optional[str] = None,
    family: Optional[str] = None,
    probe_type: str = "profile",
) -> UAPTokenProfile:
    """Infer a UAP token profile from text using stable lexical semantics.

    This is not presented as a final learned head.  It is a deterministic labeler
    for v14.3.2 corpus/probe construction and validation smoke tests.
    """
    text = text or ""
    context = local_context if local_context is not None else text[:280]
    tokens = tokenize_words(text)
    lowered = text.lower()

    pos = _term_density(tokens, POSITIVE_TERMS)
    neg = _term_density(tokens, NEGATIVE_TERMS)
    glut_signal = max(_term_density(tokens, GLUT_TERMS), _phrase_hits(lowered, {"both true and false", "truth and falsity", "support and opposition"}))
    gap_signal = _term_density(tokens, GAP_TERMS)
    obstruction_signal = max(_term_density(tokens, OBSTRUCTION_TERMS), _phrase_hits(lowered, {"does not globalize", "failure to globalize", "local to global"}))
    collapse_signal = _term_density(tokens, COLLAPSE_TERMS)
    preservation_signal = max(_term_density(tokens, PRESERVATION_TERMS), _phrase_hits(lowered, {"do not collapse", "without collapsing", "preserve both"}))

    # Make evidence-bearing text nonzero but keep classical prose close to the floor.
    if pos == 0.0 and any(w in lowered for w in ("claim", "sentence", "statement")):
        pos = 0.18
    if neg == 0.0 and any(w in lowered for w in ("but", "however", "yet", "unless")):
        neg = 0.12

    # Bilateral measures.  Glut is co-presence of support/refutation plus explicit contradiction markers.
    glut = clamp(0.55 * min(pos, neg) + 0.45 * glut_signal)
    # Gap is lack of determinate evidence plus explicit underdetermination markers.
    gap = clamp(0.55 * (1.0 - max(pos, neg)) + 0.45 * gap_signal)
    obstruction = clamp(0.40 * obstruction_signal + 0.25 * glut + 0.25 * gap + 0.10 * min(pos, neg))

    # Collapse pressure rises when the text tries to force a single classical outcome,
    # but falls when preservation language explicitly says not to erase the bilateral state.
    collapse_pressure = clamp(0.50 * collapse_signal + 0.35 * max(pos, neg) - 0.45 * preservation_signal - 0.20 * obstruction)
    classicalization_depth = clamp(1.0 - (0.38 * glut + 0.32 * gap + 0.30 * obstruction))

    return UAPTokenProfile(
        surface_token=surface_token,
        local_context=context,
        t_support=clamp(pos),
        f_support=clamp(neg),
        glut_mass=glut,
        gap_mass=gap,
        obstruction_residue=obstruction,
        collapse_pressure=collapse_pressure,
        classicalization_depth=classicalization_depth,
        family=family or infer_family(text),
        probe_type=probe_type,
    )


def sentence_profiles(text: str, *, family: Optional[str] = None, probe_type: str = "profile") -> List[UAPTokenProfile]:
    parts = [p.strip() for p in SENTENCE_RE.split(text or "") if p.strip()]
    if not parts and text:
        parts = [text]
    return [profile_from_text(p, surface_token=f"sentence_{i}", family=family, probe_type=probe_type) for i, p in enumerate(parts)]


def label_example(
    text: str,
    *,
    family: Optional[str] = None,
    split: str = "train",
    probe_type: str = "profile",
    example_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    profile = profile_from_text(text, family=family, probe_type=probe_type)
    record: Dict[str, Any] = {
        "id": example_id,
        "text": text,
        "split": split,
        "family": profile.family,
        "probe_type": probe_type,
        "uap_profile": profile.to_dict(),
        "uap_token_profiles": [p.to_dict() for p in sentence_profiles(text, family=profile.family, probe_type=probe_type)],
        "uap_schema_version": "tovah-uap-token-profile-v14.3.2",
    }
    if extra:
        record.update(dict(extra))
    return record


def score_generation_against_profile(
    prompt_profile: Mapping[str, Any] | UAPTokenProfile,
    generation: str,
    *,
    heldout_family: bool = False,
) -> ShadowDepthScores:
    """Score generated text against a target UAP profile.

    This intentionally asks a different question than cross-entropy: did the
    continuation preserve bilateral support, gap/glut, obstruction residue, and
    noncollapse where the prompt demanded it?
    """
    target = prompt_profile.to_dict() if isinstance(prompt_profile, UAPTokenProfile) else dict(prompt_profile)
    gen = profile_from_text(generation, family=str(target.get("family", "generation")), probe_type="generation")

    target_t = float(target.get("t_support", 0.0))
    target_f = float(target.get("f_support", 0.0))
    target_glut = float(target.get("glut_mass", 0.0))
    target_gap = float(target.get("gap_mass", 0.0))
    target_obstruction = float(target.get("obstruction_residue", 0.0))

    def closeness(a: float, b: float) -> float:
        return clamp(1.0 - abs(float(a) - float(b)))

    contradiction_retention = closeness(min(gen.t_support, gen.f_support), min(target_t, target_f)) if target_glut > 0.15 else 1.0
    noncollapse_under_gluts = clamp((gen.glut_mass + gen.obstruction_residue) / max(0.25, target_glut + target_obstruction)) if target_glut > 0.15 else clamp(1.0 - gen.collapse_pressure)
    gap_recognition = closeness(gen.gap_mass, target_gap) if target_gap > 0.20 else clamp(1.0 - gen.gap_mass * 0.5)
    truth_falsity_calibration = 0.5 * closeness(gen.t_support, target_t) + 0.5 * closeness(gen.f_support, target_f)
    obstruction_preservation = closeness(gen.obstruction_residue, target_obstruction) if target_obstruction > 0.12 else clamp(1.0 - gen.obstruction_residue * 0.4)
    unseen_transfer = clamp(0.70 * truth_falsity_calibration + 0.30 * obstruction_preservation) if heldout_family else 1.0
    stabilization_depth = clamp(1.0 - abs(gen.classicalization_depth - float(target.get("classicalization_depth", 1.0))))
    loop_drift_behavior = _loop_drift_score(generation)
    backliftability = _backliftability_score(generation, target_obstruction)
    support_consistency = clamp(1.0 - abs((gen.t_support - gen.f_support) - (target_t - target_f)))
    residue_preservation = closeness(gen.obstruction_residue, target_obstruction)

    return ShadowDepthScores(
        contradiction_retention=contradiction_retention,
        noncollapse_under_gluts=noncollapse_under_gluts,
        gap_recognition=gap_recognition,
        truth_falsity_calibration=truth_falsity_calibration,
        local_global_obstruction_preservation=obstruction_preservation,
        unseen_paradox_family_transfer=unseen_transfer,
        stabilization_depth=stabilization_depth,
        loop_drift_behavior=loop_drift_behavior,
        backliftability_diagnostics=backliftability,
        collapse_pressure=gen.collapse_pressure,
        support_profile_consistency=support_consistency,
        residue_preservation=residue_preservation,
    )


def _loop_drift_score(text: str) -> float:
    """Return a 0..1 loop-stability score, higher is better.

    v14.3.3 delegates to the explicit 2/3/4/5-gram loop diagnostic when
    available.  The fallback preserves the v14.3.2a bigram/run behavior so
    corpus tooling remains import-safe even outside the full package context.
    """
    try:
        from tovah_v14.training.loop_stability import score_loop_stability
        return clamp(score_loop_stability(text).loop_drift_behavior)
    except Exception:
        tokens = [_canonical_token(t) for t in tokenize_words(text)]
        if len(tokens) < 12:
            return 1.0
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            return 1.0
        unique_ratio = len(set(bigrams)) / len(bigrams)
        repeated_runs = 0
        last = None
        run = 0
        for tok in tokens:
            if tok == last:
                run += 1
                if run >= 3:
                    repeated_runs += 1
            else:
                last, run = tok, 1
        return clamp(0.85 * unique_ratio + 0.15 * (1.0 - min(1.0, repeated_runs / 3.0)))


def _backliftability_score(text: str, target_obstruction: float) -> float:
    lowered = (text or "").lower()
    lift_terms = ["backlift", "lift", "lifting", "local", "global", "gluing", "obstruction", "residue"]
    hit = sum(1 for t in lift_terms if t in lowered)
    raw = clamp(hit / 4.0)
    if target_obstruction > 0.20:
        return raw
    return clamp(1.0 - 0.25 * raw)


def average_scores(scores: Sequence[ShadowDepthScores]) -> Dict[str, float]:
    if not scores:
        return {"n": 0.0}
    keys = list(scores[0].to_dict().keys())
    out = {"n": float(len(scores))}
    for k in keys:
        out[k] = sum(s.to_dict()[k] for s in scores) / len(scores)
    return out



def bilateral_class_from_profile(profile: Mapping[str, Any] | UAPTokenProfile) -> str:
    """Return the ABKG class induced by a UAP token profile."""
    d = profile.to_dict() if isinstance(profile, UAPTokenProfile) else dict(profile)
    t = float(d.get("t_support", 0.0) or 0.0)
    f = float(d.get("f_support", 0.0) or 0.0)
    if t >= 0.55 and f >= 0.55:
        return "K"
    if t >= 0.55 and f < 0.55:
        return "A"
    if t < 0.55 and f >= 0.55:
        return "B"
    return "G"


def uap_profile_targets_from_record(record: Mapping[str, Any]) -> Dict[str, float]:
    """Extract scalar UAP profile targets from a corpus record.

    Falls back to bilateral_t/bilateral_f when older v14.3.1 records lack the
    v14.3.2 uap_profile envelope.
    """
    profile = record.get("uap_profile")
    if isinstance(profile, Mapping):
        src = profile
        t = float(src.get("t_support", record.get("bilateral_t", 0.5)) or 0.0)
        f = float(src.get("f_support", record.get("bilateral_f", 0.5)) or 0.0)
        return {
            "t_support": clamp(t),
            "f_support": clamp(f),
            "glut_mass": clamp(src.get("glut_mass", min(t, f)) or 0.0),
            "gap_mass": clamp(src.get("gap_mass", min(1.0 - t, 1.0 - f)) or 0.0),
            "obstruction_residue": clamp(src.get("obstruction_residue", 0.0) or 0.0),
            "collapse_pressure": clamp(src.get("collapse_pressure", 0.0) or 0.0),
            "classicalization_depth": clamp(src.get("classicalization_depth", 1.0 - min(t, f)) or 0.0),
        }
    t = clamp(float(record.get("bilateral_t", 0.5) or 0.5))
    f = clamp(float(record.get("bilateral_f", 0.5) or 0.5))
    glut = clamp(min(t, f))
    gap = clamp(min(1.0 - t, 1.0 - f))
    obstruction = clamp(0.35 * glut + 0.25 * gap)
    return {
        "t_support": t,
        "f_support": f,
        "glut_mass": glut,
        "gap_mass": gap,
        "obstruction_residue": obstruction,
        "collapse_pressure": clamp(0.20 * max(t, f) - 0.25 * max(glut, gap, obstruction)),
        "classicalization_depth": clamp(1.0 - (0.38 * glut + 0.32 * gap + 0.30 * obstruction)),
    }
