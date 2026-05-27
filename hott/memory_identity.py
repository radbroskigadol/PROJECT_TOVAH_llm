"""
TOVAH v14.2.0 hott/memory_identity.py — Identity witnesses for memory entries.

Priority #3 from the architecture brief:

    Right now, contradiction is mostly measured as bilateral t/f tension.
    Full HoTT would let you track the identity relation between claims.

    Example: "The module failed." vs "The module succeeded."
    This is a contradiction only if they refer to the same module,
    same version, same test, same environment, same time-window, and
    same success criterion.

This module implements identity-path construction over memory entries.
Two memories are 'about the same thing' iff we can build a Path between
them in MemoryReferentType with sufficient T-evidence.

Public:
  MemoryReferentType   — Type of (subject, version, test, environment, ...)
  build_referent       — extract a referent from a memory entry
  identity_path        — heuristic path with bilateral evidence between
                         two referents
  is_genuine_conflict  — True iff (a) the bilateral assessments contradict
                         AND (b) the referents are A-class identified
  classify_pair        — full diagnosis: same_object / different_object /
                         genuine_conflict / spurious_conflict / unknown
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott.core import Type, Id, Path, refl
from tovah_v14.hott.paraconsistent import (
    IdentityClass, PIdJudgment, classify_path,
)


# --- Referent Type ----------------------------------------------------------

@dataclass(frozen=True)
class MemoryReferent:
    """The 'thing' a memory entry is about.

    Five dimensions, ordered by selectivity (most discriminating first):
      subject:     what is the memory about? (module name, file, etc.)
      version:     which version of the subject?
      test:        which test / probe produced the observation?
      environment: in what environment? (cuda / cpu / sandbox / live)
      time_band:   coarse time bucket (so 'last hour' style)

    All five are strings. None or "" means 'unspecified' and contributes
    no evidence either way (G-class on its own).
    """
    subject: str = ""
    version: str = ""
    test: str = ""
    environment: str = ""
    time_band: str = ""

    def __bool__(self) -> bool:
        return any((self.subject, self.version, self.test,
                    self.environment, self.time_band))


def _referent_inhabits(r: Any) -> bool:
    return isinstance(r, MemoryReferent)


MemoryReferentType = Type("MemoryReferent", inhabits=_referent_inhabits)


# --- Building referents from memory dicts ----------------------------------

def build_referent(memory: Dict[str, Any]) -> MemoryReferent:
    """Heuristically extract a MemoryReferent from a memory entry.

    The memory format varies — banks store dicts with arbitrary keys —
    so we coerce. Callers can pre-normalise by filling in these fields
    on the memory itself; this function is the best-effort extractor.
    """
    ctx = memory.get("context") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    # Subject: prefer explicit fields, then context, then text head.
    subject = str(
        memory.get("subject")
        or memory.get("target")
        or memory.get("module")
        or ctx.get("module")
        or ctx.get("subject")
        or ctx.get("tool")
        or ""
    )
    version = str(memory.get("version") or ctx.get("version") or "")
    test = str(memory.get("test") or memory.get("probe") or ctx.get("test") or "")
    environment = str(
        memory.get("environment") or memory.get("env")
        or ctx.get("environment") or ""
    )
    # Time-band: 10-minute buckets by default.
    t = float(memory.get("created_at", memory.get("time", 0.0)) or 0.0)
    if t > 0:
        time_band = str(int(t // 600))  # 10-min buckets
    else:
        time_band = ""
    return MemoryReferent(
        subject=subject, version=version, test=test,
        environment=environment, time_band=time_band,
    )


# --- Identity path construction --------------------------------------------

# Two-tier weighting scheme:
#   PRIMARY dims: subject, version, test. Any mismatch on a primary
#                 dim pushes F to at least 0.6 (refuting A-class
#                 identification); all match → T accumulates.
#   SECONDARY dims: environment, time_band. They contribute to T
#                 when matching but do not refute on mismatch.
_PRIMARY_DIMS = ("subject", "version", "test")
_SECONDARY_DIMS = ("environment", "time_band")
_PRIMARY_T_WEIGHT = {"subject": 0.40, "version": 0.30, "test": 0.20}
_SECONDARY_T_WEIGHT = {"environment": 0.05, "time_band": 0.05}
# Per-dim F-penalty on disagreement (only primary dims contribute).
_PRIMARY_F_PENALTY = {"subject": 0.65, "version": 0.65, "test": 0.65}


def _dim_evidence(d1: str, d2: str) -> Tuple[float, float]:
    """For a single dimension, evidence for and against identification.
    Returns (support_local, refute_local) each in [0, 1] BEFORE weighting."""
    if not d1 or not d2:
        return 0.0, 0.0  # gap; neither supports nor refutes
    if d1 == d2:
        return 1.0, 0.0
    # Loose match: case-insensitive substring? prefix? — for now, mismatch
    # is mismatch.
    return 0.0, 1.0


def identity_path(r1: MemoryReferent, r2: MemoryReferent) -> Path:
    """Build a Path in MemoryReferentType from r1 to r2 with heuristic
    bilateral evidence.

    SEMANTICS (v14.2.0):
      Primary dimensions (subject, version, test) are *gating*: any
      mismatch on a primary dim raises F above the A-class threshold,
      preventing 'same object' identification. All-primary-match plus
      any secondary contribution yields A-class.

      Matrix outcomes:
        all primary match, secondary match     → A   (clean same-object)
        all primary match, secondary mismatch  → A   (same object, ignore env/time)
        any primary mismatch                   → B   (genuinely different)
        partial primary match (esp version)    → K   (paradox; ambiguous)
                                                     when primary disagreement
                                                     is on a non-strong dim
    """
    if r1 == r2:
        return refl(MemoryReferentType, r1)

    t_acc = 0.0
    f_acc = 0.0
    details: Dict[str, Tuple[float, float]] = {}

    # Primary dims.
    for dim in _PRIMARY_DIMS:
        s, r = _dim_evidence(getattr(r1, dim), getattr(r2, dim))
        w_t = _PRIMARY_T_WEIGHT[dim]
        f_pen = _PRIMARY_F_PENALTY[dim]
        if s > 0:
            t_acc += w_t * s
        if r > 0:
            # Bump F to the per-dim penalty (cap at 1).
            f_acc = max(f_acc, f_pen)
        details[dim] = (w_t * s, f_pen * r if r > 0 else 0.0)

    # Secondary dims (T-only).
    for dim in _SECONDARY_DIMS:
        s, _ = _dim_evidence(getattr(r1, dim), getattr(r2, dim))
        w_t = _SECONDARY_T_WEIGHT[dim]
        if s > 0:
            t_acc += w_t * s
        details[dim] = (w_t * s, 0.0)

    t_acc = max(0.0, min(1.0, t_acc))
    f_acc = max(0.0, min(1.0, f_acc))
    return Path(
        id_type=Id(MemoryReferentType, r1, r2),
        source=r1, target=r2,
        witness={"kind": "memory_referent_match", "details": details},
        bilateral=BilateralValue(t_acc, f_acc),
    )


# --- Pair classification ---------------------------------------------------

class PairDiagnosis(str, Enum):
    SAME_OBJECT_AGREE = "same_object_agree"
    SAME_OBJECT_CONFLICT = "same_object_conflict"
    DIFFERENT_OBJECT = "different_object"
    AMBIGUOUS_IDENTIFICATION = "ambiguous_identification"  # K-class on referent
    INSUFFICIENT_INFO = "insufficient_info"


@dataclass
class PairDiagnosisReport:
    diagnosis: PairDiagnosis
    referent_path: Path
    referent_class: IdentityClass
    bilateral_conflict_t: float
    bilateral_conflict_f: float
    reason: str


def is_genuine_conflict(m1: Dict[str, Any], m2: Dict[str, Any],
                        theta_t: float = 0.55, theta_f: float = 0.55
                        ) -> bool:
    """True iff (a) the bilateral assessments contradict AND
    (b) the memory referents are A-class identified.

    This is the operational answer to the brief's example:
      "module failed" vs "module succeeded" is only a real contradiction
      if both refer to the same module, version, test, env, time.

    Implementation: any bilateral 'conflict' (one says T-high, the other
    F-high) only counts when classify_path(referent_path) == A.
    """
    report = classify_pair(m1, m2, theta_t=theta_t, theta_f=theta_f)
    return report.diagnosis == PairDiagnosis.SAME_OBJECT_CONFLICT


def classify_pair(m1: Dict[str, Any], m2: Dict[str, Any],
                  theta_t: float = 0.55, theta_f: float = 0.55
                  ) -> PairDiagnosisReport:
    """Full diagnosis of a memory pair.

    Returns a PairDiagnosisReport with:
      diagnosis: one of the enum cases
      referent_path: the heuristic identity-path between the two referents
      referent_class: ABKG class of the referent_path
      bilateral_conflict_t / _f: combined bilateral evidence for/against
      reason: short prose
    """
    r1 = build_referent(m1)
    r2 = build_referent(m2)
    path = identity_path(r1, r2)
    ref_class = classify_path(path, theta_t=theta_t, theta_f=theta_f)

    # Extract bilateral assessments from the memories themselves.
    def _bv(m: Dict[str, Any]) -> Tuple[float, float]:
        bv = (m.get("bilateral_confidence") or m.get("bilateral_assessment")
              or m.get("confidence") or {})
        if hasattr(bv, "t") and hasattr(bv, "f"):
            return (float(getattr(bv, "t")), float(getattr(bv, "f")))
        if isinstance(bv, (tuple, list)) and len(bv) >= 2:
            return (float(bv[0]), float(bv[1]))
        if not isinstance(bv, dict):
            return 0.5, 0.5
        return (float(bv.get("t", bv.get("truth", 0.5)) or 0.5),
                float(bv.get("f", bv.get("falsity", 0.5)) or 0.5))

    t1, f1 = _bv(m1)
    t2, f2 = _bv(m2)

    # Conflict-evidence: m1 says T-high while m2 says F-high (or vice versa).
    # "Strength of conflict" = max(min(t1,f2), min(t2,f1)).
    conflict_strength = max(min(t1, f2), min(t2, f1))
    # Agreement-evidence: both T-high together, or both F-high together.
    # "Strength of agreement" = max(min(t1,t2), min(f1,f2)).
    agreement_strength = max(min(t1, t2), min(f1, f2))

    if ref_class == IdentityClass.B:
        return PairDiagnosisReport(
            diagnosis=PairDiagnosis.DIFFERENT_OBJECT,
            referent_path=path,
            referent_class=ref_class,
            bilateral_conflict_t=0.0,
            bilateral_conflict_f=0.0,
            reason="referents refute identification; any apparent conflict is spurious",
        )
    if ref_class == IdentityClass.K:
        return PairDiagnosisReport(
            diagnosis=PairDiagnosis.AMBIGUOUS_IDENTIFICATION,
            referent_path=path,
            referent_class=ref_class,
            bilateral_conflict_t=conflict_strength,
            bilateral_conflict_f=agreement_strength,
            reason="referents partially match (paradox); cannot decide if same",
        )
    if ref_class == IdentityClass.G:
        return PairDiagnosisReport(
            diagnosis=PairDiagnosis.INSUFFICIENT_INFO,
            referent_path=path,
            referent_class=ref_class,
            bilateral_conflict_t=conflict_strength,
            bilateral_conflict_f=agreement_strength,
            reason="no evidence for or against identification",
        )
    # A-class: same object.
    # Genuine same-object conflict iff conflict_strength dominates and
    # exceeds the threshold.
    if conflict_strength >= theta_t and conflict_strength > agreement_strength:
        return PairDiagnosisReport(
            diagnosis=PairDiagnosis.SAME_OBJECT_CONFLICT,
            referent_path=path,
            referent_class=ref_class,
            bilateral_conflict_t=conflict_strength,
            bilateral_conflict_f=agreement_strength,
            reason="confirmed same object with conflicting bilateral assessments",
        )
    return PairDiagnosisReport(
        diagnosis=PairDiagnosis.SAME_OBJECT_AGREE,
        referent_path=path,
        referent_class=ref_class,
        bilateral_conflict_t=conflict_strength,
        bilateral_conflict_f=agreement_strength,
        reason="confirmed same object; assessments compatible",
    )


# --- Aggregation helper ----------------------------------------------------

def find_genuine_conflicts(memories: Iterable[Dict[str, Any]]
                           ) -> List[Tuple[int, int, PairDiagnosisReport]]:
    """Walk a list of memories and return all (i, j, report) tuples
    where classify_pair flags SAME_OBJECT_CONFLICT.

    Useful for the kernel's contradiction-governance subsystem to filter
    out spurious gluts caused by sloppy identity matching.
    """
    mems = list(memories)
    out: List[Tuple[int, int, PairDiagnosisReport]] = []
    for i in range(len(mems)):
        for j in range(i + 1, len(mems)):
            try:
                rep = classify_pair(mems[i], mems[j])
            except Exception:
                continue
            if rep.diagnosis == PairDiagnosis.SAME_OBJECT_CONFLICT:
                out.append((i, j, rep))
    return out
