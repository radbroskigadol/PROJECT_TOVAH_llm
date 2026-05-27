"""Memory identity eval: contradiction only counts when same-referent identity is established."""
from __future__ import annotations

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.evals.common import emit, result
from tovah_v14.hott.memory_identity import PairDiagnosis, classify_pair


def run() -> dict:
    m1 = {
        "subject": "module.x", "version": "1", "test": "smoke", "environment": "cpu",
        "bilateral_assessment": BilateralValue(0.92, 0.05),
    }
    m2 = {
        "subject": "module.x", "version": "1", "test": "smoke", "environment": "cpu",
        "bilateral_assessment": BilateralValue(0.08, 0.91),
    }
    m3 = {
        "subject": "module.y", "version": "1", "test": "smoke", "environment": "cpu",
        "bilateral_assessment": BilateralValue(0.08, 0.91),
    }
    same = classify_pair(m1, m2)
    different = classify_pair(m1, m3)
    ok = same.diagnosis == PairDiagnosis.SAME_OBJECT_CONFLICT and different.diagnosis != PairDiagnosis.SAME_OBJECT_CONFLICT
    return result(
        "memory_conflict_eval",
        ok,
        same_diagnosis=same.diagnosis.value,
        different_diagnosis=different.diagnosis.value,
        same_reason=same.reason,
        different_reason=different.reason,
    )


if __name__ == "__main__":
    emit(run())
