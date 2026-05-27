"""Run all lightweight buyer-facing evals and print a JSON report."""
from __future__ import annotations

import json

from tovah_v14.evals import (
    gap_tolerance,
    high_glut_preservation,
    memory_conflict_eval,
    patch_certification_eval,
    semantic_consistency,
    smoke_language_modeling,
)


def run() -> dict:
    results = [
        smoke_language_modeling.run(),
        semantic_consistency.run(),
        high_glut_preservation.run(),
        gap_tolerance.run(),
        patch_certification_eval.run(),
        memory_conflict_eval.run(),
    ]
    return {"passed": all(r.get("passed") for r in results), "results": results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
