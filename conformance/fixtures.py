"""
TOVAH v14 conformance/fixtures.py — Baseline fixtures for conformance testing.

Each fixture defines:
- initial beta state
- expected gamma cache
- expected invariant report properties

These are regression seeds. If any fixture fails, the runtime contract is broken.
"""
from __future__ import annotations

from typing import Any, Dict, List

from tovah_v14.core.primitives import BilateralValue


# Fixture format: {name, beta, expected_nu, expected_properties}
BASELINE_FIXTURES: List[Dict[str, Any]] = [
    {
        "name": "all_true",
        "beta": {"a": BilateralValue(0.9, 0.1), "b": BilateralValue(0.8, 0.2)},
        "expected_nu": {"a": "T", "b": "T"},
        "expected": {"coherent": True, "mean_glut_below": 0.25},
    },
    {
        "name": "all_false",
        "beta": {"a": BilateralValue(0.1, 0.9), "b": BilateralValue(0.2, 0.8)},
        "expected_nu": {"a": "F", "b": "F"},
        "expected": {"coherent": True, "mean_glut_below": 0.25},
    },
    {
        "name": "all_glut",
        "beta": {"a": BilateralValue(0.8, 0.8), "b": BilateralValue(0.9, 0.7)},
        "expected_nu": {"a": "B", "b": "B"},
        "expected": {"coherent": True, "mean_glut_above": 0.5},
    },
    {
        "name": "all_gap",
        "beta": {"a": BilateralValue(0.1, 0.1), "b": BilateralValue(0.2, 0.2)},
        "expected_nu": {"a": "G", "b": "G"},
        "expected": {"coherent": True, "mean_gap_above": 0.5},
    },
    {
        "name": "mixed",
        "beta": {
            "true_key": BilateralValue(0.9, 0.1),
            "false_key": BilateralValue(0.1, 0.9),
            "glut_key": BilateralValue(0.8, 0.8),
            "gap_key": BilateralValue(0.2, 0.2),
        },
        "expected_nu": {"true_key": "T", "false_key": "F", "glut_key": "B", "gap_key": "G"},
        "expected": {"coherent": True},
    },
    {
        "name": "empty_beta",
        "beta": {},
        "expected_nu": {},
        "expected": {"coherent": True, "beta_key_count": 0},
    },
    {
        "name": "boundary_theta",
        "beta": {"exact": BilateralValue(0.55, 0.55)},
        "expected_nu": {"exact": "B"},
        "expected": {"coherent": True},
    },
    {
        "name": "just_below_theta",
        "beta": {"below": BilateralValue(0.54, 0.54)},
        "expected_nu": {"below": "G"},
        "expected": {"coherent": True},
    },
]
