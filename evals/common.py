"""Common helpers for lightweight TOVAH buyer evals."""
from __future__ import annotations

import json
from typing import Any, Dict


def result(name: str, passed: bool, **metrics: Any) -> Dict[str, Any]:
    return {"eval": name, "passed": bool(passed), "metrics": metrics}


def emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
