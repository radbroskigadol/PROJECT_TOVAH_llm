"""TOVAH v14.2.6 training/metrics.py — scale-run metric logging helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


CANONICAL_SCALE_METRICS = [
    "step", "epoch", "loss", "lm_loss", "semantic_loss", "lane_routing_loss",
    "lane_b_match_loss", "lane_c_match_loss", "mean_K_pred", "mean_G_pred",
    "mean_K_meta", "mean_G_meta", "lane_entropy", "phase", "grad_norm", "lr",
    "tokens_per_second", "samples_per_second", "gpu_memory_allocated_gb",
    "gpu_memory_reserved_gb", "checkpoint_seconds", "data_loading_seconds",
]


class ScaleMetricLogger:
    """Append-only JSONL logger for buyer scale runs.

    The logger accepts partial records. Missing canonical fields are omitted
    rather than filled with fake values.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()

    def log(self, **record: Any) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "time": round(time.time(), 6),
            "elapsed_seconds": round(time.time() - self.started_at, 6),
        }
        rec.update({k: _jsonable(v) for k, v in record.items() if v is not None})
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec


def gpu_memory_gb() -> Dict[str, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        }
    except Exception:
        return {}


def _jsonable(v: Any) -> Any:
    try:
        import torch
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.detach().cpu().item())
            return v.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)
