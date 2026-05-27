"""
TOVAH v14 debug/metrics.py — Metrics collection and trend analysis.

Maintains a rolling window of CycleMetrics for trend detection,
anomaly identification, and performance tracking.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from tovah_v14.config.paths import METRICS_DIR
from tovah_v14.config.constants import MAX_METRICS_FILES
from tovah_v14.debug.observability import CycleMetrics
from tovah_v14.persistence.state_io import save_json


class MetricsCollector:
    """Collects and persists cycle metrics."""

    def __init__(self, max_memory: int = 500) -> None:
        self.history: List[CycleMetrics] = []
        self.max_memory = max_memory

    def record(self, metrics: CycleMetrics, persist: bool = False) -> None:
        """Record a cycle metrics snapshot."""
        self.history.append(metrics)
        self.history = self.history[-self.max_memory:]

        if persist:
            fname = f"metrics_{int(metrics.timestamp)}.json"
            from dataclasses import asdict
            save_json(METRICS_DIR / fname, asdict(metrics))
            self._cleanup_disk()

    def _cleanup_disk(self) -> None:
        """Remove oldest metrics files if over limit."""
        files = sorted(METRICS_DIR.glob("metrics_*.json"), key=lambda p: p.stat().st_mtime)
        while len(files) > MAX_METRICS_FILES:
            try:
                files[0].unlink(missing_ok=True)
            except Exception:
                pass
            files.pop(0)

    def trend(self, field: str, window: int = 20) -> Dict[str, float]:
        """Compute trend for a numeric field over recent history."""
        recent = self.history[-window:]
        if not recent:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        vals = [getattr(m, field, 0.0) for m in recent]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }

    def anomalies(self, field: str, threshold: float = 2.0) -> List[int]:
        """Detect anomalous cycles where field deviates > threshold * stdev from mean."""
        recent = self.history[-100:]
        vals = [getattr(m, field, 0.0) for m in recent]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) < 5:
            return []
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        if std < 1e-8:
            return []
        anomaly_cycles = []
        for i, (m, v) in enumerate(zip(recent, vals)):
            if abs(v - mean) > threshold * std:
                anomaly_cycles.append(m.cycle)
        return anomaly_cycles
