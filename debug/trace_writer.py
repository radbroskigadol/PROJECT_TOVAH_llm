"""
TOVAH v14.2.7 debug/trace_writer.py — persist-on-evict for rolling buffers.

AUDIT FIX (v14.2.7, sec 4): the v14.2.6 rolling buffers (gate_log[-200:],
proposal_history[-500:], message_log[-200:], etc.) silently dropped the
oldest entries when full. Under high-velocity parallel execution, that
discarded root-cause evidence before observers/critics could consume it.

This module provides a single point of forensic persistence. Every
truncation site appends the soon-to-be-dropped records to an append-only
NDJSON trace file under `tovah_traces/`, so:

  - in-memory caps stay bounded (no behavior change for live consumers)
  - on-disk history is unbounded but cheap and audit-friendly
  - synthetic-fact compression (future work) has ground-truth to validate against

All write paths swallow exceptions: a failure to persist a trace MUST NOT
crash the kernel. Failures are logged at WARNING level once per minute per
trace to avoid log spam.

Public:
  TraceWriter(name) — append records to `tovah_traces/<name>.ndjson`
  evict_records(name, records) — module-level convenience
  set_trace_root(path) — override the default trace directory
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_DEFAULT_TRACE_ROOT = Path("tovah_traces")
_TRACE_ROOT_LOCK = threading.RLock()
_TRACE_ROOT: Path = _DEFAULT_TRACE_ROOT
_WARN_THROTTLE: Dict[str, float] = {}
_WARN_INTERVAL = 60.0  # one warning per trace per minute


def set_trace_root(path: str | Path) -> None:
    """Override the trace directory (used in tests and SAFE_MODE runs)."""
    global _TRACE_ROOT
    with _TRACE_ROOT_LOCK:
        _TRACE_ROOT = Path(path)


def get_trace_root() -> Path:
    with _TRACE_ROOT_LOCK:
        return Path(_TRACE_ROOT)


def _maybe_warn(trace_name: str, exc: BaseException) -> None:
    now = time.time()
    last = _WARN_THROTTLE.get(trace_name, 0.0)
    if now - last >= _WARN_INTERVAL:
        _WARN_THROTTLE[trace_name] = now
        logging.warning(
            "TraceWriter[%s]: persist-on-evict failed (%s: %s). "
            "In-memory cap still applied. This warning is throttled.",
            trace_name, type(exc).__name__, exc,
        )


def _coerce(rec: Any) -> Dict[str, Any]:
    """Best-effort coercion of arbitrary records into JSON-serializable dicts."""
    if isinstance(rec, dict):
        return rec
    if is_dataclass(rec):
        try:
            return asdict(rec)
        except Exception:
            pass
    if hasattr(rec, "to_dict"):
        try:
            return dict(rec.to_dict())
        except Exception:
            pass
    if hasattr(rec, "__dict__"):
        try:
            return {k: v for k, v in vars(rec).items() if not k.startswith("_")}
        except Exception:
            pass
    return {"value": repr(rec)}


class TraceWriter:
    """Thread-safe append-only NDJSON writer for one named trace.

    Lazily creates `tovah_traces/<name>.ndjson` on first write. Each
    record is serialized with json.dumps(default=str), so non-JSON
    values (datetime, Path, sets) are stringified rather than failing.
    """

    _instances: Dict[str, "TraceWriter"] = {}
    _instances_lock = threading.RLock()

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self._lock = threading.RLock()

    @classmethod
    def get(cls, name: str) -> "TraceWriter":
        """Return a process-wide singleton for `name`."""
        with cls._instances_lock:
            inst = cls._instances.get(name)
            if inst is None:
                inst = cls(name)
                cls._instances[name] = inst
            return inst

    def _path(self) -> Path:
        root = get_trace_root()
        return root / f"{self.name}.ndjson"

    def append(self, record: Any) -> bool:
        """Append one record. Returns True on success, False on caught failure."""
        try:
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _coerce(record)
            payload.setdefault("_trace_at", time.time())
            line = json.dumps(payload, default=str, ensure_ascii=False)
            with self._lock:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            return True
        except Exception as e:
            _maybe_warn(self.name, e)
            return False

    def extend(self, records: Iterable[Any]) -> int:
        """Append many records. Returns the count successfully persisted."""
        ok = 0
        try:
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            lines: List[str] = []
            for rec in records:
                try:
                    payload = _coerce(rec)
                    payload.setdefault("_trace_at", time.time())
                    lines.append(json.dumps(payload, default=str, ensure_ascii=False))
                    ok += 1
                except Exception:
                    continue
            if not lines:
                return 0
            with self._lock:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            return ok
        except Exception as e:
            _maybe_warn(self.name, e)
            return 0


def evict_records(name: str, records: Iterable[Any]) -> int:
    """Module-level convenience for the common `[-N:]` truncation idiom.

    Usage:
        # before:
        self.log.append(rec)
        self.log = self.log[-200:]
        # after:
        self.log.append(rec)
        if len(self.log) > 200:
            evict_records("my_log", self.log[:-200])
            self.log = self.log[-200:]
    """
    return TraceWriter.get(name).extend(records)


__all__ = ["TraceWriter", "evict_records", "set_trace_root", "get_trace_root"]
