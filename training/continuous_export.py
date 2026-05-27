"""
TOVAH v14 training/continuous_export.py — Streaming corpus emission.

A kernel hook that incrementally appends training examples as TOVAH runs,
so pretraining can happen *while* the system is live.

Wire-up: instantiate ContinuousExporter once at boot and call
`exporter.append_from_event(event)` next to each `_save_kernel_ecology_state`
call site. The exporter will append to its current shard, rotate when
shard_size is reached, and update the manifest in-place.

Lightweight by design: no dedup or paraconsistent classification at append
time — those are batch operations run on the on-disk shards (or in
memory) before final corpus release.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tovah_v14.training.corpus_builder import (
    TrainingExample, _experience_to_example, _packet_to_example,
    _module_proposal_to_example, _gate_decision_to_example,
    _wave_outcome_to_example,
)


class ContinuousExporter:
    """Streaming JSONL append-only writer.

    Thread-safe at the file-handle level; multiple kernels writing to
    distinct directories are fully independent. Within a single directory
    the lock serialises shard writes.
    """

    def __init__(self, out_dir: str | Path,
                 *, shard_size: int = 1000, prefix: str = "tovah_stream"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self._lock = threading.Lock()
        self._current_shard_idx = self._discover_next_shard()
        self._current_shard_count = 0
        self._handle = None
        self._open_current()

    def _discover_next_shard(self) -> int:
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.jsonl"))
        if not existing:
            return 0
        last = existing[-1].stem
        try:
            return int(last.split("_")[-1]) + 1
        except Exception:
            return len(existing)

    def _shard_path(self, idx: int) -> Path:
        return self.out_dir / f"{self.prefix}_{idx:05d}.jsonl"

    def _open_current(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
        self._handle = open(self._shard_path(self._current_shard_idx), "a",
                            encoding="utf-8")

    def _maybe_rotate(self) -> None:
        if self._current_shard_count >= self.shard_size:
            self._current_shard_idx += 1
            self._current_shard_count = 0
            self._open_current()

    def append(self, example: TrainingExample) -> None:
        """Append one TrainingExample to the current shard."""
        with self._lock:
            assert self._handle is not None
            self._handle.write(json.dumps(example.to_dict(), default=str, ensure_ascii=False))
            self._handle.write("\n")
            self._handle.flush()
            self._current_shard_count += 1
            self._maybe_rotate()

    def append_from_event(self, event: Dict[str, Any]) -> Optional[TrainingExample]:
        """Convert a kernel event into a TrainingExample and append.

        Recognises kernel packet log entries (including the events
        produced by `_dispatch_kernel_packet`). Returns the example
        appended, or None if the event was filtered (e.g. heartbeats).
        """
        # If event looks like a packet (has packet_kind), use the packet path.
        if "packet_kind" in event:
            ex = _packet_to_example(event)
            if ex is not None:
                self.append(ex)
            return ex
        return None

    def append_experience(self, rec: Dict[str, Any]) -> TrainingExample:
        ex = _experience_to_example(rec)
        self.append(ex)
        return ex

    def append_module_proposal(self, mp: Dict[str, Any]) -> TrainingExample:
        ex = _module_proposal_to_example(mp)
        self.append(ex)
        return ex

    def append_gate_decision(self, dec: Dict[str, Any], patch_name: str = "") -> TrainingExample:
        ex = _gate_decision_to_example(dec, patch_name=patch_name)
        self.append(ex)
        return ex

    def append_wave_outcome(self, rec: Dict[str, Any], outcome_kind: str = "resolution") -> TrainingExample:
        ex = _wave_outcome_to_example(rec, outcome_kind)
        self.append(ex)
        return ex

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                finally:
                    self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
