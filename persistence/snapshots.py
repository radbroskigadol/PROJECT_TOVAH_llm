"""
TOVAH v14 persistence/snapshots.py — Model weight snapshots.

SEMANTIC PRESERVATION:
  Snapshot format and rollback logic identical to v13.
  Snapshots are .pt files containing model state_dict and metadata.

The kernel calls these functions; they do not import kernel.
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from tovah_v14.config.paths import SNAPSHOT_DIR, BRANCH_CHECKPOINT_DIR
from tovah_v14.config.constants import MAX_SNAPSHOTS_DISK, MAX_SNAPSHOTS_MEMORY


def _slugify(text: str) -> str:
    import re
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower()).strip("_")
    return slug[:80] or f"item_{int(time.time())}"


def save_snapshot(
    model: torch.nn.Module,
    reason: str,
    meta_extra: Dict[str, Any] | None = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
    max_disk: int = MAX_SNAPSHOTS_DISK,
) -> Dict[str, Any]:
    """Save model weights + metadata to disk.

    Returns metadata dict with path and model_state for in-memory cache.
    """
    meta: Dict[str, Any] = {
        "reason": reason,
        "timestamp": time.time(),
    }
    if meta_extra:
        meta.update(meta_extra)

    fn = f"snap_{int(time.time())}_{_slugify(reason)[:30]}.pt"
    path = snapshot_dir / fn

    try:
        torch.save({"model": model.state_dict(), "meta": meta}, path)
        meta["path"] = str(path)
        meta["model_state"] = copy.deepcopy(model.state_dict())
        logging.info(f"SNAPSHOT: {reason} -> {fn}")
    except Exception as e:
        logging.error(f"SNAPSHOT SAVE FAILED: {e}")
        meta["path"] = ""
        meta["error"] = str(e)

    # Disk cleanup
    cleanup_snapshots(snapshot_dir, max_disk)

    return meta


def rollback_snapshot(
    model: torch.nn.Module,
    memory_snapshots: List[Dict[str, Any]],
    device: str = "cpu",
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Rollback model weights to last snapshot.

    Tries in-memory snapshots first, then disk.
    Returns (ok, message, metadata).
    """
    # Try memory first
    if memory_snapshots:
        snap = memory_snapshots[-1]
        model_state = snap.get("model_state")
        if model_state is not None:
            try:
                model.load_state_dict(model_state)
                return True, f"rolled back from memory: {snap.get('reason', '?')}", snap
            except Exception as e:
                logging.warning(f"memory rollback failed: {e}")

    # Fall back to disk
    snaps = sorted(snapshot_dir.glob("snap_*.pt"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        return False, "no snapshots available", {}

    try:
        ckpt = torch.load(snaps[-1], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        meta = ckpt.get("meta", {})
        return True, f"rolled back from disk: {snaps[-1].name}", meta
    except Exception as e:
        return False, f"disk rollback failed: {e}", {}


def cleanup_snapshots(
    snapshot_dir: Path = SNAPSHOT_DIR,
    max_disk: int = MAX_SNAPSHOTS_DISK,
) -> int:
    """Remove oldest snapshots if over limit. Returns count removed."""
    snaps = sorted(snapshot_dir.glob("snap_*.pt"), key=lambda p: p.stat().st_mtime)
    removed = 0
    while len(snaps) > max_disk:
        try:
            snaps[0].unlink(missing_ok=True)
        except Exception:
            pass
        snaps.pop(0)
        removed += 1
    return removed


def load_shadow_weights(
    model: torch.nn.Module,
    shadow_file: Path,
    device: str = "cpu",
) -> bool:
    """Load shadow model weights from file. Returns True on success."""
    if not shadow_file.exists():
        return False
    try:
        data = torch.load(shadow_file, map_location=device, weights_only=False)
        model.load_state_dict(data["model"])
        return True
    except Exception as e:
        logging.warning(f"shadow weight restore failed: {e}")
        return False


def save_shadow_weights(
    model: torch.nn.Module,
    shadow_file: Path,
    improvement_count: int = 0,
    max_retries: int = 3,
) -> bool:
    """Save shadow model weights atomically. Temp file + os.replace.
    Includes retry/backoff for Windows file-lock contention.
    """
    tmp = shadow_file.with_suffix(shadow_file.suffix + ".tmp")
    for attempt in range(max_retries):
        try:
            torch.save({"model": model.state_dict(), "improvement_count": improvement_count}, tmp)
            import os as _os
            _os.replace(str(tmp), str(shadow_file))
            return True
        except OSError as e:
            # Windows error 32: file in use. Backoff and retry.
            logging.warning(f"shadow weight save attempt {attempt+1} failed: {e}")
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            logging.error(f"shadow weight save failed: {e}")
            break
    # Cleanup tmp if it exists
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return False


from tovah_v14.persistence.state_io import save_json, load_json


def save_branch_checkpoint(
    branch_name: str,
    checkpoint: Dict[str, Any],
    checkpoint_dir: Path = BRANCH_CHECKPOINT_DIR,
) -> Dict[str, Any]:
    """Save a branch/ecology checkpoint to JSON. Returns metadata dict."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = _slugify(branch_name or "branch")
    path = checkpoint_dir / f"branch_{int(time.time())}_{stem[:40]}.json"
    meta = {
        "branch_name": branch_name,
        "path": str(path),
        "timestamp": time.time(),
        "saved": False,
    }
    payload = {"meta": dict(meta), "checkpoint": dict(checkpoint)}
    meta["saved"] = save_json(path, payload)
    return meta


def load_branch_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a branch checkpoint payload from disk."""
    data = load_json(path, {})
    return data if isinstance(data, dict) else {}


def list_branch_checkpoints(checkpoint_dir: Path = BRANCH_CHECKPOINT_DIR) -> List[Dict[str, Any]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("branch_*.json"), key=lambda p: p.stat().st_mtime):
        items.append({
            "path": str(path),
            "name": path.name,
            "mtime": path.stat().st_mtime,
        })
    return items[-100:]
