"""
TOVAH v14.2.9 neural/checkpointing.py — resumable frontier checkpoints.

This module provides a small production-facing checkpoint surface that works in
single-process runs and degrades gracefully under FSDP. It intentionally avoids
pretending to be a full cluster checkpointing system; the goal is reliable
resume metadata, optimizer state, RNG state, and an FSDP-compatible sharded
state-dict path when torch exposes it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from tovah_v14.neural import distributed as dist_utils


def unwrap_model(model: Any) -> Any:
    """Return the underlying module for DDP-style wrappers when possible."""
    return getattr(model, "module", model)


def _optimizer_state_dict(optimizer: Any) -> Optional[Dict[str, Any]]:
    if optimizer is None:
        return None
    if hasattr(optimizer, "state_dict"):
        return optimizer.state_dict()
    inner = getattr(optimizer, "_opt", None)
    if inner is not None and hasattr(inner, "state_dict"):
        return {"inner_optimizer": inner.state_dict(), "t": getattr(optimizer, "t", 0)}
    return None


def _load_optimizer_state_dict(optimizer: Any, state: Optional[Dict[str, Any]]) -> None:
    if optimizer is None or not state:
        return
    if hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(state)
        return
    inner = getattr(optimizer, "_opt", None)
    if inner is not None and "inner_optimizer" in state:
        inner.load_state_dict(state["inner_optimizer"])
        if hasattr(optimizer, "t"):
            optimizer.t = int(state.get("t", optimizer.t))


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as e:
            logging.warning("could not restore CUDA RNG state: %s", e)


def _fsdp_full_or_sharded_state_dict(model: Any, sharded: bool) -> Dict[str, Any]:
    """Return an FSDP-aware state dict when possible, else ordinary state_dict."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType
        from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig
    except Exception:
        return unwrap_model(model).state_dict()

    if isinstance(model, FSDP):
        if sharded:
            cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
                return model.state_dict()
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return unwrap_model(model).state_dict()



def checkpoint_manifest_path(path: str | Path) -> Path:
    """Return the manifest location for a checkpoint path or directory."""
    path = Path(path)
    if path.suffix:
        return path.with_suffix(path.suffix + ".manifest.json")
    return path / "manifest.json"


def _write_checkpoint_manifest(path: str | Path, payload: Dict[str, Any], written: Path) -> None:
    """Best-effort manifest for buyer/resume handoff."""
    if not dist_utils.is_main():
        return
    manifest = {
        "format": payload.get("format"),
        "version": payload.get("version"),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "metadata": payload.get("metadata", {}),
        "distributed": payload.get("distributed", {}),
        "written": str(written),
    }
    mp = checkpoint_manifest_path(path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

def save_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Optional[Any] = None,
    *,
    step: int = 0,
    epoch: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    sharded: bool = False,
) -> Optional[Path]:
    """Save a resumable training checkpoint.

    In distributed full-state mode only rank 0 writes. In sharded mode every
    rank may write its own state to ``path/rank_XXXXX.pt``.
    """
    path = Path(path)
    payload = {
        "format": "tovah_training_checkpoint_v1",
        "version": "14.2.9",
        "step": int(step),
        "epoch": int(epoch),
        "metadata": dict(metadata or {}),
        "model_state": _fsdp_full_or_sharded_state_dict(model, sharded=sharded),
        "optimizer_state": _optimizer_state_dict(optimizer),
        "rng_state": _rng_state(),
        "distributed": {
            "world_size": dist_utils.world_size(),
            "rank": dist_utils.rank(),
            "sharded": bool(sharded),
        },
    }
    if sharded and dist_utils.world_size() > 1:
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"rank_{dist_utils.rank():05d}.pt"
    else:
        if dist_utils.world_size() > 1 and not dist_utils.is_main():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        out = path
    torch.save(payload, out)
    _write_checkpoint_manifest(path, payload, out)
    return out


def load_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Optional[Any] = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint created by save_training_checkpoint()."""
    path = Path(path)
    if path.is_dir():
        path = path / f"rank_{dist_utils.rank():05d}.pt"
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model_state", payload)
    # For FSDP, model.load_state_dict handles sharded state when inside the
    # matching FSDP state-dict context in more advanced launchers. For the common
    # full/single-process path, this direct call is correct.
    model.load_state_dict(state, strict=strict)
    _load_optimizer_state_dict(optimizer, payload.get("optimizer_state"))
    if restore_rng:
        _load_rng_state(payload.get("rng_state"))
    return payload
