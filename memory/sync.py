"""
TOVAH v14 memory/sync.py — Governed branch-memory synchronization.

Supports three explicit actions for experimental branches:
- discard
- summarize
- promote

This keeps branch memory branching explicit rather than silently mixing branch
and sovereign memory.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tovah_v14.kernel.action_model import MemorySyncRequest
from tovah_v14.memory.provenance_graph import ProvenanceGraph
from tovah_v14.memory.store import MemoryStore


@dataclass
class MemorySyncDecision:
    request_id: str
    sync_mode: str
    requester_kernel_id: str
    target_kernel_id: str
    promoted_count: int = 0
    summarized_count: int = 0
    discarded_count: int = 0
    promoted_keys: List[str] = field(default_factory=list)
    summary_key: str = ""
    remaining_branch_items: int = 0
    note: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_request(request: MemorySyncRequest | Dict[str, Any]) -> MemorySyncRequest:
    if isinstance(request, MemorySyncRequest):
        return request
    return MemorySyncRequest(**dict(request))


def _select_branch_memory(branch_memory: List[Dict[str, Any]], memory_kinds: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    wanted = {str(k).strip().lower() for k in memory_kinds if str(k).strip()}
    if not wanted:
        return list(branch_memory), []
    selected, remaining = [], []
    for item in branch_memory:
        kind = str(item.get("kind", "episodic")).lower()
        if kind in wanted:
            selected.append(item)
        else:
            remaining.append(item)
    return selected, remaining


def _summary_payload(items: List[Dict[str, Any]], requester_kernel_id: str, sync_mode: str) -> Dict[str, Any]:
    preview = []
    for item in items[:10]:
        preview.append({
            "kind": item.get("kind", "episodic"),
            "key": item.get("key", ""),
            "tags": list(item.get("tags", []))[:8],
            "goal_context": item.get("goal_context", ""),
        })
    return {
        "origin_kernel_id": requester_kernel_id,
        "sync_mode": sync_mode,
        "entry_count": len(items),
        "preview": preview,
    }


def apply_memory_sync_request(
    store: MemoryStore,
    provenance_graph: ProvenanceGraph,
    request: MemorySyncRequest | Dict[str, Any],
    branch_memory: List[Dict[str, Any]],
    *,
    cycle: int = 0,
    state: Any = None,
) -> Tuple[MemorySyncDecision, List[Dict[str, Any]]]:
    req = _normalize_request(request)
    selected, untouched = _select_branch_memory(list(branch_memory), list(req.memory_kinds))
    mode = str(req.sync_mode or "summarize").lower()
    decision = MemorySyncDecision(
        request_id=req.request_id,
        sync_mode=mode,
        requester_kernel_id=req.requester_kernel_id,
        target_kernel_id=req.target_kernel_id,
        remaining_branch_items=len(branch_memory),
    )

    for item in selected:
        provenance_graph.record_memory_entry(item, owner_kernel_id=req.requester_kernel_id, branch_id=req.requester_kernel_id, memory_kind=str(item.get("kind", "episodic")))

    if mode == "discard":
        decision.discarded_count = len(selected)
        decision.remaining_branch_items = len(untouched)
        decision.note = f"discarded {len(selected)} branch memory entries"
        provenance_graph.record_sync_event(
            request_id=req.request_id,
            sync_mode=mode,
            owner_kernel_id=req.requester_kernel_id,
            target_kernel_id=req.target_kernel_id,
            payload=decision.to_dict(),
        )
        return decision, untouched

    if mode == "promote":
        promoted_keys = []
        for item in selected:
            kind = str(item.get("kind", "episodic"))
            key = str(item.get("key") or f"branch_{req.requester_kernel_id}_{len(promoted_keys)+1}")
            data = dict(item.get("data", {}))
            data.setdefault("source_kernel_id", req.requester_kernel_id)
            data.setdefault("sync_request_id", req.request_id)
            tags = list(item.get("tags", []))
            store.store(kind, key, data, goal_context=str(item.get("goal_context", req.rationale)), tags=tags, cycle=cycle, state=state)
            promoted_keys.append(key)
            provenance_graph.record_memory_entry({**item, "kind": kind, "key": key}, owner_kernel_id=req.target_kernel_id or "main", branch_id=req.target_kernel_id or "main", memory_kind=kind)
        decision.promoted_count = len(promoted_keys)
        decision.promoted_keys = promoted_keys
        decision.remaining_branch_items = len(untouched)
        decision.note = f"promoted {len(promoted_keys)} branch memory entries"
        provenance_graph.record_sync_event(
            request_id=req.request_id,
            sync_mode=mode,
            owner_kernel_id=req.requester_kernel_id,
            target_kernel_id=req.target_kernel_id,
            promoted_keys=promoted_keys,
            payload=decision.to_dict(),
        )
        return decision, untouched

    # default: summarize
    summary_key = f"branch_summary_{req.requester_kernel_id}_{int(time.time())}"
    payload = _summary_payload(selected, req.requester_kernel_id, mode)
    payload["rationale"] = req.rationale
    payload["memory_kinds"] = list(req.memory_kinds)
    payload["selected_keys"] = [str(item.get("key", "")) for item in selected[:50]]
    try:
        payload["serialized_preview"] = json.dumps(payload["preview"], default=str)[:1000]
    except Exception:
        pass
    store.store("semantic", summary_key, payload, goal_context=req.rationale or "branch summary", tags=["branch_sync", req.requester_kernel_id], cycle=cycle, state=state)
    provenance_graph.record_memory_entry({"kind": "semantic", "key": summary_key, "data": payload, "tags": ["branch_sync", req.requester_kernel_id]}, owner_kernel_id=req.target_kernel_id or "main", branch_id=req.target_kernel_id or "main", memory_kind="semantic")
    decision.summarized_count = len(selected)
    decision.summary_key = summary_key
    decision.remaining_branch_items = len(untouched)
    decision.note = f"summarized {len(selected)} branch memory entries"
    provenance_graph.record_sync_event(
        request_id=req.request_id,
        sync_mode=mode,
        owner_kernel_id=req.requester_kernel_id,
        target_kernel_id=req.target_kernel_id,
        summary_key=summary_key,
        payload=decision.to_dict(),
    )
    return decision, untouched
