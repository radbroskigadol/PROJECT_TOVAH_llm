"""
TOVAH v14.1.2 training/dataset.py — PyTorch Dataset + DataLoader.

AUDIT FIX (P1-2, v14.1.2): the v14.1.1 pretrain() loaded the whole
corpus into Python memory, parsed JSON on the main thread, and offered
no shuffling/prefetching/distributed support. This module replaces the
in-memory pool with a streaming Dataset that:

  - Supports torch DataLoader with `num_workers` and `pin_memory`.
  - Optionally filters by paraconsistent class (A/B/K/G).
  - Optionally chunks long examples into multiple training items.
  - Length-stratified sampling (rare long texts get proportionally more
    gradient updates than short ones).

Public:
  CorpusShardDataset(...) — torch.utils.data.IterableDataset
  build_collate_fn(tokenizer, max_len) — collate batches into tensors
"""
from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from tovah_v14.training.corpus_builder import _chunk_text, strip_envelope
from tovah_v14.tools.uap_shadow_profiles import PROFILE_KEYS, uap_profile_targets_from_record


def _classify_inline(t: float, f: float) -> str:
    """ABKG classification (matches quality_filter thresholds)."""
    t_high = t >= 0.55
    f_high = f >= 0.55
    if t_high and f_high:
        return "K"
    if t_high and not f_high:
        return "A"
    if not t_high and f_high:
        return "B"
    return "G"


class CorpusShardDataset(IterableDataset):
    """Streaming dataset over JSONL corpus shards.

    Args:
      shard_dir: directory containing tovah_stream_*.jsonl (or similar).
      max_len: max token length per example.
      class_filter: only yield examples whose paraconsistent class is in
        this set (e.g. {"A", "K"}). None = no filter.
      kind_filter: only yield examples whose kind is in this set.
      strip_envelope_text: if True (default), pass text through
        strip_envelope() so envelope syntax doesn't pollute training.
      chunk_long_text: if True, split texts longer than max_len into
        multiple chunks (chunk_overlap bytes between consecutive chunks).
      length_stratified: if True, yield longer texts proportionally more
        often (sqrt-weighted) to compensate for their lower frequency.
      seed: shuffle seed for reproducibility.

    Worker-aware: when DataLoader uses num_workers>0, each worker sees a
    disjoint partition of shards.
    """

    def __init__(self,
                 shard_dir: str | Path,
                 *,
                 max_len: int = 1024,
                 class_filter: Optional[Set[str]] = None,
                 kind_filter: Optional[Set[str]] = None,
                 strip_envelope_text: bool = True,
                 chunk_long_text: bool = True,
                 chunk_overlap: int = 128,
                 length_stratified: bool = False,
                 seed: int = 1234,
                 shuffle_shards: bool = True,
                 reservoir_size: int = 8192):
        """
        reservoir_size: in-worker reservoir buffer size for streaming shuffle.
          Larger = better I.I.D. approximation, more RAM per worker.
          Set to 0 or 1 to disable (legacy v14.2.6 deterministic emission).
          Default 8192 examples ≈ ~50–200 MB per worker at typical sizes.
        """
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.max_len = int(max_len)
        self.class_filter = class_filter
        self.kind_filter = kind_filter
        self.strip_envelope_text = strip_envelope_text
        self.chunk_long_text = chunk_long_text
        self.chunk_overlap = int(chunk_overlap)
        self.length_stratified = length_stratified
        self.seed = int(seed)
        self.shuffle_shards = shuffle_shards
        self.reservoir_size = max(0, int(reservoir_size))

    def _list_shards(self) -> List[Path]:
        return sorted(self.shard_dir.glob("*.jsonl"))

    def _partition_for_worker(self, shards: List[Path]) -> List[Path]:
        info = get_worker_info()
        if info is None:
            return shards
        # Round-robin partition: worker i gets shards [i::num_workers].
        return shards[info.id :: info.num_workers]

    def _yield_chunks(self, text: str) -> Iterator[str]:
        if not self.chunk_long_text:
            yield text
            return
        # Use byte budget = max_len for chunking; for byte tokenizer this
        # is exact, for BPE it's a generous upper bound (BPE compresses
        # ~3.5x so a max_len-byte chunk fits comfortably).
        for chunk in _chunk_text(text, chunk_bytes=self.max_len,
                                 overlap_bytes=self.chunk_overlap):
            if chunk.strip():
                yield chunk

    def _stream_shard(self, shard: Path) -> Iterator[Dict[str, Any]]:
        """Yield filtered/chunked training items from one shard. Pure stream:
        no shuffling here — the reservoir at __iter__ handles that."""
        try:
            with open(shard, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if self.kind_filter is not None and d.get("kind") not in self.kind_filter:
                        continue
                    t = float(d.get("bilateral_t", 0.5) or 0.5)
                    f = float(d.get("bilateral_f", 0.5) or 0.5)
                    cls = d.get("paraconsistent_class") or _classify_inline(t, f)
                    if self.class_filter is not None and cls not in self.class_filter:
                        continue
                    text = (d.get("text") or "").strip()
                    if self.strip_envelope_text:
                        text = strip_envelope(text)
                    if len(text) < 4:
                        continue
                    # Length-stratified: emit short texts once, long
                    # texts up to sqrt(len/max_len) times.
                    n_emit = 1
                    if self.length_stratified:
                        ratio = len(text) / max(1, self.max_len)
                        n_emit = max(1, int(round(math.sqrt(ratio))))
                    for _ in range(n_emit):
                        for chunk in self._yield_chunks(text):
                            targets = uap_profile_targets_from_record(d)
                            yield {
                                "text": chunk,
                                "bilateral_t": t,
                                "bilateral_f": f,
                                "paraconsistent_class": cls,
                                "kind": d.get("kind", ""),
                                "outcome_label": d.get("outcome_label", ""),
                                "lineage_id": d.get("lineage_id", ""),
                                "uap_profile": d.get("uap_profile", {}),
                                "uap_profile_targets": targets,
                                "uap_schema_version": d.get("uap_schema_version", ""),
                            }
        except Exception as e:
            logging.warning(f"CorpusShardDataset: skipping {shard}: {e}")

    def __iter__(self):
        shards = self._list_shards()
        if self.shuffle_shards:
            rng = random.Random(self.seed)
            shards = list(shards)
            rng.shuffle(shards)
        shards = self._partition_for_worker(shards)
        if not shards:
            return
        # Worker-local RNG so different workers see different shuffles even
        # though they share the same base seed. v14.2.6 used `seed+1` for
        # all workers, which was a no-op for the deterministic path.
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        res_rng = random.Random(self.seed + 17 * (worker_id + 1))

        if self.reservoir_size <= 1:
            # Legacy v14.2.6 path: deterministic stream. Kept for ablation
            # and for callers that opt out of the reservoir (e.g. eval).
            for shard in shards:
                yield from self._stream_shard(shard)
            return

        # Reservoir sampling across active shard streams.
        # AUDIT FIX (v14.2.7, sec 3): the v14.2.6 implementation streamed each
        # worker's shards in deterministic order, with no within-worker shuffle.
        # That left gradient updates vulnerable to systematic file-level
        # ordering bias. The reservoir buffer below preserves the streaming
        # memory profile while approximating a global I.I.D. shuffle across
        # the worker's partition.
        buffer: List[Dict[str, Any]] = []
        cap = self.reservoir_size
        for shard in shards:
            for item in self._stream_shard(shard):
                if len(buffer) < cap:
                    buffer.append(item)
                else:
                    # Reservoir replacement: with probability cap/(seen+1)
                    # we'd keep the new item; equivalently, evict a random
                    # slot. Since the buffer is full and we want to emit
                    # samples as we go (not at the end only), we evict a
                    # random slot, yield the evicted item, and replace it
                    # with the new one. This gives an exact uniform mix
                    # over a sliding window of `cap` examples.
                    idx = res_rng.randrange(cap)
                    yield buffer[idx]
                    buffer[idx] = item
        # Drain the residual buffer in random order.
        res_rng.shuffle(buffer)
        for item in buffer:
            yield item


def build_collate_fn(tokenizer, max_len: int, pad_id: int = 0):
    """Return a collate_fn that turns a list-of-dicts batch into tensors.

    Output:
      {
        "input_ids": LongTensor(B, L) — token ids (right-padded with pad_id)
        "target_ids": LongTensor(B, L) — shifted by one (for next-token loss)
        "attention_mask": FloatTensor(B, L) — 1 where real content, 0 where pad
        "bilateral_t": FloatTensor(B,)
        "bilateral_f": FloatTensor(B,)
        "kinds": list[str]
      }
    """
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids_list = [tokenizer.encode(item["text"], max_len) for item in batch]
        # We need at least 2 ids per example.
        ids_list = [ids if len(ids) >= 2 else (ids + [pad_id, pad_id])[:max_len] for ids in ids_list]
        ml = min(max_len, max(len(ids) for ids in ids_list))
        if ml < 2:
            ml = 2
        xb, yb, mask = [], [], []
        for ids in ids_list:
            ids = list(ids[:ml])
            true_len = len(ids)
            if true_len < ml:
                ids = ids + [pad_id] * (ml - true_len)
            xb.append(ids[:-1])
            yb.append(ids[1:])
            mask.append([1.0 if i < true_len - 1 else 0.0 for i in range(ml - 1)])
        out = {
            "input_ids": torch.tensor(xb, dtype=torch.long),
            "target_ids": torch.tensor(yb, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float),
            "bilateral_t": torch.tensor([item["bilateral_t"] for item in batch], dtype=torch.float),
            "bilateral_f": torch.tensor([item["bilateral_f"] for item in batch], dtype=torch.float),
            "kinds": [item.get("kind", "") for item in batch],
            "paraconsistent_classes": [item.get("paraconsistent_class", "") for item in batch],
            "uap_schema_versions": [item.get("uap_schema_version", "") for item in batch],
        }
        # v14.3.2: B-shaped scalar targets for the UAP/ShadowHoTT auxiliary
        # ontology.  They are optional in older shards and synthesized from
        # bilateral_t/bilateral_f when missing.
        profile_targets = []
        for item in batch:
            targets = item.get("uap_profile_targets")
            if not isinstance(targets, dict):
                targets = uap_profile_targets_from_record(item)
            profile_targets.append(targets)
        out["uap_profile_targets"] = {
            key: torch.tensor([float(t.get(key, 0.0) or 0.0) for t in profile_targets], dtype=torch.float)
            for key in PROFILE_KEYS
        }
        return out

    return _collate
