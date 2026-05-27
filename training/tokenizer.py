"""
TOVAH v14.1.2 training/tokenizer.py — Tokenizer abstraction.

AUDIT FIX (P1-1, v14.1.2): byte-level vocab=256 gives ~3.5x worse
sequence-length efficiency than BPE on natural language. This module
provides a uniform tokenizer interface so the same training code can
run with either encoding.

Backends:
  - "byte" (default, always available): vocab_size=256, encode = raw bytes
  - "bpe" (optional, requires `tokenizers` package): trained on corpus
    shards, vocab_size=8192 by default

If `tokenizers` is not installed, only "byte" backend is available and
attempts to use "bpe" log a warning and fall back to "byte".

Public:
  ByteTokenizer  — vocab=256 byte-level (always works)
  BPETokenizer   — trained BPE (vocab configurable, requires tokenizers)
  train_bpe(shard_dir, vocab_size, save_path) — trains+saves
  load_tokenizer(spec) — load by name ("byte") or path (".json")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer interface used by training."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def name(self) -> str: ...

    def encode(self, text: str, max_len: int) -> List[int]: ...

    def decode(self, ids: List[int]) -> str: ...

    def info(self) -> Dict[str, Any]: ...


# --- Byte-level (always available) -----------------------------------------

class ByteTokenizer:
    """UTF-8 byte-level tokenizer. vocab_size=256.

    The original TOVAH tokenizer. No training required, no dependencies.
    Use for live operation and for backward compatibility.
    """

    @property
    def vocab_size(self) -> int:
        return 256

    @property
    def name(self) -> str:
        return "byte"

    def encode(self, text: str, max_len: int) -> List[int]:
        raw = text.encode("utf-8", errors="ignore")[:max_len]
        if not raw:
            raw = b" "
        return list(raw)

    def decode(self, ids: List[int]) -> str:
        return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")

    def info(self) -> Dict[str, Any]:
        return {"name": "byte", "vocab_size": 256}


# --- BPE (optional) --------------------------------------------------------

class BPETokenizer:
    """BPE wrapper around HuggingFace `tokenizers` library."""

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self._vocab_size = hf_tokenizer.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def name(self) -> str:
        return "bpe"

    def encode(self, text: str, max_len: int) -> List[int]:
        enc = self._tok.encode(text)
        ids = enc.ids[:max_len]
        if not ids:
            # Use vocab_size-1 as fallback padding-friendly id; specific tokenizer
            # may have an explicit pad id, but we don't depend on that.
            ids = [0]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def info(self) -> Dict[str, Any]:
        return {"name": "bpe", "vocab_size": self._vocab_size}


def train_bpe(shard_dir: str | Path,
              *,
              vocab_size: int = 8192,
              save_path: Optional[str | Path] = None,
              max_files: int = 500) -> "BPETokenizer":
    """Train a byte-level BPE on JSONL corpus shards.

    Reads the `text` field from each line, feeds it to the tokenizer's
    BPE trainer. Saves to `save_path` (default tovah_corpus/tokenizer.json)
    if given.
    """
    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as e:
        raise RuntimeError(
            "BPE backend requires the `tokenizers` package. "
            "Install with: pip install tokenizers"
        ) from e

    shard_dir = Path(shard_dir)
    shards = sorted(shard_dir.glob("*.jsonl"))[:max_files]
    if not shards:
        raise FileNotFoundError(f"no JSONL shards in {shard_dir}")

    # Stream texts as iterator of strings.
    def _iter_texts():
        for shard in shards:
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
                        t = (d.get("text") or "").strip()
                        if t:
                            yield t
            except Exception as e:
                logging.warning(f"train_bpe: skipping {shard}: {e}")

    tok = HFTokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_iter_texts(), trainer=trainer)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_path))
        logging.info(f"BPE tokenizer saved to {save_path} (vocab_size={tok.get_vocab_size()})")

    return BPETokenizer(tok)


# --- Factory ---------------------------------------------------------------

def load_tokenizer(spec: str) -> Tokenizer:
    """Load a tokenizer by spec.

    Acceptable specs:
      "byte"               → ByteTokenizer
      "/path/to/tok.json"  → BPETokenizer loaded from disk
    """
    if spec in (None, "", "byte"):
        return ByteTokenizer()
    p = Path(spec)
    if p.exists() and p.is_file():
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError:
            logging.warning(
                "Tokenizer file %s exists but `tokenizers` is not installed; "
                "falling back to byte-level.", spec,
            )
            return ByteTokenizer()
        hf_tok = HFTokenizer.from_file(str(p))
        return BPETokenizer(hf_tok)
    logging.warning("Unknown tokenizer spec %r; falling back to byte-level.", spec)
    return ByteTokenizer()
