#!/usr/bin/env python3
"""Create deterministic held-out and adversarial validation shards by paradox family.

This makes `unseen_paradox_family_transfer` meaningful by ensuring selected
families are absent from train shards and present only in heldout/adversarial
validation shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

FAMILY_KEYS = ("family", "paradox_family", "shadow_family", "kind")
DEFAULT_HOLDOUT_HINTS = (
    "Yablo",
    "Knower",
    "Goodman",
    "Gettier",
    "measurement contextuality",
    "proof assumption collision",
)


def read_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                obj.setdefault("_source_file", path.name)
                yield obj


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = dict(r)
            obj.pop("_source_file", None)
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def family_of(row: Mapping[str, object]) -> str:
    for k in FAMILY_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    text = str(row.get("text") or "")
    return "unknown:" + hashlib.sha1(text[:500].encode("utf-8", "ignore")).hexdigest()[:8]


def choose_holdouts(families: Sequence[str], n: int, explicit: Sequence[str]) -> List[str]:
    fams = sorted(set(families))
    if explicit:
        selected = [f for f in fams if f in explicit or any(e.lower() in f.lower() for e in explicit)]
        if selected:
            return selected
    selected = [f for f in fams if any(h.lower() in f.lower() for h in DEFAULT_HOLDOUT_HINTS)]
    if len(selected) >= n:
        return selected[:n]
    for f in fams:
        if f not in selected:
            selected.append(f)
        if len(selected) >= n:
            break
    return selected


def adversarialize(row: Mapping[str, object], idx: int) -> Dict[str, object]:
    """Light paraphrase wrapper preserving UAP labels/profile.

    This is intentionally conservative: it changes surface/domain framing without
    altering the underlying support profile.
    """
    out = dict(row)
    text = str(out.get("text") or out.get("prompt") or "")
    wrappers = [
        "In a legal memo, restate the same obstruction without resolving it: ",
        "As a database integrity note, preserve both local supports: ",
        "In a proof-audit setting, keep the gap/glut profile intact: ",
        "As scientific model conflict prose, do not classicalize this: ",
    ]
    prefix = wrappers[idx % len(wrappers)]
    out["text"] = prefix + text
    out["adversarial_surface"] = True
    out["adversarial_transform"] = "domain_wrapper_preserve_uap_profile"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate v14.3.3 held-out/adversarial family split")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout-family", action="append", default=[])
    ap.add_argument("--n-holdout-families", type=int, default=6)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1433)
    ap.add_argument("--copy-tokenizer", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []
    for path in sorted(in_dir.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise SystemExit(f"no JSONL rows found in {in_dir}")

    families = [family_of(r) for r in rows]
    holdouts = set(choose_holdouts(families, args.n_holdout_families, args.holdout_family))

    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    heldout: List[Dict[str, object]] = []
    adv: List[Dict[str, object]] = []
    for i, r in enumerate(rows):
        fam = family_of(r)
        r = dict(r)
        r["v14_3_3_family"] = fam
        if fam in holdouts:
            r["heldout_family"] = True
            heldout.append(r)
            adv.append(adversarialize(r, i))
        else:
            # deterministic row split; avoids leaking selected families into train.
            h = int(hashlib.sha1(json.dumps(r, sort_keys=True, default=str).encode("utf-8", "ignore")).hexdigest(), 16)
            if (h % 10000) < int(args.val_ratio * 10000):
                val.append(r)
            else:
                train.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": write_jsonl(out_dir / "train_v14_3_3_family_split.jsonl", train),
        "val": write_jsonl(out_dir / "val_v14_3_3_family_split.jsonl", val),
        "heldout": write_jsonl(out_dir / "heldout_v14_3_3_family_split.jsonl", heldout),
        "adversarial": write_jsonl(out_dir / "val_adversarial_v14_3_3.jsonl", adv),
    }
    if args.copy_tokenizer:
        src = Path(args.copy_tokenizer)
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

    manifest = {
        "schema": "tovah-v14.3.3-heldout-family-split",
        "source_dir": str(in_dir),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "holdout_families": sorted(holdouts),
        "counts": counts,
        "note": "Families listed in holdout_families are excluded from train_v14_3_3_family_split.jsonl.",
    }
    (out_dir / "split_manifest_v14_3_3.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
