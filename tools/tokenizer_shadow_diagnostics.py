"""Tokenizer diagnostics for UAP glut/gap/residue regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from tovah_v14.tools.uap_shadow_profiles import profile_from_text
except Exception:  # pragma: no cover
    from tovah_v14.tools.uap_shadow_profiles import profile_from_text  # type: ignore

HIGH_VALUE_KEYS = ("glut_mass", "gap_mass", "obstruction_residue")
REGION_TERMS = [
    "truth-support", "refutation-support", "falsity-support", "glut", "gap", "obstruction",
    "residue", "local-global", "backliftability", "collapse", "classicalization", "bilateral",
    "paraconsistent", "De Morgan", "involution",
]


def _load_tokenizer(path: str | None):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
        return Tokenizer.from_file(path)
    except Exception:
        return None


def _encode(tokenizer: Any, text: str) -> List[str]:
    if tokenizer is None:
        return re.findall(r"\S+", text)
    try:
        enc = tokenizer.encode(text)
        return list(enc.tokens)
    except Exception:
        return re.findall(r"\S+", text)


def _fragmentation(tokens: Sequence[str], term: str) -> float:
    if not term:
        return 0.0
    joined = "".join(t.replace("Ġ", " ") for t in tokens).lower()
    if term.lower() not in joined and term.lower().replace("-", " ") not in joined:
        return 0.0
    term_parts = re.split(r"[-_\s]+", term.lower())
    hits = sum(1 for p in term_parts if p and any(p in tok.lower() for tok in tokens))
    return max(0.0, len(term_parts) - max(1, hits)) / max(1, len(term_parts))


def diagnose_text(text: str, tokenizer_path: str | None = None) -> Dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_path)
    tokens = _encode(tokenizer, text)
    profile = profile_from_text(text)
    high_region = max(getattr(profile, k) for k in HIGH_VALUE_KEYS)
    term_frag = {term: _fragmentation(tokens, term) for term in REGION_TERMS if term.lower() in text.lower()}
    avg_frag = sum(term_frag.values()) / len(term_frag) if term_frag else 0.0
    return {
        "schema_version": "tovah-tokenizer-shadow-diagnostics-v14.3.2",
        "tokenizer": tokenizer_path or "whitespace-fallback",
        "n_chars": len(text),
        "n_tokens": len(tokens),
        "uap_profile": profile.to_dict(),
        "high_glut_gap_residue_region": high_region,
        "region_term_fragmentation": term_frag,
        "avg_region_fragmentation": avg_frag,
        "diagnostic_warning": bool(high_region > 0.35 and avg_frag > 0.25),
        "warning_meaning": "High glut/gap/residue text is being fragmented around semantic-control terms." if high_region > 0.35 and avg_frag > 0.25 else "none",
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for p in args.inputs:
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for file in files:
            for rec in iter_jsonl(file):
                rows.append({"file": str(file), "id": rec.get("id"), **diagnose_text(str(rec.get("text", "")), args.tokenizer)})
    report = {
        "n": len(rows),
        "avg_fragmentation": sum(r["avg_region_fragmentation"] for r in rows) / len(rows) if rows else 0.0,
        "warnings": sum(1 for r in rows if r["diagnostic_warning"]),
        "rows": rows[:1000],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
