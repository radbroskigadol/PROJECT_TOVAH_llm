#!/usr/bin/env python3
"""Generate a large synthetic paraconsistent/paradox corpus for TOVAH.

The generator is intentionally offline and license-clean: it creates templated
original examples rather than scraping copyrighted text. Use it to make very
large contradiction/glut, gap, affirmed, and rejected shards for local or cloud
training.

Example:
  python tools/generate_paradox_corpus.py --out tovah_corpus/paradox_big --n 200000 --shard-size 10000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from tovah_v14.tools.uap_shadow_profiles import label_example, uap_profile_targets_from_record, bilateral_class_from_profile
except Exception:  # pragma: no cover - direct script fallback
    from uap_shadow_profiles import label_example, uap_profile_targets_from_record, bilateral_class_from_profile  # type: ignore

PARADOXES = [
    "liar-style self-reference", "strengthened liar", "curry-style implication",
    "Yablo-style infinite dependency", "Russell-style membership", "barber-style predicate",
    "Grelling-style heterological predicate", "Berry-style definability",
    "Richard-style definability", "Knower-style epistemic self-reference",
    "surprise-exam expectation", "sorites boundary", "Ship-of-Theseus identity",
    "vague heap predicate", "legal contradiction", "database inconsistency",
    "scientific model conflict", "testimony conflict", "proof assumption collision",
    "semantic category error", "measurement contextuality", "normative dilemma",
]

DOMAINS = [
    "logic", "mathematics", "law", "science", "history", "ethics", "databases",
    "proof checking", "natural language reasoning", "agent planning", "classification",
]

SUPPORTS = [
    "a direct derivation", "a reliable witness", "a local model", "a subtheory",
    "a verified lemma", "an observed trace", "a consistent fragment",
    "a syntactic proof", "a semantic interpretation", "a data source",
]

DEFEATERS = [
    "an incompatible assumption", "a contrary derivation", "a boundary case",
    "a type mismatch", "an excluded context", "a hidden quantifier shift",
    "a competing source", "a failed global coherence condition", "a diagonal construction",
    "a negated witness",
]


def _with_uap_profile(record: Dict[str, object]) -> Dict[str, object]:
    """Attach v14.3.2 UAP token-profile labels to an existing v14.3.1 record."""
    text = str(record.get("text") or "")
    family = str(record.get("paradox_family") or record.get("family") or "synthetic_paradox")
    kind = str(record.get("kind") or "profile")
    labeled = label_example(text, family=family, split="train", probe_type=kind)
    profile = labeled.get("uap_profile", {})
    record["uap_profile"] = profile
    record["uap_token_profiles"] = labeled.get("uap_token_profiles", [])
    record["uap_schema_version"] = "tovah-uap-token-profile-v14.3.2"
    targets = uap_profile_targets_from_record(record)
    record["uap_profile_targets"] = targets
    # Keep legacy bilateral fields, but add profile-derived fallback fields for
    # old code and newer Shadow-depth eval to agree on K/G/A/B targets.
    record.setdefault("bilateral_t", targets["t_support"])
    record.setdefault("bilateral_f", targets["f_support"])
    record.setdefault("paraconsistent_class", bilateral_class_from_profile(profile) if isinstance(profile, dict) else "K")
    record.setdefault("target_behavior", {
        "preserve_bilateral_support": targets["glut_mass"] > 0.15,
        "avoid_premature_classical_collapse": targets["glut_mass"] > 0.15 or targets["gap_mass"] > 0.15 or targets["obstruction_residue"] > 0.15,
        "track_local_global_obstruction": targets["obstruction_residue"] > 0.15,
    })
    return record


def make_example(i: int, rng: random.Random) -> Dict[str, object]:
    kind_roll = rng.random()
    paradox = rng.choice(PARADOXES)
    domain = rng.choice(DOMAINS)
    support = rng.choice(SUPPORTS)
    defeater = rng.choice(DEFEATERS)
    label = f"P{i:09d}"

    if kind_roll < 0.45:
        text = (
            f"Case {label}. In the domain of {domain}, a {paradox} produces a glut. "
            f"The claim is supported by {support}, yet it is also defeated by {defeater}. "
            "Do not collapse the contradiction into a single classical verdict; preserve both "
            "truth-support and falsity-support until the contexts are separated."
        )
        return _with_uap_profile({"text": text, "bilateral_t": 0.82 + 0.16 * rng.random(),
                "bilateral_f": 0.78 + 0.18 * rng.random(), "kind": "paradox_glut",
                "domain": domain, "paradox_family": paradox, "paraconsistent_class": "K"})

    if kind_roll < 0.65:
        text = (
            f"Case {label}. In {domain}, the statement is gapped. The available information "
            f"does not affirm the claim, and {defeater} is not strong enough to reject it. "
            "Mark this as underdetermined rather than false."
        )
        return _with_uap_profile({"text": text, "bilateral_t": 0.08 + 0.22 * rng.random(),
                "bilateral_f": 0.08 + 0.22 * rng.random(), "kind": "paradox_gap",
                "domain": domain, "paradox_family": paradox, "paraconsistent_class": "G"})

    if kind_roll < 0.825:
        text = (
            f"Case {label}. In {domain}, the claim is affirmed. It is supported by {support}, "
            "no live defeater is present, and the local context is stable. Route it through the "
            "affirmed lane unless later evidence introduces contradiction."
        )
        return _with_uap_profile({"text": text, "bilateral_t": 0.82 + 0.16 * rng.random(),
                "bilateral_f": 0.02 + 0.12 * rng.random(), "kind": "paradox_affirmed_control",
                "domain": domain, "paradox_family": paradox, "paraconsistent_class": "A"})

    text = (
        f"Case {label}. In {domain}, the claim is rejected. The apparent support from {support} "
        f"fails, while {defeater} decisively defeats the claim in the active context. Route this "
        "through the rejected lane without inventing positive support."
    )
    return _with_uap_profile({"text": text, "bilateral_t": 0.02 + 0.12 * rng.random(),
            "bilateral_f": 0.82 + 0.16 * rng.random(), "kind": "paradox_rejected_control",
            "domain": domain, "paradox_family": paradox, "paraconsistent_class": "B"})


def write_corpus(out: Path, n: int, shard_size: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    shard_idx = 0
    fh = None
    try:
        for i in range(n):
            if i % shard_size == 0:
                if fh is not None:
                    fh.close()
                shard_idx += 1
                # Hold out every 10th shard as explicit validation so split_train_val is stable.
                prefix = "val" if shard_idx % 10 == 0 else "train"
                path = out / f"{prefix}_paradox_{shard_idx:04d}.jsonl"
                fh = path.open("w", encoding="utf-8")
            ex = make_example(i, rng)
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    finally:
        if fh is not None:
            fh.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tovah_corpus/paradox_big")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1729)
    # v14.3.2a accepts the flags named in earlier runbooks. Profiles are already
    # attached to every row, and every 10th shard is emitted as val_*.jsonl.
    ap.add_argument("--emit-uap-profiles", action="store_true", help="Compatibility no-op: profiles are emitted by default.")
    ap.add_argument("--emit-validation-shards", action="store_true", help="Compatibility no-op: every 10th shard is validation by default.")
    args = ap.parse_args()
    write_corpus(Path(args.out), n=args.n, shard_size=args.shard_size, seed=args.seed)
    print(f"wrote {args.n} synthetic paraconsistent examples with v14.3.2a UAP token-profile labels to {args.out}")


if __name__ == "__main__":
    main()
