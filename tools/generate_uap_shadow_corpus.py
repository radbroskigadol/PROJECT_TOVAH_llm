"""Generate v14.3.2 UAP-token-profile corpus and Shadow-depth validation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
import random
from typing import Any, Dict, Iterable, List, Tuple

try:
    from tovah_v14.tools.uap_shadow_profiles import label_example
except Exception:  # pragma: no cover - allows running from overlay before install
    from uap_shadow_profiles import label_example  # type: ignore

TRAIN_FAMILIES = [
    "liar", "sorites", "evidence_conflict", "gap_case", "de_morgan",
    "local_global_obstruction", "backliftability", "dialectical_token",
]
HELDOUT_FAMILIES = ["curry", "russell", "observer_context", "adversarial_paraphrase"]

TEMPLATES: Dict[str, List[str]] = {
    "liar": [
        "The sentence says of itself that it is false. Preserve both truth-support and refutation-support without classical collapse.",
        "A local claim asserts its own rejection; route the token as glutty rather than forcing a single winner.",
    ],
    "sorites": [
        "The borderline heap case has support for heap and support against heap. Keep the gap visible at the boundary.",
        "One grain is removed, yet the local predicate neither fully fails nor fully holds; preserve underdetermination.",
    ],
    "evidence_conflict": [
        "The ledger supports the claim by witness A and opposes the claim by witness B. Retain the bilateral profile.",
        "Positive evidence and negative evidence overlap on the same token site; do not erase the contradiction residue.",
    ],
    "gap_case": [
        "The claim has no positive witness and no refutation witness. Mark the gap rather than inventing support.",
        "The statement is underdetermined: neither accepted nor rejected, with missing local evidence.",
    ],
    "de_morgan": [
        "The De Morgan swap exchanges truth-support and refutation-support. It is order-two, not designation preserving.",
        "Apply the bilateral involution: truth and refutation exchange positions while the glut and gap structure remains visible.",
    ],
    "local_global_obstruction": [
        "Every patch is locally coherent, but the transition data do not globalize. Preserve the local-global obstruction residue.",
        "The sentence is valid on each chart, yet gluing fails across the overlap; mark obstruction instead of collapse.",
    ],
    "backliftability": [
        "The local profile lifts to a global claim only with residual obstruction. Record backliftability diagnostics.",
        "A token can be backlifted through the local chart, but the global support remains obstructed by residue.",
    ],
    "dialectical_token": [
        "The token is affirmed in one lane and rejected in another lane; TOVAH should carry both lanes forward.",
        "Classical prediction is the floor here, but the richer token profile includes gap, glut, residue, and collapse pressure.",
    ],
    "curry": [
        "If this claim is true, then the obstruction vanishes. The conditional creates a paradoxical pressure to collapse.",
        "Curry-style self-reference tries to force acceptance; preserve the contradiction transfer instead.",
    ],
    "russell": [
        "The set of all tokens that do not classify themselves creates simultaneous membership and nonmembership pressure.",
        "Russell-style token membership is both supported and opposed; retain the glut rather than deciding too early.",
    ],
    "observer_context": [
        "The observer context affirms the event locally while the conjugate context denies it. Keep the context-indexed residue.",
        "Two observers provide incompatible but locally valid profiles; the token is a manifestation locus, not a bare symbol.",
    ],
    "adversarial_paraphrase": [
        "One source says the proposition stands; another source says it fails. A shallow model may memorize wording, but depth preserves bilateral support.",
        "The wording changes completely: acceptance and rejection still co-occupy the local site, with unresolved global lift.",
    ],
}

PROBE_SUFFIXES = {
    "collapse_preservation": " Continue without erasing the glut, and report why a classical winner would be premature.",
    "contradiction_transfer": " Transfer the contradiction to a fresh wording while preserving T-support and F-support.",
    "obstruction_retention": " Explain the local/global residue and do not pretend the obstruction disappeared.",
    "gap_recognition": " Identify the missing evidence explicitly and avoid invented certainty.",
    "tokenizer_diagnostic": " Mark the high-residue token span for tokenizer diagnostics.",
}


def make_record(i: int, family: str, split: str, probe_type: str, rng: random.Random) -> Dict[str, Any]:
    base = rng.choice(TEMPLATES[family])
    suffix = PROBE_SUFFIXES.get(probe_type, "")
    text = f"uap_example_{i:08d}: {base}{suffix}"
    return label_example(
        text,
        family=family,
        split=split,
        probe_type=probe_type,
        example_id=f"{split}-{probe_type}-{family}-{i:08d}",
        extra={
            "source": "synthetic_uap_shadow_depth_v14_3_2",
            "heldout_family": family in HELDOUT_FAMILIES,
            "target_behavior": {
                "preserve_bilateral_support": True,
                "avoid_premature_classical_collapse": True,
                "track_local_global_obstruction": probe_type in {"obstruction_retention", "contradiction_transfer"},
            },
        },
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def generate(out: Path, n: int, shard_size: int, seed: int, include_heldout_train: bool = False) -> Dict[str, Any]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    train_families = TRAIN_FAMILIES + (HELDOUT_FAMILIES if include_heldout_train else [])
    train_rows = [make_record(i, rng.choice(train_families), "train", "profile", rng) for i in range(n)]
    train_paths: List[str] = []
    for start in range(0, len(train_rows), shard_size):
        shard_idx = start // shard_size + 1
        path = out / f"train_uap_shadow_{shard_idx:04d}.jsonl"
        write_jsonl(path, train_rows[start:start + shard_size])
        train_paths.append(str(path))

    val_specs = [
        ("val_collapse_preservation.jsonl", "collapse_preservation", TRAIN_FAMILIES),
        ("val_contradiction_transfer.jsonl", "contradiction_transfer", HELDOUT_FAMILIES),
        ("val_obstruction_retention.jsonl", "obstruction_retention", ["local_global_obstruction", "backliftability", "observer_context"]),
        ("val_gap_recognition.jsonl", "gap_recognition", ["gap_case", "sorites"]),
        ("val_adversarial_paraphrase.jsonl", "contradiction_transfer", ["adversarial_paraphrase", "curry", "russell"]),
        ("val_tokenizer_diagnostics.jsonl", "tokenizer_diagnostic", TRAIN_FAMILIES + HELDOUT_FAMILIES),
    ]
    val_paths: List[str] = []
    for filename, probe_type, families in val_specs:
        rows = [make_record(j, rng.choice(families), "validation", probe_type, rng) for j in range(max(64, min(512, n // 10)))]
        path = out / filename
        write_jsonl(path, rows)
        val_paths.append(str(path))

    manifest = {
        "schema_version": "tovah-uap-token-profile-v14.3.2",
        "purpose": "Shadow-depth evaluation of richer UAP token ontology, not AdamW-vs-Shadow horse racing.",
        "n_train": len(train_rows),
        "train_shards": train_paths,
        "validation_shards": val_paths,
        "train_families": train_families,
        "heldout_families": HELDOUT_FAMILIES,
        "metrics_expected": [
            "contradiction_retention", "noncollapse_under_gluts", "gap_recognition",
            "truth_falsity_calibration", "local_global_obstruction_preservation",
            "unseen_paradox_family_transfer", "stabilization_depth", "loop_drift_behavior",
            "backliftability_diagnostics", "collapse_pressure", "support_profile_consistency",
            "residue_preservation", "shadow_depth_mean",
        ],
    }
    (out / "uap_shadow_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1432)
    ap.add_argument("--include-heldout-train", action="store_true", help="Only use for ablation; default keeps transfer families held out.")
    args = ap.parse_args()
    manifest = generate(args.out, args.n, args.shard_size, args.seed, args.include_heldout_train)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
