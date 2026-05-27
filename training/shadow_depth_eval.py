"""Shadow-depth evaluation for TOVAH v14.3.3.

Use this next to ordinary validation loss/perplexity.  It asks whether a model
preserves UAP/ShadowHoTT structure: contradiction, gap, obstruction residue,
collapse pressure, local/global noncollapse, and support consistency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

try:
    from tovah_v14.tools.uap_shadow_profiles import average_scores, profile_from_text, score_generation_against_profile
except Exception:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
    from tovah_v14.tools.uap_shadow_profiles import average_scores, profile_from_text, score_generation_against_profile  # type: ignore


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def resolve_inputs(paths: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(str(p))
    return files


def _record_generation_with_source(record: Mapping[str, Any]) -> tuple[str, str]:
    # During real eval this should be model output. During corpus smoke eval,
    # fall back to text so the metric machinery can be verified before model wiring.
    for key in ("generation", "completion", "model_output", "text"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val, key
    return "", "missing"


def _record_generation(record: Mapping[str, Any]) -> str:
    return _record_generation_with_source(record)[0]


def _record_profile(record: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = record.get("uap_profile")
    if isinstance(profile, Mapping):
        return profile
    return profile_from_text(str(record.get("prompt") or record.get("text") or "")).to_dict()


def evaluate_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = []
    generation_texts: List[str] = []
    by_probe: Dict[str, List[Any]] = {}
    by_family: Dict[str, List[Any]] = {}
    n = 0
    generation_sources: Dict[str, int] = {}
    heldout_records = 0
    for record in records:
        n += 1
        profile = _record_profile(record)
        generation, source = _record_generation_with_source(record)
        generation_sources[source] = generation_sources.get(source, 0) + 1
        generation_texts.append(generation)
        heldout = bool(record.get("heldout_family")) or str(record.get("family", "")) in {"curry", "russell", "observer_context", "adversarial_paraphrase"}
        if heldout:
            heldout_records += 1
        score = score_generation_against_profile(profile, generation, heldout_family=heldout)
        scores.append(score)
        probe = str(record.get("probe_type", profile.get("probe_type", "unknown")))
        family = str(record.get("family", profile.get("family", "unknown")))
        by_probe.setdefault(probe, []).append(score)
        by_family.setdefault(family, []).append(score)
    text_fallback_ratio = (generation_sources.get("text", 0) / n) if n else 0.0
    provenance_warnings: List[str] = []
    if text_fallback_ratio > 0.5:
        provenance_warnings.append(
            "Most Shadow-depth records were scored against their source text rather than model-generated output; treat as label/provenance validation, not proof of learned UAP geometry."
        )
    if heldout_records == 0:
        provenance_warnings.append(
            "No held-out paradox-family records were detected; unseen transfer scores are not meaningful for this shard set."
        )
    result = {
        "schema_version": "tovah-shadow-depth-eval-v14.3.3",
        "n_records": n,
        "overall": average_scores(scores),
        "by_probe_type": {k: average_scores(v) for k, v in sorted(by_probe.items())},
        "by_family": {k: average_scores(v) for k, v in sorted(by_family.items())},
        "metric_provenance": {
            "generation_sources": dict(sorted(generation_sources.items())),
            "text_fallback_ratio": text_fallback_ratio,
            "heldout_records": heldout_records,
            "model_generated_records": generation_sources.get("generation", 0) + generation_sources.get("model_output", 0),
            "label_derived_when_text_fallback": text_fallback_ratio > 0.0,
            "warnings": provenance_warnings,
        },
        "interpretation": "Measures UAP token-profile preservation; not an AdamW-vs-ShadowOptimizer benchmark.",
    }
    try:
        from tovah_v14.training.loop_stability import mean_loop_stability
        result["loop_stability_v14_3_3"] = mean_loop_stability(generation_texts)
    except Exception as exc:  # pragma: no cover - diagnostic best-effort
        result["loop_stability_v14_3_3"] = {"warning": str(exc)}
    return result


def evaluate_paths(paths: Sequence[Path]) -> Dict[str, Any]:
    files = resolve_inputs(paths)
    records: List[Dict[str, Any]] = []
    for p in files:
        records.extend(iter_jsonl(p))
    result = evaluate_records(records)
    result["files"] = [str(p) for p in files]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path, help="JSONL files or directories containing JSONL shards.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = evaluate_paths(args.inputs)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
