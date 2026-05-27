"""
TOVAH v14.3.3 training/eval.py — tokenizer-aware model evaluation harness.

Public:
  held_out_perplexity(model, shard_dir, tokenizer=..., ...) -> dict
  token_top1_accuracy(model, shard_dir, tokenizer=..., ...) -> dict
  gen_sample(model, prompt, tokenizer=..., ...) -> str
  detect_divergence(loss_history, ...) -> dict
  bilateral_calibration(model, shard_dir, tokenizer=..., ...) -> dict
  run_full_eval(model, shard_dir, tokenizer=..., ...) -> dict

CLI:
  python -m tovah_v14.training.eval --checkpoint CKPT --shard-dir SHARDS --tokenizer tokenizer.json --out eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Allow both supported invocation styles:
#   python -m tovah_v14.training.eval  (from package parent)
#   python training/eval.py            (from inside the tovah_v14 folder)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.training.tokenizer import ByteTokenizer
from tovah_v14.training.shadow_depth_eval import evaluate_paths as shadow_depth_evaluate_paths
from tovah_v14.config.constants import MODEL_PROFILES
from tovah_v14.neural.checkpointing import load_training_checkpoint
from tovah_v14.training.tokenizer import load_tokenizer


# --- Shard partitioning ----------------------------------------------------

def _list_shards(shard_dir: Path) -> List[Path]:
    return sorted(Path(shard_dir).glob("*.jsonl"))


def split_train_val(shard_dir: str | Path,
                    val_fraction: float = 0.1,
                    seed: int = 1234) -> Tuple[List[Path], List[Path]]:
    """Split shards into train and val deterministically.

    If explicit validation shards exist (``val_*.jsonl`` or ``*_val*.jsonl``),
    use those as validation and all other JSONL shards as training. This lets
    local runs keep stable, domain-balanced validation files. Otherwise, fall
    back to the historical deterministic shard-index split.
    """
    shards = _list_shards(Path(shard_dir))
    if not shards:
        return [], []
    explicit_val = [
        s for s in shards
        if s.name.startswith("val_") or "_val" in s.name or s.name.startswith("validation_")
    ]
    if explicit_val and len(explicit_val) < len(shards):
        val_set = set(explicit_val)
        return [s for s in shards if s not in val_set], explicit_val

    rng = random.Random(seed)
    indices = list(range(len(shards)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(shards) * val_fraction))) if val_fraction > 0 else 0
    n_val = min(n_val, len(shards) - 1) if val_fraction < 1 else len(shards)
    val_idx = set(indices[:n_val])
    train = [s for i, s in enumerate(shards) if i not in val_idx]
    val = [s for i, s in enumerate(shards) if i in val_idx]
    return train, val


def _iter_jsonl(shards: List[Path], max_examples: Optional[int] = None):
    """Yield parsed JSONL dicts from a list of shards, optionally capped."""
    n = 0
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
                    yield d
                    n += 1
                    if max_examples is not None and n >= max_examples:
                        return
        except Exception as e:
            logging.warning("eval: skipping shard %s: %s", shard, e)


def _tok(tokenizer=None):
    return tokenizer if tokenizer is not None else ByteTokenizer()


def _encode_text(text: str, tokenizer, max_len: int) -> List[int]:
    ids = tokenizer.encode(text, max_len)
    if len(ids) < 2:
        ids = (ids + [0, 0])[:2]
    return ids[:max_len]


# --- Held-out perplexity ---------------------------------------------------

def held_out_perplexity(
    model: ShadowTokenCore,
    shard_dir: str | Path,
    *,
    val_shards: Optional[List[Path]] = None,
    val_fraction: float = 0.1,
    max_examples: int = 500,
    device: str = "cpu",
    batch_size: int = 4,
    tokenizer=None,
) -> Dict[str, Any]:
    """Compute tokenizer-level cross-entropy perplexity on held-out shards."""
    tokenizer = _tok(tokenizer)
    shard_dir = Path(shard_dir)
    if val_shards is None:
        _, val_shards = split_train_val(shard_dir, val_fraction=val_fraction)
    if not val_shards:
        return {"perplexity": float("nan"), "bits_per_byte": float("nan"),
                "bits_per_token": float("nan"), "cross_entropy_nats": float("nan"),
                "n_tokens": 0, "n_examples": 0,
                "val_shards": [], "warning": "no validation shards available",
                "tokenizer": tokenizer.info()}

    model.eval()
    total_ce_nats = 0.0
    total_tokens = 0
    n_examples = 0
    batch_texts: List[str] = []

    def _process_batch(texts: List[str]) -> None:
        nonlocal total_ce_nats, total_tokens
        if not texts:
            return
        encoded = []
        for txt in texts:
            encoded.append(_encode_text(txt, tokenizer, model.max_len))
        ml = min(model.max_len, max(len(x) for x in encoded))
        xb, yb, mask = [], [], []
        for ids in encoded:
            ids = ids[:ml]
            true_len = len(ids)
            if len(ids) < ml:
                ids = ids + [0] * (ml - len(ids))
            xb.append(ids[:-1])
            yb.append(ids[1:])
            mask.append([1 if i < true_len - 1 else 0 for i in range(ml - 1)])
        x = torch.tensor(xb, dtype=torch.long, device=device)
        y = torch.tensor(yb, dtype=torch.long, device=device)
        m = torch.tensor(mask, dtype=torch.float, device=device)
        with torch.no_grad():
            tl, _, _ = model(x)
        logp = F.log_softmax(tl, dim=-1)
        true_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce_tokens = -(true_logp * m).sum().item()
        n_t = int(m.sum().item())
        if n_t > 0:
            total_ce_nats += ce_tokens
            total_tokens += n_t

    for d in _iter_jsonl(val_shards, max_examples=max_examples):
        text = (d.get("text") or "").strip()
        if len(text) < 2:
            continue
        batch_texts.append(text)
        n_examples += 1
        if len(batch_texts) >= batch_size:
            _process_batch(batch_texts)
            batch_texts = []
    _process_batch(batch_texts)

    if total_tokens == 0:
        return {"perplexity": float("nan"), "bits_per_byte": float("nan"),
                "bits_per_token": float("nan"), "cross_entropy_nats": float("nan"),
                "n_tokens": 0, "n_examples": n_examples,
                "val_shards": [str(s) for s in val_shards],
                "warning": "no tokens scored", "tokenizer": tokenizer.info()}

    mean_ce_nats = total_ce_nats / total_tokens
    bits_per_token = mean_ce_nats / math.log(2)
    perplexity = math.exp(mean_ce_nats)
    return {
        "perplexity": perplexity,
        # Backward-compatible key; for BPE this is token-level bits, not bytes.
        "bits_per_byte": bits_per_token,
        "bits_per_token": bits_per_token,
        "cross_entropy_nats": mean_ce_nats,
        "n_tokens": total_tokens,
        "n_examples": n_examples,
        "val_shards": [str(s) for s in val_shards],
        "tokenizer": tokenizer.info(),
    }


# --- Top-1 token accuracy --------------------------------------------------

def token_top1_accuracy(
    model: ShadowTokenCore,
    shard_dir: str | Path,
    *,
    val_shards: Optional[List[Path]] = None,
    val_fraction: float = 0.1,
    max_examples: int = 500,
    device: str = "cpu",
    tokenizer=None,
) -> Dict[str, Any]:
    """Fraction of next-token predictions where argmax == truth."""
    tokenizer = _tok(tokenizer)
    shard_dir = Path(shard_dir)
    if val_shards is None:
        _, val_shards = split_train_val(shard_dir, val_fraction=val_fraction)
    if not val_shards:
        return {"top1_accuracy": float("nan"), "n_tokens": 0, "warning": "no val shards",
                "tokenizer": tokenizer.info()}

    model.eval()
    correct = 0
    total = 0
    for d in _iter_jsonl(val_shards, max_examples=max_examples):
        text = (d.get("text") or "").strip()
        if len(text) < 2:
            continue
        ids = _encode_text(text, tokenizer, model.max_len)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device)
        with torch.no_grad():
            tl, _, _ = model(x)
        preds = tl[0].argmax(dim=-1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    if total == 0:
        return {"top1_accuracy": float("nan"), "n_tokens": 0, "tokenizer": tokenizer.info()}
    return {"top1_accuracy": correct / total, "n_correct": correct,
            "n_tokens": total, "random_baseline": 1.0 / tokenizer.vocab_size,
            "tokenizer": tokenizer.info()}


# --- Generation sample (qualitative) ---------------------------------------

def gen_sample(
    model: ShadowTokenCore,
    prompt: str,
    *,
    max_tokens: int = 200,
    temperature: float = 0.8,
    device: str = "cpu",
    seed: Optional[int] = None,
    tokenizer=None,
) -> str:
    """Generate a continuation from ``prompt`` using the active tokenizer."""
    tokenizer = _tok(tokenizer)
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    try:
        ids = tokenizer.encode(prompt, max(1, model.max_len - 1))
        if len(ids) < 1:
            ids = [0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        produced: List[int] = []
        for _ in range(max_tokens):
            with torch.no_grad():
                tl, _, _ = model(x)
            last_logits = tl[0, -1]
            if temperature <= 1e-6:
                next_id = int(last_logits.argmax().item())
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            produced.append(next_id)
            new_ids = (ids + produced)[-(model.max_len - 1):]
            x = torch.tensor([new_ids], dtype=torch.long, device=device)
        return tokenizer.decode(ids + produced)
    except Exception as e:
        logging.warning("gen_sample failed: %s", e)
        return prompt + f" [generation failed: {e}]"


# --- Divergence detector ---------------------------------------------------

def detect_divergence(
    loss_history: Sequence[float],
    *,
    window: int = 100,
    blowup_ratio: float = 10.0,
) -> Dict[str, Any]:
    """Flag pathological training trajectories."""
    if not loss_history:
        return {"diverging": False, "reason": "empty_history"}
    last = float(loss_history[-1])
    if not math.isfinite(last):
        return {"diverging": True, "reason": "non_finite_loss", "last": last}
    recent = loss_history[-window:]
    if len(recent) < max(8, window // 4):
        return {"diverging": False, "reason": "insufficient_history",
                "n_recent": len(recent)}
    med = statistics.median(recent[:-1] or recent)
    if med > 0 and last > blowup_ratio * med:
        return {"diverging": True, "reason": "blowup",
                "last": last, "median_recent": med, "ratio": last / med}
    tail = recent[-(window // 4):]
    if len(tail) >= 4 and all(tail[i] < tail[i+1] for i in range(len(tail) - 1)):
        return {"diverging": True, "reason": "monotone_increasing",
                "tail_sample": list(tail[:8])}
    return {"diverging": False, "reason": "ok", "last": last, "median_recent": med}


# --- Bilateral calibration --------------------------------------------------

def bilateral_calibration(
    model: ShadowTokenCore,
    shard_dir: str | Path,
    *,
    val_shards: Optional[List[Path]] = None,
    val_fraction: float = 0.1,
    max_examples: int = 200,
    device: str = "cpu",
    tokenizer=None,
) -> Dict[str, Any]:
    """Probe whether model entropy tracks labeled bilateral mass."""
    tokenizer = _tok(tokenizer)
    shard_dir = Path(shard_dir)
    if val_shards is None:
        _, val_shards = split_train_val(shard_dir, val_fraction=val_fraction)
    if not val_shards:
        return {"warning": "no val shards", "tokenizer": tokenizer.info()}

    model.eval()
    entropies = []
    t_labels = []
    f_labels = []
    for d in _iter_jsonl(val_shards, max_examples=max_examples):
        text = (d.get("text") or "").strip()
        if len(text) < 4:
            continue
        try:
            ids = tokenizer.encode(text, max(2, model.max_len))
            if len(ids) < 2:
                continue
            x = torch.tensor([ids[:model.max_len]], dtype=torch.long, device=device)
            with torch.no_grad():
                mix, _, _, _ = model.next_token_distribution(x, alpha=1.0, temperature=1.0)
            ent = float((-mix * torch.log(mix + 1e-8)).sum(dim=-1).mean().item())
        except Exception:
            continue
        entropies.append(ent)
        t_labels.append(float(d.get("bilateral_t", 0.5) or 0.5))
        f_labels.append(float(d.get("bilateral_f", 0.5) or 0.5))

    n = len(entropies)
    if n < 4:
        return {"warning": "insufficient examples", "n": n, "tokenizer": tokenizer.info()}

    def _corr(a, b):
        ma = sum(a) / len(a); mb = sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((y - mb) ** 2 for y in b)
        if va == 0 or vb == 0:
            return 0.0
        return sum((a[i] - ma) * (b[i] - mb) for i in range(len(a))) / math.sqrt(va * vb)

    return {
        "n": n,
        "mean_entropy": statistics.mean(entropies),
        "mean_t": statistics.mean(t_labels),
        "mean_f": statistics.mean(f_labels),
        "corr_entropy_t": _corr(entropies, t_labels),
        "corr_entropy_f": _corr(entropies, f_labels),
        "tokenizer": tokenizer.info(),
    }



# --- Model-generated Shadow-depth probe ------------------------------------

def _prompt_from_record(record: Mapping[str, Any], *, max_prompt_chars: int = 220) -> str:
    """Construct a short evaluation prompt from a labeled corpus record.

    The prompt strips explicit target-behavior suffixes when possible. This
    makes generated Shadow-depth probing a harder check than label-only
    preservation over source text.
    """
    text = str(record.get("prompt") or record.get("text") or "").strip()
    if not text:
        family = str(record.get("family") or record.get("paradox_family") or "unknown")
        probe = str(record.get("probe_type") or "profile")
        text = f"UAP family={family}; probe={probe}. Continue with the appropriate paraconsistent token profile."
    cut_points = [
        " Continue without", " Transfer the contradiction", " Explain the local/global",
        " Identify the missing", " Mark the high-residue", " Do not collapse",
    ]
    for cp in cut_points:
        idx = text.find(cp)
        if idx > 40:
            text = text[:idx].rstrip()
            break
    return text[:max_prompt_chars].rstrip() + "\nContinuation:"


def model_shadow_depth_probe(
    model: ShadowTokenCore,
    shard_dir: str | Path,
    *,
    val_shards: Optional[List[Path]] = None,
    val_fraction: float = 0.1,
    max_examples: int = 0,
    max_gen_tokens: int = 80,
    temperature: float = 0.0,
    device: str = "cpu",
    seed: int = 1234,
    tokenizer=None,
) -> Dict[str, Any]:
    """Score model-generated continuations against held-out UAP profiles.

    This separates learned/generated behavior from the label/provenance
    validation performed by ``training.shadow_depth_eval.evaluate_paths``.
    Keep ``max_examples`` small for CPU runs.
    """
    tokenizer = _tok(tokenizer)
    shard_dir = Path(shard_dir)
    if val_shards is None:
        _, val_shards = split_train_val(shard_dir, val_fraction=val_fraction, seed=seed)
    if max_examples <= 0:
        return {
            "enabled": False,
            "reason": "set max_examples_shadow_model > 0 to run generated Shadow-depth probing",
            "metric_provenance": "not_run",
        }
    if not val_shards:
        return {"enabled": True, "warning": "no validation shards", "n_records": 0, "tokenizer": tokenizer.info()}

    from tovah_v14.training.shadow_depth_eval import evaluate_records
    from tovah_v14.training.loop_stability import enrich_generation_record, mean_loop_stability

    rng = random.Random(seed)
    rows = list(_iter_jsonl(val_shards, max_examples=max_examples * 4))
    if not rows:
        return {"enabled": True, "warning": "no validation records", "n_records": 0, "tokenizer": tokenizer.info()}
    rng.shuffle(rows)
    rows = rows[:max_examples]

    generated_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        prompt = _prompt_from_record(row)
        generation = gen_sample(
            model,
            prompt,
            max_tokens=max_gen_tokens,
            temperature=temperature,
            device=device,
            seed=seed + i,
            tokenizer=tokenizer,
        )
        rec = {
            **dict(row),
            "prompt": prompt,
            "generation": generation,
            "model_output": generation,
            "eval_generation_source": "model_generated",
        }
        generated_rows.append(dict(enrich_generation_record(rec)))
    result = evaluate_records(generated_rows)
    result["loop_stability_v14_3_3"] = mean_loop_stability([r.get("generation", "") for r in generated_rows])
    result["enabled"] = True
    result["metric_provenance"] = "model_generated_continuations"
    result["max_examples"] = max_examples
    result["max_gen_tokens"] = max_gen_tokens
    result["temperature"] = temperature
    result["tokenizer"] = tokenizer.info()
    return result

# --- Full eval composite ---------------------------------------------------

def run_full_eval(
    model: ShadowTokenCore,
    shard_dir: str | Path,
    *,
    val_fraction: float = 0.1,
    max_examples_ppl: int = 500,
    max_examples_acc: int = 500,
    max_examples_calib: int = 200,
    gen_prompts: Optional[List[str]] = None,
    n_gen_samples: int = 2,
    max_examples_shadow_model: int = 0,
    shadow_model_max_gen_tokens: int = 80,
    shadow_model_temperature: float = 0.0,
    device: str = "cpu",
    seed: int = 1234,
    tokenizer=None,
) -> Dict[str, Any]:
    """Run every eval probe and return one combined dict."""
    tokenizer = _tok(tokenizer)
    shard_dir = Path(shard_dir)
    _, val_shards = split_train_val(shard_dir, val_fraction=val_fraction, seed=seed)

    out: Dict[str, Any] = {"shard_dir": str(shard_dir),
                            "val_shards": [str(s) for s in val_shards],
                            "tokenizer": tokenizer.info()}
    out["perplexity"] = held_out_perplexity(
        model, shard_dir, val_shards=val_shards,
        max_examples=max_examples_ppl, device=device, tokenizer=tokenizer,
    )
    out["top1_accuracy"] = token_top1_accuracy(
        model, shard_dir, val_shards=val_shards,
        max_examples=max_examples_acc, device=device, tokenizer=tokenizer,
    )
    out["calibration"] = bilateral_calibration(
        model, shard_dir, val_shards=val_shards,
        max_examples=max_examples_calib, device=device, tokenizer=tokenizer,
    )
    samples = []
    if gen_prompts is None:
        gen_prompts = [
            "research finding: ",
            "tool_use result: ",
            "the result was ",
        ]
    for prompt in gen_prompts[:n_gen_samples]:
        samples.append({
            "prompt": prompt,
            "generation": gen_sample(model, prompt, max_tokens=120,
                                     temperature=0.8, device=device, seed=seed,
                                     tokenizer=tokenizer),
        })
    out["gen_samples"] = samples
    try:
        out["shadow_depth"] = shadow_depth_evaluate_paths(val_shards or [Path(shard_dir)])
    except Exception as exc:
        out["shadow_depth"] = {"warning": f"shadow-depth eval unavailable: {exc}"}
    try:
        out["model_shadow_depth"] = model_shadow_depth_probe(
            model,
            shard_dir,
            val_shards=val_shards,
            max_examples=max_examples_shadow_model,
            max_gen_tokens=shadow_model_max_gen_tokens,
            temperature=shadow_model_temperature,
            device=device,
            seed=seed,
            tokenizer=tokenizer,
        )
    except Exception as exc:
        out["model_shadow_depth"] = {"warning": f"model-generated Shadow-depth eval unavailable: {exc}"}
    return out

# --- CLI entrypoint --------------------------------------------------------

def _load_eval_model(
    checkpoint: str | Path,
    *,
    tokenizer,
    profile_name: str = "auto",
    device: str = "cpu",
    strict: bool = True,
) -> tuple[ShadowTokenCore, Dict[str, Any]]:
    """Load a ShadowTokenCore from a TOVAH training checkpoint."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    metadata = dict(payload.get("metadata", {}) or {}) if isinstance(payload, Mapping) else {}
    if profile_name == "auto":
        profile_name = str(metadata.get("profile_name") or metadata.get("profile") or "heavy")
    if profile_name not in MODEL_PROFILES:
        raise KeyError(f"unknown model profile {profile_name!r}; available={sorted(MODEL_PROFILES)}")
    model = ShadowTokenCore(vocab_size=tokenizer.vocab_size, **MODEL_PROFILES[profile_name]).to(device)
    payload = load_training_checkpoint(
        checkpoint,
        model,
        optimizer=None,
        map_location=device,
        strict=strict,
        restore_rng=False,
    )
    model.eval()
    return model, payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run TOVAH validation, including v14.3.3 Shadow-depth and loop-stability checks.")
    ap.add_argument("--checkpoint", required=True, type=Path, help="Training checkpoint .pt file or checkpoint directory.")
    ap.add_argument("--shard-dir", required=True, type=Path, help="Directory containing JSONL corpus shards.")
    ap.add_argument("--tokenizer", default="byte", help="Tokenizer spec: 'byte' or path to tokenizer.json.")
    ap.add_argument("--out", type=Path, default=None, help="Write combined eval JSON to this path.")
    ap.add_argument("--profile", default="auto", help="Model profile to instantiate, or 'auto' from checkpoint metadata.")
    ap.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-examples-ppl", type=int, default=500)
    ap.add_argument("--max-examples-acc", type=int, default=500)
    ap.add_argument("--max-examples-calib", type=int, default=200)
    ap.add_argument("--n-gen-samples", type=int, default=0, help="Qualitative generation samples to include.")
    ap.add_argument("--max-examples-shadow-model", type=int, default=0,
                    help="Optional generated-continuation Shadow-depth probe count. Use small values on CPU, e.g. 8-32.")
    ap.add_argument("--shadow-model-max-gen-tokens", type=int, default=80)
    ap.add_argument("--shadow-model-temperature", type=float, default=0.0)
    ap.add_argument("--non-strict", action="store_true", help="Load checkpoint with strict=False for compatibility triage.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tokenizer = load_tokenizer(str(args.tokenizer))
    model, payload = _load_eval_model(
        args.checkpoint,
        tokenizer=tokenizer,
        profile_name=args.profile,
        device=args.device,
        strict=not args.non_strict,
    )
    result = run_full_eval(
        model,
        args.shard_dir,
        val_fraction=args.val_fraction,
        max_examples_ppl=args.max_examples_ppl,
        max_examples_acc=args.max_examples_acc,
        max_examples_calib=args.max_examples_calib,
        n_gen_samples=args.n_gen_samples,
        max_examples_shadow_model=args.max_examples_shadow_model,
        shadow_model_max_gen_tokens=args.shadow_model_max_gen_tokens,
        shadow_model_temperature=args.shadow_model_temperature,
        device=args.device,
        seed=args.seed,
        tokenizer=tokenizer,
    )
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_metadata"] = dict(payload.get("metadata", {}) or {}) if isinstance(payload, Mapping) else {}
    result["eval_schema_version"] = "tovah-full-eval-v14.3.3"
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

