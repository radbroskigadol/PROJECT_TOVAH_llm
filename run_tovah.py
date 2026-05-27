#!/usr/bin/env python3
"""
TOVAH v14.3.2 — Run file.

Launch:
  python run_tovah.py          (from parent directory of tovah_v14/)
  python tovah_v14/run_tovah.py (from anywhere — self-resolving)
  python -m tovah_v14.run_tovah (if tovah_v14 is on PYTHONPATH)
"""
import argparse
import json
import logging
import os
import sys
import time

# --- Path resolution: ensure tovah_v14 package is importable ---
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# --- Thread cap BEFORE any torch import to prevent startup hang ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - TOVAH - %(levelname)s - %(message)s")


def build_api():
    """Build advisor API clients from environment."""
    from typing import Callable, Dict
    api: Dict[str, Callable[[str], str]] = {}

    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None

    grok_key = os.getenv("GROK_API_KEY", "").strip()
    if grok_key and OpenAI is not None:
        from tovah_v14.config.settings import SHADOWHOTT_SYSTEM_CONTEXT
        client = OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")

        def _mk(model: str) -> Callable[[str], str]:
            def call(prompt: str) -> str:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SHADOWHOTT_SYSTEM_CONTEXT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.5,
                        max_tokens=2200,
                    )
                    content = getattr(getattr(r.choices[0], "message", None), "content", None) if r.choices else None
                    return str(content).strip() if content else ""
                except BaseException as e:
                    logging.error(f"{model}: {type(e).__name__}: {e}")
                    return ""
            return call

        api["grok_code_api"] = _mk("grok-code-fast-1")
        api["grok_reasoning_api"] = _mk("grok-4-1-fast-reasoning")
        api["grok_fast_reasoning_api"] = _mk("grok-4-fast-reasoning")
        api["grok_non_reasoning_api"] = _mk("grok-4-fast-non-reasoning")

    return api


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run TOVAH or launch corpus pretraining.")
    parser.add_argument("--pretrain", action="store_true", help="run training/pretrain.py instead of the live kernel loop")
    parser.add_argument("--shard-dir", default="", help="JSONL shard directory; defaults to tovah_corpus/stream")
    parser.add_argument("--profile", dest="profile_name", default=os.getenv("TOVAH_PROFILE", "standard"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="microbatches per optimizer update")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--snapshot-every-steps", type=int, default=None)
    parser.add_argument("--save-path", default="")
    parser.add_argument("--metrics-path", default="", help="optional JSONL metrics output path for scale runs")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--tokenizer", dest="tokenizer_spec", default=os.getenv("TOVAH_TOKENIZER", "auto"), help="byte, auto, bpe, auto-bpe, or path to tokenizer.json")
    parser.add_argument("--train-bpe-if-missing", action="store_true", help="train a BPE tokenizer from --shard-dir when no tokenizer.json exists")
    parser.add_argument("--bpe-vocab-size", type=int, default=8192)
    parser.add_argument("--bpe-save-path", default="")
    parser.add_argument("--optimizer", dest="optimizer_kind", choices=["shadow", "uap_shadow", "adamw", "muon", "hybrid"], default=None)
    parser.add_argument("--uap-classical-floor", type=float, default=0.15, help="minimum AdamW-classicalized gradient floor inside UAP ShadowOptimizer")
    parser.add_argument("--uap-classical-ceiling", type=float, default=0.85, help="maximum AdamW-classicalized influence inside UAP ShadowOptimizer")
    parser.add_argument("--uap-geometry-lr", type=float, default=0.01, help="slow geometry-gate learning rate for UAP classicalization/shadow split")
    parser.add_argument("--uap-weight-decay", type=float, default=None, help="decoupled weight decay for UAP ShadowOptimizer; default 0.1")
    parser.add_argument("--uap-max-update-rms", type=float, default=1.0, help="size-invariant RMS cap for UAP updates; default 1.0 preserves AdamW-like sign-step scale")
    parser.add_argument("--uap-trust-clip", type=float, default=0.0, help="optional symmetric relative trust-ratio clip; 0 disables it")
    parser.add_argument("--hybrid-gate-lr", type=float, default=0.02, help="AdamW-vs-UAPShadow hybrid gate learning rate")
    parser.add_argument("--hybrid-min-adamw-weight", type=float, default=0.15, help="minimum AdamW weight in the external hybrid optimizer")
    parser.add_argument("--uap-aux-weight", type=float, default=0.05, help="v14.3.2 weight for UAP token-profile auxiliary ontology losses")
    parser.add_argument("--uap-loop-penalty-weight", type=float, default=0.0, help="deprecated/no-op; retained for compatibility")
    parser.add_argument("--uap-sheaf-weight", type=float, default=0.0, help="v14.3.5 optional differentiable sheaf-obstruction auxiliary weight")
    parser.add_argument("--bilateral-mode", choices=["shared", "dual"], default="shared")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--use-ddp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--length-stratified", action="store_true")
    parser.add_argument("--class-filter", default="", help="comma-separated paraconsistent classes, e.g. A,K,G")
    parser.add_argument("--kind-filter", default="", help="comma-separated corpus kind filters")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--frontier-semantic-mode", choices=["auto", "hidden", "logits"], default="auto")
    parser.add_argument("--fsdp-mixed-precision", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--save-sharded", action="store_true")
    parser.add_argument("--estimate-frontier-memory", action="store_true", help="print frontier memory estimate and exit before model allocation")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    logging.info("Starting TOVAH v14.3.3...")

    from tovah_v14.config.paths import ensure_directories
    ensure_directories()

    if args.pretrain:
        from tovah_v14.config.paths import CORPUS_STREAM_DIR
        from tovah_v14.training.pretrain import pretrain
        shard_dir = args.shard_dir or str(CORPUS_STREAM_DIR)
        summary = pretrain(
            shard_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            max_steps=args.max_steps,
            save_path=args.save_path or None,
            metrics_path=args.metrics_path or None,
            device=args.device,
            dtype=args.dtype,
            profile_name=args.profile_name,
            optimizer_kind=args.optimizer_kind,
            tokenizer_spec=args.tokenizer_spec,
            train_bpe_if_missing=args.train_bpe_if_missing,
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_save_path=args.bpe_save_path or None,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            val_fraction=args.val_fraction,
            eval_every_steps=args.eval_every_steps,
            snapshot_every_steps=args.snapshot_every_steps,
            class_filter=[x.strip() for x in args.class_filter.split(',') if x.strip()] or None,
            kind_filter=[x.strip() for x in args.kind_filter.split(',') if x.strip()] or None,
            length_stratified=args.length_stratified,
            bilateral_mode=args.bilateral_mode,
            gradient_checkpointing=args.gradient_checkpointing,
            use_fsdp=args.use_fsdp,
            use_ddp=args.use_ddp,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            log_every=args.log_every,
            frontier_semantic_mode=args.frontier_semantic_mode,
            fsdp_mixed_precision=args.fsdp_mixed_precision,
            uap_classical_floor=args.uap_classical_floor,
            uap_classical_ceiling=args.uap_classical_ceiling,
            uap_geometry_lr=args.uap_geometry_lr,
            uap_weight_decay=args.uap_weight_decay,
            uap_max_update_rms=args.uap_max_update_rms,
            uap_trust_clip=args.uap_trust_clip,
            hybrid_gate_lr=args.hybrid_gate_lr,
            hybrid_min_adamw_weight=args.hybrid_min_adamw_weight,
            uap_aux_weight=args.uap_aux_weight,
            uap_loop_penalty_weight=args.uap_loop_penalty_weight,
            uap_sheaf_weight=args.uap_sheaf_weight,
            resume_from=args.resume_from or None,
            save_sharded=args.save_sharded,
            estimate_memory_only=args.estimate_frontier_memory,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    from tovah_v14.kernel.kernel import ProtozoanKernel
    kernel = ProtozoanKernel(api=build_api(), is_original=True)
    ecology = kernel.get_kernel_ecology_summary()
    logging.info(
        "Kernel ecology booted | mode=%s | hub=%s | subkernels=%s | packets=%s",
        ecology.get("boot_mode"),
        ecology.get("hub_present"),
        ecology.get("subkernel_count"),
        ecology.get("packet_log_entries"),
    )

    # --- Preflight: verify subsystem readiness before entering loop ---
    from tovah_v14.kernel.preflight import run_preflight
    preflight = run_preflight(kernel)
    if not preflight.ok:
        logging.error(f"PREFLIGHT FAILED: {preflight.errors}")
        logging.error("Refusing to enter run loop. Fix errors and restart.")
        sys.exit(1)
    logging.info("Preflight passed. Entering main loop.")

    while True:
        try:
            kernel.run_loop(3600)
        except KeyboardInterrupt:
            logging.info("Shutdown.")
            break
        except Exception as e:
            logging.error(f"Crash: {e}. Reboot 30s...")
            time.sleep(30)


if __name__ == "__main__":
    main()
