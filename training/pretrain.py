"""
TOVAH v14.3.3 training/pretrain.py — Production pretraining entry point.

AUDIT FIXES applied (per audit P0/P1):
  P0-5  — held-out eval per epoch (perplexity, accuracy, calibration)
  P0-5  — divergence detection + automatic rollback
  P1-1  — tokenizer abstraction (byte or BPE)
  P1-2  — PyTorch DataLoader with num_workers
  P1-3  — bf16 / fp16 mixed precision (autocast)
  P1-4  — LR warmup + cosine decay (via ShadowOptimizer.set_schedule)
  P3    — gradient accumulation + NaN/Inf rollback
  P3    — periodic checkpointing

Public:
  pretrain(shard_dir, *, model, optimizer, tokenizer, epochs, batch_size,
           grad_accum_steps, max_examples, dtype, eval_every_steps,
           save_path, log_every) -> Dict[str, Any]
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.neural.optimizer import ShadowOptimizer
from tovah_v14.neural.adamw import make_optimizer
from tovah_v14.neural.scaling import FRONTIER_PROFILES, make_scalable_model, ScalableBilateralCore, estimate_frontier_memory
from tovah_v14.neural import distributed as dist_utils
from tovah_v14.neural.checkpointing import save_training_checkpoint, load_training_checkpoint
from tovah_v14.neural.training import (
    semantic_rank_nullity_loss,
    metadata_weighted_semantic_loss,
    lane_routing_loss,
    lane_semantic_matching_loss,
    phase_from_semantic_state,
)
from tovah_v14.config.constants import MODEL_PROFILES
from tovah_v14.training.dataset import CorpusShardDataset, build_collate_fn
from tovah_v14.training.tokenizer import load_tokenizer, train_bpe
from tovah_v14.training.eval import (
    run_full_eval, detect_divergence, split_train_val,
)
from tovah_v14.training.metrics import ScaleMetricLogger, gpu_memory_gb
from tovah_v14.training.uap_aux_losses import semantic_outputs_from_supports, uap_auxiliary_losses
from tovah_v14.training.loop_stability import repetition_penalty_from_logits
from tovah_v14.training.sheaf_regularizer import sequence_sheaf_obstruction_loss


def _unwrap_model(model: Any) -> Any:
    """Return the underlying module for DDP/FSDP-style wrappers."""
    return getattr(model, "module", model)


def _model_attr(model: Any, name: str, default: Any = None) -> Any:
    base = _unwrap_model(model)
    return getattr(base, name, getattr(model, name, default))


def _make_default_model_optimizer(
    device: str,
    profile_name: str,
    vocab_size: int = 256,
    *,
    optimizer_kind: Optional[str] = None,
    tokenizer_spec: Optional[str] = None,
    train_bpe_if_missing: bool = False,
    bpe_vocab_size: int = 8192,
    bpe_save_path: Optional[str | Path] = None,
    bilateral_mode: str = "shared",
    gradient_checkpointing: bool = False,
    tied_embeddings: bool = True,
    use_fsdp: bool = False,
    use_ddp: bool = False,
    fsdp_mixed_precision: Optional[str] = None,
    uap_classical_floor: float = 0.15,
    uap_classical_ceiling: float = 0.85,
    uap_geometry_lr: float = 0.01,
    uap_weight_decay: Optional[float] = None,
    uap_max_update_rms: float = 1.0,
    uap_trust_clip: float = 0.0,
    hybrid_gate_lr: float = 0.02,
    hybrid_min_adamw_weight: float = 0.15,
) -> Tuple[Any, Any]:
    """Build the requested model/optimizer pair.

    Classic profiles (debug/standard/heavy/large) use ShadowTokenCore +
    ShadowOptimizer. Frontier profiles use ScalableBilateralCore + Muon by
    default and optionally wrap the model in DDP/FSDP when launched under
    torchrun.
    """
    profile_name = str(profile_name or "standard").strip().lower()
    is_frontier = profile_name in FRONTIER_PROFILES

    if is_frontier:
        local_rank = None
        if use_fsdp or use_ddp or dist_utils.is_distributed_available():
            local_rank = dist_utils.init_distributed()
            if local_rank is not None and torch.cuda.is_available():
                device = f"cuda:{local_rank}"
        model = make_scalable_model(
            profile_name,
            vocab_size=vocab_size,
            bilateral_mode=bilateral_mode,
            gradient_checkpointing=gradient_checkpointing,
            tied_embeddings=tied_embeddings,
        ).to(device)
        if local_rank is not None:
            if use_fsdp:
                model = dist_utils.wrap_fsdp(model, mixed_precision=fsdp_mixed_precision)
            elif use_ddp or dist_utils.world_size() > 1:
                model = dist_utils.wrap_ddp(model)
        optimizer = make_optimizer(
            model.parameters(),
            kind=optimizer_kind or "muon",
            base_lr=3e-4,
            uap_classical_floor=uap_classical_floor,
            uap_classical_ceiling=uap_classical_ceiling,
            uap_geometry_lr=uap_geometry_lr,
            uap_weight_decay=uap_weight_decay,
            uap_max_update_rms=uap_max_update_rms,
            uap_trust_clip=uap_trust_clip,
            hybrid_gate_lr=hybrid_gate_lr,
            hybrid_min_adamw_weight=hybrid_min_adamw_weight,
        )
        return model, optimizer

    if profile_name not in MODEL_PROFILES:
        valid = sorted(list(MODEL_PROFILES) + list(FRONTIER_PROFILES))
        raise ValueError(
            f"unknown profile {profile_name!r}; valid profiles are: {valid}. "
            "This is intentionally strict so accidental names such as 'tiny' "
            "cannot silently fall back to 'standard'."
        )
    profile = dict(MODEL_PROFILES[profile_name])
    model = ShadowTokenCore(vocab_size=vocab_size, **profile).to(device)
    optimizer = make_optimizer(
        model.parameters(),
        kind=optimizer_kind or "shadow",
        base_lr=3e-4,
        uap_classical_floor=uap_classical_floor,
        uap_classical_ceiling=uap_classical_ceiling,
        uap_geometry_lr=uap_geometry_lr,
        uap_weight_decay=uap_weight_decay,
        uap_max_update_rms=uap_max_update_rms,
        uap_trust_clip=uap_trust_clip,
        hybrid_gate_lr=hybrid_gate_lr,
        hybrid_min_adamw_weight=hybrid_min_adamw_weight,
    )
    return model, optimizer


def _phase_from_batch_state(tp: torch.Tensor, fp: torch.Tensor,
                            batch: Dict[str, Any], *,
                            con_budget: float, gap_budget: float,
                            device: str) -> str:
    """Metadata-aware, mean-glut/gap phase selector.

    v14.2.2 replaces the shape-dependent sum(K)>5 and h==0 rule. The
    phase now responds to true high-glut/high-gap corpus metadata as well as
    the model's mean bilateral state.
    """
    bt = batch.get("bilateral_t")
    bf = batch.get("bilateral_f")
    if bt is not None and bf is not None:
        bt = bt.to(device=device)
        bf = bf.to(device=device)
    return phase_from_semantic_state(
        tp, fp, bilateral_t=bt, bilateral_f=bf,
        con_budget=con_budget, gap_budget=gap_budget,
    )


class _NullCtx:
    """no-op autocast replacement when AMP is disabled."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _supports_hidden_semantics(model: Any) -> bool:
    """True when the model exposes compact B×L×1 semantic heads."""
    base = _unwrap_model(model)
    return hasattr(base, "semantic_T") and hasattr(base, "semantic_F")


def _forward_for_pretrain(model: Any, x: torch.Tensor, *,
                          frontier_semantic_mode: str = "auto"):
    """Forward helper that chooses frontier semantic auxiliary tensors.

    ``auto`` selects hidden-state semantic supports for ScalableBilateralCore,
    avoiding full-vocab F logits and full-vocab sigmoid(T/F) auxiliary tensors.
    Classic models fall back to logits for compatibility.
    """
    mode = str(frontier_semantic_mode or "auto").lower()
    if mode not in {"auto", "hidden", "logits"}:
        raise ValueError("frontier_semantic_mode must be auto, hidden, or logits")
    use_hidden = (mode in {"auto", "hidden"}) and _supports_hidden_semantics(model)
    if use_hidden:
        try:
            tl, fl, gl, tp, fp_ = model(
                x,
                return_semantic_supports=True,
                semantic_aux_mode="hidden",
                skip_f_logits=True,
            )
            return tl, fl, gl, tp, fp_, "hidden"
        except TypeError:
            if mode == "hidden":
                raise
            # Wrapper or older model did not accept kwargs; fall through.
    tl, fl, gl = model(x)
    return tl, fl, gl, torch.sigmoid(tl), torch.sigmoid(fl), "logits"


def _compute_pretrain_loss(
    model: Any,
    batch: Dict[str, Any],
    *,
    con_budget: float,
    gap_budget: float,
    lambda_budget: float,
    device: str,
    dtype: torch.dtype,
    frontier_semantic_mode: str = "auto",
    uap_aux_weight: float = 0.0,
    uap_loop_penalty_weight: float = 0.0,
    uap_sheaf_weight: float = 0.0,
) -> Tuple[torch.Tensor, float, str, int]:
    """Forward pass and loss construction for one microbatch.

    Returns ``(loss_tensor, loss_float, phase, token_count)``. Optimizer steps
    are intentionally not performed here so the caller can do real gradient
    accumulation across multiple microbatches.
    """
    x = batch["input_ids"].to(device, non_blocking=True)
    y = batch["target_ids"].to(device, non_blocking=True)
    m = batch["attention_mask"].to(device, non_blocking=True)

    use_amp = (dtype in (torch.bfloat16, torch.float16)) and device.startswith("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else _NullCtx()
    with ctx:
        tl, fl, gl, tp, fp_, semantic_mode_used = _forward_for_pretrain(
            model, x, frontier_semantic_mode=frontier_semantic_mode,
        )
        logp = F.log_softmax(tl, dim=-1)
        true_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        mask_sum = m.sum().clamp_min(1.0)
        task_loss = -(true_logp * m).sum() / mask_sum
        bt = batch.get("bilateral_t")
        bf = batch.get("bilateral_f")
        has_metadata = bt is not None and bf is not None
        sem_loss, _, _, _, _ = semantic_rank_nullity_loss(
            tp, fp_, (0.0 if has_metadata else con_budget), (0.0 if has_metadata else gap_budget),
        )
        ge = (-F.softmax(gl, dim=-1) * F.log_softmax(gl, dim=-1)).sum(dim=-1).mean()
        # v14.3.5: when per-example bilateral metadata is present, the global
        # K/G prior is disabled so it does not fight A/B/K/G labels.
        total = task_loss + (0.0 if has_metadata else 0.3) * sem_loss - 0.01 * ge

        if has_metadata:
            bt = bt.to(device=device, dtype=tp.dtype, non_blocking=True)
            bf = bf.to(device=device, dtype=tp.dtype, non_blocking=True)
            meta_sem, _meta_stats = metadata_weighted_semantic_loss(
                tp, fp_, bt, bf, attention_mask=m,
                con_budget=con_budget, gap_budget=gap_budget,
            )
            lane_loss = lane_routing_loss(gl, bt, bf, attention_mask=m)
            lane_match, _lane_match_stats = lane_semantic_matching_loss(
                tp, fp_, gl, bt, bf, attention_mask=m,
            )
            total = total + 0.15 * meta_sem + 0.05 * lane_loss + 0.10 * lane_match
            if uap_sheaf_weight:
                sheaf_loss, _sheaf_stats = sequence_sheaf_obstruction_loss(
                    tp, fp_, attention_mask=m, bilateral_t=bt, bilateral_f=bf,
                )
                total = total + float(uap_sheaf_weight) * sheaf_loss
        elif uap_sheaf_weight:
            sheaf_loss, _sheaf_stats = sequence_sheaf_obstruction_loss(tp, fp_, attention_mask=m)
            total = total + float(uap_sheaf_weight) * sheaf_loss

        # v14.3.4: do NOT add a logits-concentration penalty during pretraining.
        # It is an anti-confidence regularizer that fights CE. Loop control is
        # handled by decode/eval diagnostics and, later, sequence-level rewards.
        if uap_loop_penalty_weight and not getattr(_compute_pretrain_loss, "_loop_penalty_warned", False):
            logging.warning("--uap-loop-penalty-weight is deprecated/no-op in v14.3.4; use decode-time repetition controls")
            setattr(_compute_pretrain_loss, "_loop_penalty_warned", True)

        # v14.3.2/v14.3.3 Shadow-depth objective scaffold.  This is deliberately an
        # ontology-preservation auxiliary term, not an AdamW-vs-Shadow race.
        if uap_aux_weight and batch.get("uap_profile_targets"):
            try:
                aux_outputs = semantic_outputs_from_supports(tp, fp_, attention_mask=m)
                aux_losses = uap_auxiliary_losses(aux_outputs, batch["uap_profile_targets"])
                total = total + float(uap_aux_weight) * aux_losses["uap_aux_total"]
            except Exception as exc:
                logging.warning("uap auxiliary loss skipped: %s", exc)

    phase = _phase_from_batch_state(
        tp.detach(), fp_.detach(), batch,
        con_budget=con_budget, gap_budget=gap_budget, device=device,
    )
    return total, float(total.detach().item()), phase, int(m.sum().item())


def _train_one_batch(
    model: Any,
    optimizer: Any,
    batch: Dict[str, Any],
    *,
    con_budget: float,
    gap_budget: float,
    lambda_budget: float,
    device: str,
    dtype: torch.dtype,
    frontier_semantic_mode: str = "auto",
) -> Tuple[float, str]:
    """Backward-compatible one-microbatch update used by legacy tests."""
    optimizer.zero_grad()
    loss, loss_val, phase, _tokens = _compute_pretrain_loss(
        model, batch,
        con_budget=con_budget, gap_budget=gap_budget,
        lambda_budget=lambda_budget, device=device, dtype=dtype,
        frontier_semantic_mode=frontier_semantic_mode,
        uap_aux_weight=0.0,
        uap_loop_penalty_weight=0.0,
    )
    loss.backward()
    if hasattr(optimizer, "step_grads"):
        optimizer.step_grads(phase=phase, loss_value=loss_val)
    else:
        optimizer.step(loss, phase=phase)
    return loss_val, phase


def pretrain(
    shard_dir: str | Path,
    *,
    model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    tokenizer=None,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    max_examples: Optional[int] = None,
    max_steps: Optional[int] = None,
    class_filter: Optional[List[str]] = None,
    kind_filter: Optional[List[str]] = None,
    length_stratified: bool = False,
    con_budget: float = 0.12,
    gap_budget: float = 0.20,
    lambda_budget: float = 0.05,
    dtype: str = "fp32",
    warmup_steps: Optional[int] = None,
    min_lr_ratio: float = 0.1,
    val_fraction: float = 0.1,
    eval_every_steps: Optional[int] = None,
    snapshot_every_steps: Optional[int] = None,
    abort_on_divergence: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    save_path: Optional[str | Path] = None,
    metrics_path: Optional[str | Path] = None,
    log_every: int = 50,
    device: str = "cpu",
    profile_name: str = "standard",
    seed: int = 1234,
    optimizer_kind: Optional[str] = None,
    tokenizer_spec: Optional[str] = None,
    train_bpe_if_missing: bool = False,
    bpe_vocab_size: int = 8192,
    bpe_save_path: Optional[str | Path] = None,
    bilateral_mode: str = "shared",
    gradient_checkpointing: bool = False,
    tied_embeddings: bool = True,
    use_fsdp: bool = False,
    use_ddp: bool = False,
    fsdp_mixed_precision: Optional[str] = None,
    frontier_semantic_mode: str = "auto",
    uap_classical_floor: float = 0.15,
    uap_classical_ceiling: float = 0.85,
    uap_geometry_lr: float = 0.01,
    uap_weight_decay: Optional[float] = None,
    uap_max_update_rms: float = 1.0,
    uap_trust_clip: float = 0.0,
    hybrid_gate_lr: float = 0.02,
    hybrid_min_adamw_weight: float = 0.15,
    uap_aux_weight: float = 0.05,
    uap_loop_penalty_weight: float = 0.0,
    uap_sheaf_weight: float = 0.0,
    resume_from: Optional[str | Path] = None,
    save_sharded: bool = False,
    estimate_memory_only: bool = False,
) -> Dict[str, Any]:
    """Run batched pretraining over a JSONL corpus directory.

    Returns a structured summary dict.
    """
    shard_dir = Path(shard_dir)

    # Tokenizer. BPE is the preferred/main path when a tokenizer file exists
    # or when --train-bpe-if-missing is requested; byte remains the safe
    # fallback for fresh checkouts and tests without the optional dependency.
    if tokenizer is None:
        tok_spec = tokenizer_spec or os.environ.get("TOVAH_TOKENIZER", "auto")
        if str(tok_spec).lower() in {"auto", "bpe", "auto-bpe"}:
            candidates = []
            if bpe_save_path:
                candidates.append(Path(bpe_save_path))
            candidates.append(shard_dir / "tokenizer.json")
            candidates.append(Path("tovah_corpus") / "tokenizer.json")
            chosen = next((c for c in candidates if c.exists()), None)
            if chosen is not None:
                tokenizer = load_tokenizer(str(chosen))
            elif train_bpe_if_missing:
                out_path = Path(bpe_save_path) if bpe_save_path else (shard_dir / "tokenizer.json")
                tokenizer = train_bpe(shard_dir, vocab_size=bpe_vocab_size, save_path=out_path)
            else:
                logging.warning(
                    "BPE tokenizer requested but no tokenizer.json was found; "
                    "falling back to byte. Use --train-bpe-if-missing or "
                    "--bpe-save-path to make BPE the active path."
                )
                tokenizer = load_tokenizer("byte")
        else:
            tokenizer = load_tokenizer(str(tok_spec))
    vocab_size = tokenizer.vocab_size

    memory_estimate = None
    if str(profile_name).lower() in FRONTIER_PROFILES:
        memory_estimate = estimate_frontier_memory(
            str(profile_name).lower(),
            vocab_size=vocab_size,
            bilateral_mode=bilateral_mode,
            tied_embeddings=tied_embeddings,
            dtype=dtype,
            batch_size=batch_size,
            seq_len=None,
            world_size=max(1, dist_utils.world_size()),
            use_fsdp=use_fsdp,
            optimizer=optimizer_kind or "adamw",
            gradient_checkpointing=gradient_checkpointing,
        )
        if estimate_memory_only:
            return {
                "profile_name": profile_name,
                "tokenizer": tokenizer.info(),
                "vocab_size": vocab_size,
                "memory_estimate": memory_estimate,
                "estimated_only": True,
            }

    if not shard_dir.exists():
        raise FileNotFoundError(f"shard_dir does not exist: {shard_dir}")

    # Model & optimizer.
    if model is None or optimizer is None:
        m_built, o_built = _make_default_model_optimizer(
            device, profile_name, vocab_size=vocab_size,
            optimizer_kind=optimizer_kind,
            bilateral_mode=bilateral_mode,
            gradient_checkpointing=gradient_checkpointing,
            tied_embeddings=tied_embeddings,
            use_fsdp=use_fsdp,
            use_ddp=use_ddp,
            fsdp_mixed_precision=fsdp_mixed_precision,
            uap_classical_floor=uap_classical_floor,
            uap_classical_ceiling=uap_classical_ceiling,
            uap_geometry_lr=uap_geometry_lr,
            uap_weight_decay=uap_weight_decay,
            uap_max_update_rms=uap_max_update_rms,
            uap_trust_clip=uap_trust_clip,
            hybrid_gate_lr=hybrid_gate_lr,
            hybrid_min_adamw_weight=hybrid_min_adamw_weight,
        )
        if model is None:
            model = m_built
        if optimizer is None:
            optimizer = o_built

    model_vocab_size = _model_attr(model, "vocab_size", None)
    if model_vocab_size is not None and model_vocab_size != vocab_size:
        raise ValueError(
            f"tokenizer.vocab_size ({vocab_size}) != model.vocab_size "
            f"({model_vocab_size}). Train a model with matching vocab or "
            f"use the byte tokenizer."
        )

    # If distributed initialization or CUDA placement changed the effective
    # device during model construction, follow the model's actual device for
    # batch movement and autocast decisions.
    try:
        device = str(next(_unwrap_model(model).parameters()).device)
    except Exception:
        pass

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    if torch_dtype != torch.float32 and not device.startswith("cuda"):
        logging.warning("AMP requested but device=%s; falling back to fp32.", device)
        torch_dtype = torch.float32

    resumed_payload = None
    if resume_from is not None:
        resumed_payload = load_training_checkpoint(
            resume_from, model, optimizer, map_location=device, strict=False, restore_rng=True,
        )
        logging.info(
            "pretrain: resumed checkpoint from %s at step=%s epoch=%s",
            resume_from, resumed_payload.get("step"), resumed_payload.get("epoch"),
        )

    train_shards, val_shards = split_train_val(shard_dir, val_fraction=val_fraction, seed=seed)
    if not train_shards:
        raise FileNotFoundError(f"no JSONL shards in {shard_dir}")

    class_set = set(class_filter) if class_filter else None
    kind_set = set(kind_filter) if kind_filter else None

    model_max_len = int(_model_attr(model, "max_len", 1024))

    dataset = CorpusShardDataset(
        shard_dir,
        max_len=model_max_len,
        class_filter=class_set,
        kind_filter=kind_set,
        strip_envelope_text=True,
        chunk_long_text=True,
        length_stratified=length_stratified,
        seed=seed,
        shuffle_shards=True,
    )
    # Restrict to train shards.
    dataset._list_shards = lambda: list(train_shards)  # type: ignore

    collate_fn = build_collate_fn(tokenizer, max_len=model_max_len, pad_id=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and device.startswith("cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Pre-count for LR schedule.
    n_examples_train = 0
    for shard in train_shards:
        try:
            with open(shard, "r", encoding="utf-8") as fh:
                for _ in fh:
                    n_examples_train += 1
        except Exception:
            pass
    if max_examples is not None:
        n_examples_train = min(n_examples_train, max_examples)
    steps_per_epoch_estimate = max(1, n_examples_train // max(1, batch_size))
    total_steps = min(
        max_steps if max_steps is not None else 10**9,
        steps_per_epoch_estimate * epochs,
    )
    if warmup_steps is not None and warmup_steps > 0:
        optimizer.set_schedule(
            warmup_steps=warmup_steps,
            total_steps=max(warmup_steps + 1, total_steps),
            min_lr_ratio=min_lr_ratio,
        )

    snapshots: List[Dict[str, Any]] = []
    metric_logger = ScaleMetricLogger(metrics_path) if metrics_path else None

    def _snapshot(reason: str) -> None:
        snap = {"reason": reason, "step": optimizer.t,
                "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        snapshots.append(snap)
        snapshots[:] = snapshots[-3:]

    def _rollback() -> bool:
        if not snapshots:
            return False
        snap = snapshots[-1]
        model.load_state_dict(snap["state_dict"])
        logging.warning("Rolled back to snapshot from step %d (reason=%s)",
                        snap["step"], snap["reason"])
        return True

    _snapshot("pre_pretrain")

    t_start = time.time()
    losses_per_epoch: List[List[float]] = []
    eval_history: List[Dict[str, Any]] = []
    total_steps_run = 0
    aborted_for = ""
    last_phase = ""

    logging.info(
        f"pretrain v14.3.2: dir={shard_dir} train_shards={len(train_shards)} "
        f"val_shards={len(val_shards)} batch_size={batch_size} "
        f"grad_accum={grad_accum_steps} epochs={epochs} steps_per_epoch≈{steps_per_epoch_estimate} "
        f"dtype={dtype} num_workers={num_workers} tokenizer={tokenizer.name}/{vocab_size} "
        f"profile={profile_name} optimizer={optimizer.__class__.__name__} "
        f"fsdp={use_fsdp} ddp={use_ddp} fsdp_mp={fsdp_mixed_precision} semantic_aux={frontier_semantic_mode} uap_aux_weight={uap_aux_weight} uap_loop_penalty_weight={uap_loop_penalty_weight} uap_sheaf_weight={uap_sheaf_weight}"
    )

    grad_accum_steps = max(1, int(grad_accum_steps or 1))

    for ep in range(epochs):
        ep_losses: List[float] = []
        ep_step = 0
        accum_count = 0
        accum_loss_sum = 0.0
        accum_token_sum = 0
        accum_phase = "Active Learning"
        optimizer.zero_grad()

        for batch in loader:
            if max_steps is not None and total_steps_run >= max_steps:
                break
            try:
                loss, loss_val_micro, phase, token_count = _compute_pretrain_loss(
                    model, batch,
                    con_budget=con_budget, gap_budget=gap_budget,
                    lambda_budget=lambda_budget,
                    device=device, dtype=torch_dtype,
                    frontier_semantic_mode=frontier_semantic_mode,
                    uap_aux_weight=uap_aux_weight,
                    uap_loop_penalty_weight=uap_loop_penalty_weight,
                    uap_sheaf_weight=uap_sheaf_weight,
                )
                (loss / grad_accum_steps).backward()
            except Exception as e:
                logging.error("pretrain batch failed (skipping): %s", e)
                continue

            accum_count += 1
            accum_loss_sum += loss_val_micro
            accum_token_sum += token_count
            accum_phase = phase

            if accum_count < grad_accum_steps:
                continue

            loss_val = accum_loss_sum / max(1, accum_count)
            if hasattr(optimizer, "step_grads"):
                optimizer.step_grads(phase=accum_phase, loss_value=loss_val)
            else:
                # Compatibility fallback for external optimizers with a raw
                # torch-style .step(); gradients are already accumulated.
                optimizer.step()
            optimizer.zero_grad()

            last_phase = accum_phase
            ep_losses.append(loss_val)
            ep_step += 1
            total_steps_run += 1

            if metric_logger is not None:
                metric_logger.log(
                    step=total_steps_run,
                    epoch=ep + 1,
                    loss=loss_val,
                    phase=accum_phase,
                    lr=getattr(optimizer, "last_stats", {}).get("lr"),
                    tokens=accum_token_sum,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    optimizer_mode=getattr(optimizer, "last_stats", {}).get("mode"),
                    adamw_weight=getattr(optimizer, "last_stats", {}).get("adamw_weight"),
                    shadow_weight=getattr(optimizer, "last_stats", {}).get("shadow_weight"),
                    adamw_score=getattr(optimizer, "last_stats", {}).get("adamw_score"),
                    shadow_score=getattr(optimizer, "last_stats", {}).get("shadow_score"),
                    hybrid_score_diff=getattr(optimizer, "last_stats", {}).get("hybrid_score_diff"),
                    hybrid_score_diff_ema=getattr(optimizer, "last_stats", {}).get("hybrid_score_diff_ema"),
                    hybrid_gate_advantage=getattr(optimizer, "last_stats", {}).get("hybrid_gate_advantage"),
                    hybrid_reward=getattr(optimizer, "last_stats", {}).get("hybrid_reward"),
                    hybrid_reward_ema=getattr(optimizer, "last_stats", {}).get("hybrid_reward_ema"),
                    uap_classical_weight=getattr(optimizer, "last_stats", {}).get("uap_classical_weight"),
                    uap_shadow_weight=getattr(optimizer, "last_stats", {}).get("uap_shadow_weight"),
                    uap_obstruction=getattr(optimizer, "last_stats", {}).get("uap_obstruction"),
                    uap_obstruction_ema=getattr(optimizer, "last_stats", {}).get("uap_obstruction_ema"),
                    uap_residue_mass=getattr(optimizer, "last_stats", {}).get("uap_residue_mass"),
                    uap_collapse_pressure=getattr(optimizer, "last_stats", {}).get("uap_collapse_pressure"),
                    uap_trust_ratio_mean=getattr(optimizer, "last_stats", {}).get("uap_trust_ratio_mean"),
                    uap_update_rms_mean=getattr(optimizer, "last_stats", {}).get("uap_update_rms_mean"),
                    uap_effective_scale_mean=getattr(optimizer, "last_stats", {}).get("uap_effective_scale_mean"),
                    uap_max_update_rms=getattr(optimizer, "last_stats", {}).get("uap_max_update_rms"),
                    uap_gate_advantage=getattr(optimizer, "last_stats", {}).get("uap_gate_advantage"),
                    shadow_uap_classical_weight=getattr(optimizer, "last_stats", {}).get("shadow_uap_classical_weight"),
                    shadow_uap_shadow_weight=getattr(optimizer, "last_stats", {}).get("shadow_uap_shadow_weight"),
                    shadow_uap_obstruction=getattr(optimizer, "last_stats", {}).get("shadow_uap_obstruction"),
                    shadow_uap_residue_mass=getattr(optimizer, "last_stats", {}).get("shadow_uap_residue_mass"),
                    shadow_uap_collapse_pressure=getattr(optimizer, "last_stats", {}).get("shadow_uap_collapse_pressure"),
                    **gpu_memory_gb(),
                )

            accum_count = 0
            accum_loss_sum = 0.0
            accum_token_sum = 0

            # Divergence guards.
            if not math.isfinite(loss_val):
                logging.error("pretrain: non-finite loss at step %d", total_steps_run)
                if abort_on_divergence:
                    if _rollback():
                        if hasattr(optimizer, "_state_initialized"):
                            optimizer._state_initialized = False
                        ep_losses[-1] = float("nan")
                        continue
                    aborted_for = "non_finite_loss"
                    break

            if total_steps_run > 50 and total_steps_run % 50 == 0:
                div = detect_divergence(ep_losses, window=100, blowup_ratio=10.0)
                if div.get("diverging"):
                    logging.warning("pretrain: divergence flag: %s", div)
                    if abort_on_divergence and div.get("reason") == "non_finite_loss":
                        _rollback()
                        if hasattr(optimizer, "_state_initialized"):
                            optimizer._state_initialized = False

            if log_every and total_steps_run % log_every == 0:
                window = ep_losses[-log_every:]
                avg = sum(window) / max(1, len(window))
                logging.info(
                    f"pretrain ep={ep+1}/{epochs} step={ep_step} total={total_steps_run} "
                    f"loss={loss_val:.4f} avg{log_every}={avg:.4f} phase={accum_phase} "
                    f"lr={getattr(optimizer, 'last_stats', {}).get('lr', 0):.2e}"
                )

            if eval_every_steps and total_steps_run % eval_every_steps == 0 and val_shards:
                eval_result = run_full_eval(
                    model, shard_dir, val_fraction=val_fraction,
                    max_examples_ppl=200, max_examples_acc=200,
                    max_examples_calib=100, device=device, seed=seed,
                    tokenizer=tokenizer,
                )
                eval_result["step"] = total_steps_run
                eval_history.append(eval_result)
                ppl = eval_result.get("perplexity", {}).get("perplexity")
                acc = eval_result.get("top1_accuracy", {}).get("top1_accuracy")
                logging.info(
                    "pretrain eval step=%d: ppl=%s acc=%s",
                    total_steps_run,
                    f"{ppl:.3f}" if isinstance(ppl, float) and math.isfinite(ppl) else str(ppl),
                    f"{acc:.3f}" if isinstance(acc, float) and math.isfinite(acc) else str(acc),
                )

            if snapshot_every_steps and total_steps_run % snapshot_every_steps == 0:
                _snapshot(f"step_{total_steps_run}")

        # If the epoch ended with a partial accumulation and we have not hit
        # the user-requested max_steps, apply it rather than discarding work.
        if accum_count > 0 and (max_steps is None or total_steps_run < max_steps):
            loss_val = accum_loss_sum / max(1, accum_count)
            if hasattr(optimizer, "step_grads"):
                optimizer.step_grads(phase=accum_phase, loss_value=loss_val)
            else:
                optimizer.step()
            optimizer.zero_grad()
            last_phase = accum_phase
            ep_losses.append(loss_val)
            ep_step += 1
            total_steps_run += 1
            if metric_logger is not None:
                metric_logger.log(
                    step=total_steps_run, epoch=ep + 1, loss=loss_val,
                    phase=accum_phase,
                    lr=getattr(optimizer, "last_stats", {}).get("lr"),
                    tokens=accum_token_sum, batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    optimizer_mode=getattr(optimizer, "last_stats", {}).get("mode"),
                    adamw_weight=getattr(optimizer, "last_stats", {}).get("adamw_weight"),
                    shadow_weight=getattr(optimizer, "last_stats", {}).get("shadow_weight"),
                    adamw_score=getattr(optimizer, "last_stats", {}).get("adamw_score"),
                    shadow_score=getattr(optimizer, "last_stats", {}).get("shadow_score"),
                    hybrid_score_diff=getattr(optimizer, "last_stats", {}).get("hybrid_score_diff"),
                    hybrid_score_diff_ema=getattr(optimizer, "last_stats", {}).get("hybrid_score_diff_ema"),
                    hybrid_gate_advantage=getattr(optimizer, "last_stats", {}).get("hybrid_gate_advantage"),
                    hybrid_reward=getattr(optimizer, "last_stats", {}).get("hybrid_reward"),
                    hybrid_reward_ema=getattr(optimizer, "last_stats", {}).get("hybrid_reward_ema"),
                    uap_classical_weight=getattr(optimizer, "last_stats", {}).get("uap_classical_weight"),
                    uap_shadow_weight=getattr(optimizer, "last_stats", {}).get("uap_shadow_weight"),
                    uap_obstruction=getattr(optimizer, "last_stats", {}).get("uap_obstruction"),
                    uap_obstruction_ema=getattr(optimizer, "last_stats", {}).get("uap_obstruction_ema"),
                    uap_residue_mass=getattr(optimizer, "last_stats", {}).get("uap_residue_mass"),
                    uap_collapse_pressure=getattr(optimizer, "last_stats", {}).get("uap_collapse_pressure"),
                    uap_trust_ratio_mean=getattr(optimizer, "last_stats", {}).get("uap_trust_ratio_mean"),
                    uap_update_rms_mean=getattr(optimizer, "last_stats", {}).get("uap_update_rms_mean"),
                    uap_effective_scale_mean=getattr(optimizer, "last_stats", {}).get("uap_effective_scale_mean"),
                    uap_max_update_rms=getattr(optimizer, "last_stats", {}).get("uap_max_update_rms"),
                    uap_gate_advantage=getattr(optimizer, "last_stats", {}).get("uap_gate_advantage"),
                    shadow_uap_classical_weight=getattr(optimizer, "last_stats", {}).get("shadow_uap_classical_weight"),
                    shadow_uap_shadow_weight=getattr(optimizer, "last_stats", {}).get("shadow_uap_shadow_weight"),
                    shadow_uap_obstruction=getattr(optimizer, "last_stats", {}).get("shadow_uap_obstruction"),
                    shadow_uap_residue_mass=getattr(optimizer, "last_stats", {}).get("shadow_uap_residue_mass"),
                    shadow_uap_collapse_pressure=getattr(optimizer, "last_stats", {}).get("shadow_uap_collapse_pressure"),
                    **gpu_memory_gb(),
                )

        losses_per_epoch.append(ep_losses)
        if aborted_for:
            break

        if val_shards:
            eval_result = run_full_eval(
                model, shard_dir, val_fraction=val_fraction,
                max_examples_ppl=400, max_examples_acc=400,
                max_examples_calib=200, device=device, seed=seed,
                tokenizer=tokenizer,
            )
            eval_result["epoch"] = ep + 1
            eval_history.append(eval_result)
            ppl = eval_result.get("perplexity", {}).get("perplexity")
            acc = eval_result.get("top1_accuracy", {}).get("top1_accuracy")
            logging.info(
                "pretrain ep=%d eval: ppl=%s acc=%s",
                ep + 1,
                f"{ppl:.3f}" if isinstance(ppl, float) and math.isfinite(ppl) else str(ppl),
                f"{acc:.3f}" if isinstance(acc, float) and math.isfinite(acc) else str(acc),
            )

    t_total = time.time() - t_start

    if save_path is not None:
        save_path = Path(save_path)
        written = save_training_checkpoint(
            save_path, model, optimizer, step=total_steps_run, epoch=len(losses_per_epoch),
            metadata={
                "profile_name": profile_name,
                "frontier_semantic_mode": frontier_semantic_mode,
                "memory_estimate": memory_estimate,
                "tokenizer": tokenizer.info(),
                "uap_loop_penalty_weight": uap_loop_penalty_weight,
                "uap_sheaf_weight": uap_sheaf_weight,
            },
            sharded=save_sharded,
        )
        if written is not None:
            logging.info(f"pretrain: saved training checkpoint to {written}")

    return {
        "shard_dir": str(shard_dir),
        "train_shards": [str(s) for s in train_shards],
        "val_shards": [str(s) for s in val_shards],
        "tokenizer": tokenizer.info(),
        "vocab_size": vocab_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "total_steps": total_steps_run,
        "epoch_avg_loss": [
            sum(L) / max(1, len(L)) if L else float("nan")
            for L in losses_per_epoch
        ],
        "epoch_first_loss": [L[0] if L else None for L in losses_per_epoch],
        "epoch_last_loss": [L[-1] if L else None for L in losses_per_epoch],
        "final_phase": last_phase,
        "profile_name": profile_name,
        "model_class": _unwrap_model(model).__class__.__name__,
        "optimizer_class": optimizer.__class__.__name__,
        "distributed_world_size": dist_utils.world_size(),
        "use_fsdp": use_fsdp,
        "use_ddp": use_ddp,
        "fsdp_mixed_precision": fsdp_mixed_precision,
        "frontier_semantic_mode": frontier_semantic_mode,
        "uap_aux_weight": uap_aux_weight,
        "uap_loop_penalty_weight": uap_loop_penalty_weight,
        "uap_sheaf_weight": uap_sheaf_weight,
        "uap_schema_version": "tovah-uap-token-profile-v14.3.3",
        "memory_estimate": memory_estimate,
        "resumed_from": str(resume_from) if resume_from else None,
        "save_sharded": save_sharded,
        "wall_time_seconds": round(t_total, 3),
        "save_path": str(save_path) if save_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "eval_history": eval_history,
        "aborted": aborted_for or None,
    }
