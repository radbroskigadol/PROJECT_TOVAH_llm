"""
TOVAH v14.2.6 neural/distributed.py — Multi-GPU training scaffolding.

This module is the integration point for scaling beyond single-GPU. It
provides:

  - is_distributed_available() — quick check for torch.distributed
  - init_distributed()         — initialize a process group (NCCL default)
  - wrap_ddp(model)            — wrap a model in DistributedDataParallel
  - wrap_fsdp(model)           — wrap a model in FullyShardedDataParallel
                                  (FSDP, available in torch ≥ 1.12)
  - rank, world_size, is_main  — convenience accessors
  - barrier(), cleanup()       — process synchronization

HONEST SCOPE:
  This is SCAFFOLDING, not a turnkey distributed-training driver. The
  wrappers exist and the integration points are documented; running an
  actual multi-node training requires an external launcher (torchrun,
  slurm srun, etc.) plus distributed-aware dataloaders. The `pretrain()`
  entry point in training/pretrain.py uses these wrappers when env vars
  RANK/WORLD_SIZE are set — that's the integration surface.

  Pipeline parallelism, tensor parallelism, and ZeRO-3 are NOT
  implemented — they require either Megatron-LM-style layer-wise
  rewrites or third-party libraries (deepspeed, accelerate). Those
  belong to v15+.

Typical launch (single-host, 4 GPUs):

    torchrun --nproc_per_node=4 run_tovah.py \\
        --pretrain --profile frontier_2b --use-fsdp

The pretrain entry point detects distributed env vars and wires up
the model with FSDP / DDP automatically.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional


def is_distributed_available() -> bool:
    """True iff torch.distributed is importable AND env says we're in
    a distributed run (RANK + WORLD_SIZE set)."""
    try:
        import torch.distributed as dist
        return ("RANK" in os.environ
                and "WORLD_SIZE" in os.environ
                and dist.is_available())
    except Exception:
        return False


def init_distributed(backend: str = "nccl") -> Optional[int]:
    """Initialize the default process group.

    Returns the local rank (int) if successful, None if torch.distributed
    is not available / not configured.

    Recognized env vars: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
    MASTER_PORT (standard torchrun set).
    """
    if not is_distributed_available():
        return None
    import torch
    import torch.distributed as dist
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        use_backend = backend
    else:
        # gloo for CPU fallback (mostly for testing).
        use_backend = "gloo"
    dist.init_process_group(backend=use_backend, rank=rank, world_size=world_size)
    logging.info(
        "distributed: initialized rank=%d/%d local_rank=%d backend=%s",
        rank, world_size, local_rank, use_backend,
    )
    return local_rank


def rank() -> int:
    """Current rank, or 0 if not distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def world_size() -> int:
    """World size, or 1 if not distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def is_main() -> bool:
    """True on the rank-0 process (or always True in single-process runs)."""
    return rank() == 0


def barrier() -> None:
    """Block until all ranks reach this point."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def cleanup() -> None:
    """Destroy the process group."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# --- Model wrappers --------------------------------------------------------

def wrap_ddp(model: Any, *, find_unused_parameters: bool = False) -> Any:
    """Wrap a model in DistributedDataParallel.

    Returns the wrapped model. Caller must have initialized the process
    group first (call init_distributed()).

    DDP replicates the full model on every rank — fine for models up to
    the per-GPU memory limit (~7B on 24GB GPUs with FP16). For larger
    models, use wrap_fsdp.
    """
    import torch
    import torch.nn.parallel
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("wrap_ddp called before init_distributed()")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )


def fsdp_mixed_precision_policy(dtype: Optional[str] = None) -> Optional[Any]:
    """Build a torch FSDP MixedPrecision policy, or None for fp32/default.

    Kept as a separate helper so tests and launch guards can validate the
    policy without initializing a distributed process group.
    """
    if dtype is None or str(dtype).lower() in {"", "none", "fp32", "float32"}:
        return None
    try:
        import torch
        from torch.distributed.fsdp import MixedPrecision
    except Exception as e:  # pragma: no cover - depends on torch build
        raise RuntimeError("FSDP mixed precision requires torch.distributed.fsdp") from e
    d = str(dtype).lower()
    if d in {"bf16", "bfloat16"}:
        t = torch.bfloat16
    elif d in {"fp16", "float16", "half"}:
        t = torch.float16
    else:
        raise ValueError("FSDP dtype must be fp32, bf16, or fp16")
    return MixedPrecision(param_dtype=t, reduce_dtype=t, buffer_dtype=t)


def wrap_fsdp(model: Any, *,
              auto_wrap_min_params: int = int(1e8),
              mixed_precision: Optional[str] = None,
              cpu_offload: bool = False,
              use_orig_params: bool = True) -> Any:
    """Wrap a model in FullyShardedDataParallel (FSDP).

    FSDP shards parameters, gradients, and optimizer state across ranks
    — necessary for models that don't fit on one GPU. Available in torch
    ≥ 1.12, fully matured in 2.0+.

    `auto_wrap_min_params` controls which submodules get wrapped as their
    own shard groups. The default (1e8 params) is a good starting point;
    too small wraps everything in tiny shards, too large defeats sharding.

    Returns the wrapped model. Like wrap_ddp, requires init_distributed()
    to have been called first.

    HONEST NOTE: FSDP has interactions with custom optimizers (it expects
    optimizer state-dict APIs that the bilateral ShadowOptimizer doesn't
    fully implement). For FSDP training, use the AdamW path.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except ImportError as e:
        raise RuntimeError(
            "FSDP requires torch ≥ 1.12 with distributed support"
        ) from e

    import torch
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("wrap_fsdp called before init_distributed()")

    import functools
    policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=auto_wrap_min_params,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
    mixed = fsdp_mixed_precision_policy(mixed_precision)
    cpu_offload_obj = None
    if cpu_offload:
        try:
            from torch.distributed.fsdp import CPUOffload
            cpu_offload_obj = CPUOffload(offload_params=True)
        except Exception:
            logging.warning("FSDP CPUOffload unavailable in this torch build")
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mixed,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload_obj,
        use_orig_params=use_orig_params,
    )


# --- Distributed data sampler -----------------------------------------------

def distributed_sampler_for_dataset(dataset: Any, *,
                                    shuffle: bool = True,
                                    seed: int = 0) -> Optional[Any]:
    """Build a DistributedSampler for a Dataset, or None if not distributed.

    Use the result as the `sampler` argument to a DataLoader. For
    IterableDataset (like CorpusShardDataset), the dataset itself is
    worker-and-rank-aware via get_worker_info() / dist.get_rank() — no
    external sampler needed.
    """
    if not is_distributed_available():
        return None
    try:
        from torch.utils.data import DistributedSampler
        from torch.utils.data import IterableDataset
        if isinstance(dataset, IterableDataset):
            # IterableDataset partitions internally.
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=shuffle,
            seed=seed,
        )
    except Exception as e:
        logging.warning("distributed_sampler_for_dataset failed: %s", e)
        return None
