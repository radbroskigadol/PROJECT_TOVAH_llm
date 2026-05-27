# FSDP_RUNBOOK.md — TOVAH v14.2.6

## Single-node smoke

```bash
NPROC_PER_NODE=4 PROFILE=frontier_2b MAX_STEPS=100 scripts/train_fsdp_single_node.sh
```

## Multi-node reference

The scheduler must provide:

```bash
NNODES=2
NODE_RANK=0
MASTER_ADDR=<rank0-host>
MASTER_PORT=29500
NPROC_PER_NODE=8
SHARD_DIR=/data/tovah/shards
PROFILE=frontier_13b
```

Then run:

```bash
scripts/train_fsdp_multi_node.sh
```

## Recommended first strategy

Use FSDP + bf16 + hidden semantic heads first. Add tensor parallelism or
DeepSpeed ZeRO-3 only after `frontier_2b` and `frontier_7b` reveal concrete
memory/throughput bottlenecks on buyer hardware.

## Checkpointing

Use `--save-sharded` for FSDP scale runs. v14.2.6 writes a checkpoint manifest
next to the checkpoint path. Buyers should rehearse:

1. train 20 steps
2. save checkpoint
3. kill the job
4. resume with `--resume-from`
5. verify loss and step continuity

## NCCL/environment notes

Typical variables a buyer may tune:

```bash
NCCL_DEBUG=INFO
NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_BLOCKING_WAIT=1
CUDA_DEVICE_MAX_CONNECTIONS=1
OMP_NUM_THREADS=8
```

Do not run live autonomy mode on cluster login nodes.
