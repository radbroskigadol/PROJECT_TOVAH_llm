"""TOVAH v14.2.6 training/scale_ladder.py — buyer scaling ladder metadata.

This module is deliberately lightweight: it gives a buyer/operator a concrete
progression from CPU/debug validation to 13B reference planning without forcing
allocation of large models. It is used by docs, scripts, and presale smoke tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from tovah_v14.config.constants import MODEL_PROFILES
from tovah_v14.neural.scaling import FRONTIER_PROFILES, estimate_frontier_memory, estimate_param_count


@dataclass(frozen=True)
class ScaleStage:
    name: str
    profile: str
    target_params: str
    semantic_mode: str
    recommended_device: str
    recommended_world_size: int
    batch_size: int
    grad_accum_steps: int
    context_len: int
    notes: str


SCALE_LADDER: List[ScaleStage] = [
    ScaleStage(
        name="debug_5m",
        profile="debug",
        target_params="~5M-class classic ShadowTokenCore",
        semantic_mode="logits",
        recommended_device="CPU or single consumer GPU",
        recommended_world_size=1,
        batch_size=2,
        grad_accum_steps=1,
        context_len=256,
        notes="Install, corpus, pretrain, checkpoint, and eval smoke validation.",
    ),
    ScaleStage(
        name="research_50m",
        profile="large",
        target_params="~50M-class classic ShadowTokenCore",
        semantic_mode="logits",
        recommended_device="single GPU",
        recommended_world_size=1,
        batch_size=4,
        grad_accum_steps=4,
        context_len=1024,
        notes="Validates bilateral logits and ShadowOptimizer behavior before frontier mode.",
    ),
    ScaleStage(
        name="frontier_dev",
        profile="frontier_dev",
        target_params="frontier architecture, dev scale",
        semantic_mode="hidden",
        recommended_device="single GPU preferred; CPU for smoke only",
        recommended_world_size=1,
        batch_size=1,
        grad_accum_steps=8,
        context_len=512,
        notes="First ScalableBilateralCore/RoPE/GQA/hidden-semantic-head validation.",
    ),
    ScaleStage(
        name="frontier_2b",
        profile="frontier_2b",
        target_params="~2B–3B class depending vocab/mode",
        semantic_mode="hidden",
        recommended_device="multi-GPU FSDP or 80GB-class GPU",
        recommended_world_size=4,
        batch_size=1,
        grad_accum_steps=32,
        context_len=2048,
        notes="First serious FSDP checkpoint/resume/throughput milestone.",
    ),
    ScaleStage(
        name="frontier_7b",
        profile="frontier_7b",
        target_params="~7B class",
        semantic_mode="hidden",
        recommended_device="multi-node or 8x80GB-class FSDP",
        recommended_world_size=8,
        batch_size=1,
        grad_accum_steps=64,
        context_len=4096,
        notes="Requires real profiler pass, sharded checkpointing, and sustained data streaming.",
    ),
    ScaleStage(
        name="frontier_13b_reference",
        profile="frontier_13b",
        target_params="~13B class in shared bilateral mode",
        semantic_mode="hidden",
        recommended_device="buyer cluster; FSDP-first, TP/ZeRO-3 optional follow-on",
        recommended_world_size=16,
        batch_size=1,
        grad_accum_steps=128,
        context_len=4096,
        notes="Reference launch plan, not a claim of completed 13B training.",
    ),
]


def ladder_as_dicts() -> List[Dict[str, Any]]:
    return [asdict(stage) for stage in SCALE_LADDER]


def memory_plan(*, vocab_size: int = 50257, dtype: str = "bf16") -> Dict[str, Any]:
    """Return capacity-planning estimates for frontier stages without allocating models."""
    out: Dict[str, Any] = {"vocab_size": vocab_size, "dtype": dtype, "stages": []}
    for stage in SCALE_LADDER:
        row = asdict(stage)
        if stage.profile in FRONTIER_PROFILES:
            row["memory_estimate"] = estimate_frontier_memory(
                stage.profile,
                vocab_size=vocab_size,
                batch_size=stage.batch_size,
                seq_len=min(stage.context_len, FRONTIER_PROFILES[stage.profile]["max_len"]),
                world_size=stage.recommended_world_size,
                use_fsdp=stage.recommended_world_size > 1,
                dtype=dtype,
                gradient_checkpointing=True,
                bilateral_mode="shared",
            )
            row["estimated_params"] = estimate_param_count(stage.profile, vocab_size=vocab_size)
        else:
            row["model_profile"] = MODEL_PROFILES.get(stage.profile, {})
        out["stages"].append(row)
    return out


def main() -> None:
    print(json.dumps(memory_plan(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
