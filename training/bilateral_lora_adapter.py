"""v14.3.5 QLoRA/DoRA + bilateral adapter utilities.

This file remains dependency-light.  When PEFT is installed it can build a
QLoRA/DoRA base adapter; independently, ``BilateralDoRALinear`` provides a pure
PyTorch directional low-rank adapter with separate T/F lanes and metadata gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BilateralLoRAConfig:
    base_model_name: str = "Qwen/Qwen3-7B"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    load_in_4bit: bool = True
    use_dora: bool = True


class BilateralDoRALinear(nn.Module):
    """Low-rank directional adapter with separate affirmation/refutation lanes.

    The frozen base weight supplies the classical trunk.  Two low-rank updates
    specialize the direction for T and F supports; gates may be supplied from
    corpus metadata or semantic heads.  This is the adapter-scale version of
    TOVAH's bilateral ontology.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout)
        in_f, out_f = base.in_features, base.out_features
        self.A_T = nn.Parameter(torch.empty(self.r, in_f))
        self.B_T = nn.Parameter(torch.zeros(out_f, self.r))
        self.A_F = nn.Parameter(torch.empty(self.r, in_f))
        self.B_F = nn.Parameter(torch.zeros(out_f, self.r))
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1).clamp_min(1e-6).clone())
        nn.init.kaiming_uniform_(self.A_T, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.A_F, a=5 ** 0.5)

    def _delta(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.dropout(x), A), B) * self.scale

    def forward(self, x: torch.Tensor, gate_T: Optional[torch.Tensor] = None, gate_F: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_out = self.base(x)
        dT = self._delta(x, self.A_T, self.B_T)
        dF = self._delta(x, self.A_F, self.B_F)
        if gate_T is None:
            gate_T = torch.ones_like(dT[..., :1])
        if gate_F is None:
            gate_F = torch.zeros_like(dF[..., :1])
        return base_out + gate_T * dT + gate_F * dF


def build_bilateral_lora_model(config: BilateralLoRAConfig = BilateralLoRAConfig()):
    """Build a PEFT LoRA/DoRA model when optional dependencies are installed."""
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore
    except Exception as exc:  # pragma: no cover - optional deps
        raise RuntimeError(
            "QLoRA/DoRA adapter mode requires optional packages: transformers, peft, and optionally bitsandbytes"
        ) from exc

    kwargs = {}
    if config.load_in_4bit:
        kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name, **kwargs)
    peft_cfg = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
        use_dora=config.use_dora,
    )
    return get_peft_model(model, peft_cfg)


__all__ = ["BilateralLoRAConfig", "BilateralDoRALinear", "build_bilateral_lora_model"]
