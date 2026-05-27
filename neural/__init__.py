"""
TOVAH v14 neural — ShadowHoTT bilateral neural components.

Owns:
- ShadowTokenCore (bilateral transformer; v13 architecture, preserved)
- ScalableBilateralCore (v14.2.6 frontier-scale architecture with RoPE,
                         GQA, gradient checkpointing, tied embeddings)
- ShadowOptimizer (bilateral evidence-based optimizer)
- AdamWWrapper and MuonWrapper frontier-scaling optimizer paths
- Scoring, Training step, Paraconsistent invariants/loss
- Distributed scaffolding (v14.2.6: DDP / FSDP wrappers)

This module imports from config/, core/, and hott/ only.
It MUST NOT import from kernel/, tools/, mutation/, persistence/, or debug/.
"""
from tovah_v14.neural.shadow_model import (
    DualLinear,
    BilateralAttention,
    BilateralFFN,
    BilateralBlock,
    ShadowTokenCore,
)
from tovah_v14.neural.optimizer import ShadowOptimizer
from tovah_v14.neural.scoring import shadow_score_text, shadow_score_scalar
from tovah_v14.neural.training import (
    train_shadow_step,
    compute_paraconsistent_invariants,
    semantic_rank_nullity_loss,
)
# v14.2.6 scaling additions.
from tovah_v14.neural.scaling import (
    ScalableBilateralCore,
    ScalableBilateralBlock,
    BilateralGQAttention,
    BilateralSwiGLU,
    BilateralBelnapMoEFFN,
    RMSNorm,
    FRONTIER_PROFILES,
    make_scalable_model,
    estimate_param_count,
    estimate_frontier_memory,
)
from tovah_v14.neural.adamw import AdamWWrapper, make_optimizer
from tovah_v14.neural.muon import MuonWrapper, zeropower_via_newtonschulz
from tovah_v14.neural.checkpointing import save_training_checkpoint, load_training_checkpoint
from tovah_v14.neural import distributed

from tovah_v14.neural.bilateral_ssm import BilateralSSMBlock
from tovah_v14.neural.paraconsistent_smooth import smooth_min, smooth_max, lukasiewicz_tnorm, probabilistic_tconorm
