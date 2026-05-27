"""
TOVAH v14 neural/training.py — Training step and paraconsistent losses.

SEMANTIC PRESERVATION:
  compute_paraconsistent_invariants and semantic_rank_nullity_loss preserve the
  v14.1.2 mean-reduced bilateral objective.

v14.2.3 HIGH-GLUT GRADIENT PATCH:
  - live and pretrain paths can consume corpus bilateral_t/bilateral_f metadata;
  - K-heavy examples receive lane-B routing pressure rather than blind collapse;
  - G-heavy examples receive lane-C routing pressure;
  - contradiction/gap budgets are relaxed per example when metadata says the
    example is truly glut/gap-heavy;
  - phase detection uses mean glut/gap plus metadata, not shape-dependent sums
    or the h_lambda == 0 condition.

v14.2.3 COMPLETION:
  - explicit Lane-B contradiction-preservation regularizer;
  - explicit Lane-C gap-tolerance regularizer;
  - metadata K/G matching is now differentiable and wired into live/pretrain loss.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tovah_v14.neural.paraconsistent_smooth import smooth_min, smooth_max

from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.neural.optimizer import ShadowOptimizer
from tovah_v14.neural.scoring import encode_bytes


def compute_paraconsistent_invariants(
    T: torch.Tensor,
    Fv: torch.Tensor,
    lambda_budget: float = 0.05,
) -> Tuple[Tuple[float, float, float, float], float, int]:
    """Compute Sigma=(A,B,K,G), sigma (contradiction mass), h_lambda.

    FORMULA:
      A = relu(T - F)       # classical truth
      B = relu(F - T)       # classical falsity
      K = min(T, F)         # contradiction kernel
      G = 1 - max(T, F)     # gap
      sigma = K.sum()
      h_lambda = iterative kernel removal depth
    """
    A = torch.relu(T - Fv)
    B = torch.relu(Fv - T)
    K = torch.minimum(T, Fv)
    G = 1.0 - torch.maximum(T, Fv)
    Sigma = (A.sum().item(), B.sum().item(), K.sum().item(), G.sum().item())
    sigma = K.sum().item()
    h_lambda = 0
    curr_T = T.detach().clone()
    curr_F = Fv.detach().clone()
    while True:
        kernel = torch.minimum(curr_T, curr_F)
        if kernel.sum().item() < 1e-4:
            break
        removal = torch.clamp(kernel, max=lambda_budget)
        next_T = torch.relu(curr_T - removal)
        next_F = torch.relu(curr_F - removal)
        if torch.allclose(curr_T, next_T, atol=1e-7) and torch.allclose(
            curr_F, next_F, atol=1e-7
        ):
            break
        curr_T, curr_F = next_T, next_F
        h_lambda += 1
        if h_lambda > 5000:
            break
    return Sigma, sigma, h_lambda


def differentiable_paraconsistent_surrogates(
    T: torch.Tensor,
    Fv: torch.Tensor,
    temperature: float = 32.0,
) -> Dict[str, torch.Tensor]:
    """Differentiable K/G/A/B surrogates for training-time use."""
    k = smooth_min(T, Fv, temperature=temperature)
    max_tf = smooth_max(T, Fv, temperature=temperature)
    g = 1.0 - max_tf
    a = F.relu(T - Fv)
    b = F.relu(Fv - T)
    return {"A": a, "B": b, "K": k.clamp(0.0, 1.0), "G": g.clamp(0.0, 1.0)}


def semantic_rank_nullity_loss(
    T: torch.Tensor,
    Fv: torch.Tensor,
    con_budget: float = 0.12,
    gap_budget: float = 0.20,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, float, float, float, float]:
    """ShadowHoTT semantic calibration loss for compact supports.

    v14.3.4 deliberately stops treating all contradiction/gap mass as an error.
    ``con_budget`` and ``gap_budget`` are target priors, not hard ceilings.  This
    keeps classical examples near low K/G while allowing metadata-weighted losses
    to preserve genuine gluts and gaps instead of forcing classicalization.

    Uses element-wise averages so the loss is O(1) across sequence length and
    across hidden-support vs legacy full-vocab tensors.
    """
    dim_con = torch.sum(torch.minimum(T, Fv))
    dim_gap = torch.sum(1.0 - torch.maximum(T, Fv))
    n = max(1, T.numel())
    con_avg = dim_con / n
    gap_avg = dim_gap / n
    con_target = torch.as_tensor(0.0 if con_budget is None else con_budget,
                                 device=T.device, dtype=T.dtype)
    gap_target = torch.as_tensor(0.0 if gap_budget is None else gap_budget,
                                 device=T.device, dtype=T.dtype)
    loss = alpha * (con_avg - con_target).pow(2) + beta * (gap_avg - gap_target).pow(2)
    return (
        loss,
        float(dim_con.item()),
        float(dim_gap.item()),
        float(con_avg.item()),
        float(gap_avg.item()),
    )


def paraconsistent_example_weights(
    bilateral_t: torch.Tensor,
    bilateral_f: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return per-example classical/glut/gap weights from corpus metadata.

    The four semantic lanes are runtime projections, not corpus classes. For
    training-control purposes we route:
      - classical true-only or false-only examples to lane A;
      - high-glut examples to lane B;
      - high-gap examples to lane C;
      - lane D remains a forced-readout lane and is not used as a default target.
    """
    t = torch.clamp(bilateral_t.float(), 0.0, 1.0)
    f = torch.clamp(bilateral_f.float(), 0.0, 1.0)
    k = torch.minimum(t, f)
    g = torch.minimum(1.0 - t, 1.0 - f)
    classical = torch.clamp(torch.abs(t - f), 0.0, 1.0)
    return {"classical": classical, "K": k, "G": g, "t": t, "f": f}


def _mean_gate_log_probs(
    gate_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return B×4 mean log-probs from B×4 or B×L×4 gate logits."""
    logp = F.log_softmax(gate_logits, dim=-1)
    if logp.dim() == 2:
        return logp
    if logp.dim() != 3:
        raise ValueError(f"gate_logits must have rank 2 or 3, got {tuple(logp.shape)}")
    if attention_mask is None:
        return logp.mean(dim=1)
    m = attention_mask.to(logp.device, dtype=logp.dtype)
    if m.shape[1] != logp.shape[1]:
        L = min(m.shape[1], logp.shape[1])
        m = m[:, :L]
        logp = logp[:, :L, :]
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (logp * m.unsqueeze(-1)).sum(dim=1) / denom


def lane_routing_loss(
    gate_logits: torch.Tensor,
    bilateral_t: torch.Tensor,
    bilateral_f: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Metadata-aware four-lane routing loss.

    K-heavy examples are softly routed toward lane B; G-heavy examples toward
    lane C; classical non-glut/non-gap examples toward lane A. Lane D is not
    targeted by ordinary training because it is the forced-totalization/readout
    lane. This turns the four lanes into real gradient-routing surfaces instead
    of merely generation-time labels.
    """
    w = paraconsistent_example_weights(bilateral_t.to(gate_logits.device), bilateral_f.to(gate_logits.device))
    eps = torch.full_like(w["K"], 1e-4)
    zero = torch.zeros_like(w["K"])
    target = torch.stack([w["classical"], w["K"], w["G"], zero], dim=-1) + eps.unsqueeze(-1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    logp = _mean_gate_log_probs(gate_logits, attention_mask=attention_mask)
    severity = 1.0 + torch.maximum(w["K"], w["G"])
    return (-(target * logp).sum(dim=-1) * severity).mean()


def _per_example_semantic_masses(
    T: torch.Tensor,
    Fv: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-example predicted K/G mass from B×L×V T/F tensors.

    K is the contradiction/glut kernel min(T,F). G is the paracomplete gap
    1-max(T,F). Both are averaged over vocabulary and valid tokens so the
    target scale is stable across vocab size and sequence length.
    """
    if T.shape != Fv.shape:
        raise ValueError(f"T and Fv must have identical shapes, got {tuple(T.shape)} and {tuple(Fv.shape)}")
    if T.dim() < 2:
        raise ValueError(f"T/Fv must include a batch dimension, got rank {T.dim()}")

    if T.dim() == 2:
        pred_k = torch.minimum(T, Fv).mean(dim=-1)
        pred_g = (1.0 - torch.maximum(T, Fv)).mean(dim=-1)
        return pred_k, pred_g

    # Standard path: B×L×V. Extra trailing feature dims are collapsed.
    reduce_dims = tuple(range(2, T.dim()))
    con_tok = torch.minimum(T, Fv).mean(dim=reduce_dims)
    gap_tok = (1.0 - torch.maximum(T, Fv)).mean(dim=reduce_dims)
    if attention_mask is not None:
        m = attention_mask.to(device=T.device, dtype=T.dtype)
        if m.dim() != 2:
            raise ValueError(f"attention_mask must be B×L, got {tuple(m.shape)}")
        if m.shape[1] != con_tok.shape[1]:
            L = min(m.shape[1], con_tok.shape[1])
            con_tok = con_tok[:, :L]
            gap_tok = gap_tok[:, :L]
            m = m[:, :L]
        denom = m.sum(dim=-1).clamp_min(1.0)
        return (con_tok * m).sum(dim=-1) / denom, (gap_tok * m).sum(dim=-1) / denom
    return con_tok.mean(dim=-1), gap_tok.mean(dim=-1)


def _mean_gate_probs(
    gate_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return B×4 mean probabilities from B×4 or B×L×4 gate logits."""
    probs = F.softmax(gate_logits, dim=-1)
    if probs.dim() == 2:
        return probs
    if probs.dim() != 3:
        raise ValueError(f"gate_logits must have rank 2 or 3, got {tuple(probs.shape)}")
    if attention_mask is None:
        return probs.mean(dim=1)
    m = attention_mask.to(probs.device, dtype=probs.dtype)
    if m.shape[1] != probs.shape[1]:
        L = min(m.shape[1], probs.shape[1])
        probs = probs[:, :L, :]
        m = m[:, :L]
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (probs * m.unsqueeze(-1)).sum(dim=1) / denom


def _smooth_match_penalty(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: float = 0.03,
    beta: float = 0.05,
) -> torch.Tensor:
    """Huber-like match penalty with a dead-zone tolerance.

    The tolerance prevents noisy token-level estimates from forcing exact K/G
    equality on every mini-batch. Outside the dead zone the penalty is smooth L1.
    """
    err = torch.relu(torch.abs(predicted - target) - tolerance)
    beta_t = torch.as_tensor(beta, device=predicted.device, dtype=predicted.dtype).clamp_min(1e-6)
    return torch.where(err < beta_t, 0.5 * err.pow(2) / beta_t, err - 0.5 * beta_t)


def _compute_lane_b_regularizer(
    T: torch.Tensor,
    Fv: torch.Tensor,
    gate_logits: torch.Tensor,
    bilateral_t: torch.Tensor,
    bilateral_f: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    tolerance: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Contradiction-preserving Lane-B regularizer.

    Mathematical target:
      K_pred(x) = mean[min(T,F)] for the example.
      K_meta(x) = min(bilateral_t,bilateral_f).
      p_B(x)    = mean softmax(gate_logits)_B.

    The raw penalty is smooth-L1(max(|K_pred-K_meta|-tol,0)). The returned loss
    is weighted outside this function by K_meta in the calling training loss.
    We additionally multiply by (0.25 + 0.75*p_B) so semantic matching is
    strongest when the lane gate has selected B, but non-zero before routing is
    perfect. This keeps the function differentiable with respect to both the
    bilateral logits and the lane gate.
    """
    device, dtype = T.device, T.dtype
    meta_k = torch.minimum(
        bilateral_t.to(device=device, dtype=dtype).clamp(0.0, 1.0),
        bilateral_f.to(device=device, dtype=dtype).clamp(0.0, 1.0),
    )
    pred_k, _pred_g = _per_example_semantic_masses(T, Fv, attention_mask=attention_mask)
    gate_probs = _mean_gate_probs(gate_logits, attention_mask=attention_mask)
    p_b = gate_probs[:, 1]
    raw = _smooth_match_penalty(pred_k, meta_k, tolerance=tolerance)
    lane_focus = 0.25 + 0.75 * p_b
    loss = raw * lane_focus
    stats = {
        "lane_b_pred_k_mean": float(pred_k.detach().mean().item()),
        "lane_b_meta_k_mean": float(meta_k.detach().mean().item()),
        "lane_b_gate_mean": float(p_b.detach().mean().item()),
        "lane_b_raw_penalty_mean": float(raw.detach().mean().item()),
    }
    return loss, stats


def _compute_lane_c_regularizer(
    T: torch.Tensor,
    Fv: torch.Tensor,
    gate_logits: torch.Tensor,
    bilateral_t: torch.Tensor,
    bilateral_f: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    tolerance: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Gap-tolerating Lane-C regularizer.

    Mathematical target:
      G_pred(x) = mean[1 - max(T,F)] for the example.
      G_meta(x) = min(1-bilateral_t,1-bilateral_f).
      p_C(x)    = mean softmax(gate_logits)_C.

    The raw penalty is smooth-L1(max(|G_pred-G_meta|-tol,0)). The returned loss
    is weighted outside this function by G_meta in the calling training loss.
    The lane-focus factor mirrors Lane B, making semantic gap matching strongest
    under active Lane C while keeping gradients alive before the gate converges.
    """
    device, dtype = T.device, T.dtype
    bt = bilateral_t.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    bf = bilateral_f.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    meta_g = torch.minimum(1.0 - bt, 1.0 - bf)
    _pred_k, pred_g = _per_example_semantic_masses(T, Fv, attention_mask=attention_mask)
    gate_probs = _mean_gate_probs(gate_logits, attention_mask=attention_mask)
    p_c = gate_probs[:, 2]
    raw = _smooth_match_penalty(pred_g, meta_g, tolerance=tolerance)
    lane_focus = 0.25 + 0.75 * p_c
    loss = raw * lane_focus
    stats = {
        "lane_c_pred_g_mean": float(pred_g.detach().mean().item()),
        "lane_c_meta_g_mean": float(meta_g.detach().mean().item()),
        "lane_c_gate_mean": float(p_c.detach().mean().item()),
        "lane_c_raw_penalty_mean": float(raw.detach().mean().item()),
    }
    return loss, stats


def lane_semantic_matching_loss(
    T: torch.Tensor,
    Fv: torch.Tensor,
    gate_logits: torch.Tensor,
    bilateral_t: torch.Tensor,
    bilateral_f: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    tolerance: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """K/G metadata matching loss for Lane B and Lane C.

    The regularizer functions return raw per-example penalties. This combiner
    applies the requested K/G multipliers:
      L_B = K_meta * LaneBRegularizer
      L_C = G_meta * LaneCRegularizer
    then averages over the mini-batch. This makes contradiction preservation and
    gap tolerance vanish naturally for examples whose metadata says they are not
    K/G-heavy.
    """
    device, dtype = T.device, T.dtype
    w = paraconsistent_example_weights(
        bilateral_t.to(device=device, dtype=dtype),
        bilateral_f.to(device=device, dtype=dtype),
    )
    lane_b, b_stats = _compute_lane_b_regularizer(
        T, Fv, gate_logits, bilateral_t, bilateral_f,
        attention_mask=attention_mask, tolerance=tolerance,
    )
    lane_c, c_stats = _compute_lane_c_regularizer(
        T, Fv, gate_logits, bilateral_t, bilateral_f,
        attention_mask=attention_mask, tolerance=tolerance,
    )
    b_weighted = w["K"] * lane_b
    c_weighted = w["G"] * lane_c
    loss = (b_weighted + c_weighted).mean()
    stats: Dict[str, float] = {}
    stats.update(b_stats)
    stats.update(c_stats)
    stats.update({
        "lane_semantic_k_weight_mean": float(w["K"].detach().mean().item()),
        "lane_semantic_g_weight_mean": float(w["G"].detach().mean().item()),
        "lane_semantic_b_weighted_mean": float(b_weighted.detach().mean().item()),
        "lane_semantic_c_weighted_mean": float(c_weighted.detach().mean().item()),
    })
    return loss, stats


def metadata_weighted_semantic_loss(
    T: torch.Tensor,
    Fv: torch.Tensor,
    bilateral_t: torch.Tensor,
    bilateral_f: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    con_budget: float = 0.12,
    gap_budget: float = 0.20,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-example semantic loss using corpus bilateral metadata.

    The ordinary semantic loss treats all contradiction/gap mass as equally
    suspicious. For known K/G examples, that over-collapses evidence. This loss
    relaxes the contradiction budget for K-heavy examples and the gap budget for
    G-heavy examples, while still penalizing excess mass beyond the metadata-
    adjusted budget.
    """
    device = T.device
    t_meta = bilateral_t.to(device=device, dtype=T.dtype).clamp(0.0, 1.0)
    f_meta = bilateral_f.to(device=device, dtype=T.dtype).clamp(0.0, 1.0)
    w = paraconsistent_example_weights(t_meta, f_meta)

    con_tok = torch.minimum(T, Fv).mean(dim=-1)  # B×L
    gap_tok = (1.0 - torch.maximum(T, Fv)).mean(dim=-1)
    if attention_mask is not None:
        m = attention_mask.to(device=device, dtype=T.dtype)
        if m.shape[1] != con_tok.shape[1]:
            L = min(m.shape[1], con_tok.shape[1])
            con_tok = con_tok[:, :L]
            gap_tok = gap_tok[:, :L]
            m = m[:, :L]
        denom = m.sum(dim=-1).clamp_min(1.0)
        con_avg = (con_tok * m).sum(dim=-1) / denom
        gap_avg = (gap_tok * m).sum(dim=-1) / denom
    else:
        con_avg = con_tok.mean(dim=-1)
        gap_avg = gap_tok.mean(dim=-1)

    # Preserve true contradictions/gaps by making metadata the target, not a
    # softplus wall.  v14.3.5 removes the global-prior fight: if a record says
    # A/B-classical, K/G targets are 0; if it says K/G-heavy, the target rises.
    con_target = torch.clamp(w["K"], 0.0, 0.90)
    gap_target = torch.clamp(w["G"], 0.0, 0.90)
    con_weight = 1.0 + w["K"]
    gap_weight = 1.0 + w["G"]
    loss_b = con_weight * (con_avg - con_target).pow(2) + gap_weight * (gap_avg - gap_target).pow(2)
    stats = {
        "metadata_k_mean": float(w["K"].detach().mean().item()),
        "metadata_g_mean": float(w["G"].detach().mean().item()),
        "predicted_k_mean": float(con_avg.detach().mean().item()),
        "predicted_g_mean": float(gap_avg.detach().mean().item()),
    }
    return loss_b.mean(), stats


def phase_from_semantic_state(
    T: torch.Tensor,
    Fv: torch.Tensor,
    *,
    bilateral_t: Optional[torch.Tensor] = None,
    bilateral_f: Optional[torch.Tensor] = None,
    con_budget: float = 0.12,
    gap_budget: float = 0.20,
    high_glut_threshold: float = 0.55,
    high_gap_threshold: float = 0.55,
) -> str:
    """Shape-invariant phase selection from mean K/G and optional metadata.

    v14.2.2 replaces the old `K > 5 and h == 0` rule. The old rule depended on
    tensor size and rarely fired for removable high-glut states. This rule uses
    mean predicted contradiction/gap mass plus corpus metadata when available.
    """
    with torch.no_grad():
        pred_k = torch.minimum(T.detach().float(), Fv.detach().float()).mean().item()
        pred_g = (1.0 - torch.maximum(T.detach().float(), Fv.detach().float())).mean().item()
        meta_k = 0.0
        meta_g = 0.0
        if bilateral_t is not None and bilateral_f is not None:
            w = paraconsistent_example_weights(bilateral_t.detach().float(), bilateral_f.detach().float())
            meta_k = float(w["K"].mean().item())
            meta_g = float(w["G"].mean().item())

    if max(pred_k, meta_k) >= max(high_glut_threshold, con_budget * 2.0):
        return "Collapse-Resistant Paradox"
    if max(pred_g, meta_g) >= max(high_gap_threshold, gap_budget * 2.0):
        return "Collapse-Resistant Paradox"
    if pred_k < con_budget * 0.5 and pred_g < gap_budget * 0.5 and meta_k < 0.25 and meta_g < 0.25:
        return "Classical"
    return "Active Learning"


def _normalise_corpus_item(item: Any) -> Tuple[str, Optional[float], Optional[float]]:
    """Accept old string corpus items and new metadata-bearing records."""
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("content") or "")
        t = item.get("bilateral_t")
        f = item.get("bilateral_f")
        if t is None or f is None:
            ba = item.get("bilateral_assessment")
            if hasattr(ba, "t") and hasattr(ba, "f"):
                t, f = float(ba.t), float(ba.f)
            elif isinstance(ba, dict):
                t, f = ba.get("t"), ba.get("f")
        try:
            return text, float(t), float(f)
        except Exception:
            return text, None, None
    return str(item), None, None


def train_shadow_step(
    model: ShadowTokenCore,
    optimizer: ShadowOptimizer,
    corpus: Sequence[Any],
    con_budget: float = 0.12,
    gap_budget: float = 0.20,
    lambda_budget: float = 0.05,
    device: str = "cpu",
) -> Tuple[float, str]:
    """Train one step using ShadowHoTT principled loss.

    Corpus items may be strings (legacy path) or dicts carrying `text`,
    `bilateral_t`, and `bilateral_f`. Metadata-bearing items activate the
    v14.2.2 high-glut/gap lane-routing objective.
    """
    encoded = []
    meta_t, meta_f = [], []
    any_meta = False
    for item in corpus:
        txt, t, f = _normalise_corpus_item(item)
        ids = list(txt.encode("utf-8", errors="ignore")[: model.max_len])
        if len(ids) < 2:
            ids = [32, 46]
        encoded.append(ids)
        if t is not None and f is not None:
            any_meta = True
            meta_t.append(float(t))
            meta_f.append(float(f))
        else:
            meta_t.append(0.5)
            meta_f.append(0.5)

    ml = min(model.max_len, max(len(x) for x in encoded))
    xb, yb, mask = [], [], []
    for ids in encoded:
        ids = ids[:ml]
        true_len = len(ids)
        if len(ids) < ml:
            ids = ids + [32] * (ml - len(ids))
        xb.append(ids[:-1])
        yb.append(ids[1:])
        mask.append([1.0 if i < true_len - 1 else 0.0 for i in range(ml - 1)])
    x = torch.tensor(xb, dtype=torch.long, device=device)
    y = torch.tensor(yb, dtype=torch.long, device=device)
    m = torch.tensor(mask, dtype=torch.float, device=device)

    tl, fl, gl = model(x)
    task_loss = F.cross_entropy(tl.reshape(-1, tl.shape[-1]), y.reshape(-1))
    tp, fp = torch.sigmoid(tl), torch.sigmoid(fl)
    sem_loss, _, _, _, _ = semantic_rank_nullity_loss(
        tp, fp, (0.0 if any_meta else con_budget), (0.0 if any_meta else gap_budget)
    )
    ge = (-F.softmax(gl, dim=-1) * F.log_softmax(gl, dim=-1)).sum(dim=-1).mean()
    total = task_loss + (0.0 if any_meta else 0.3) * sem_loss - 0.01 * ge

    bt = bf = None
    if any_meta:
        bt = torch.tensor(meta_t, dtype=torch.float, device=device)
        bf = torch.tensor(meta_f, dtype=torch.float, device=device)
        meta_sem, _ = metadata_weighted_semantic_loss(
            tp, fp, bt, bf, attention_mask=m,
            con_budget=con_budget, gap_budget=gap_budget,
        )
        lane_loss = lane_routing_loss(gl, bt, bf, attention_mask=m)
        lane_match, _lane_match_stats = lane_semantic_matching_loss(
            tp, fp, gl, bt, bf, attention_mask=m,
        )
        total = total + 0.15 * meta_sem + 0.05 * lane_loss + 0.10 * lane_match

    training_phase = phase_from_semantic_state(
        tp.detach(), fp.detach(), bilateral_t=bt, bilateral_f=bf,
        con_budget=con_budget, gap_budget=gap_budget,
    )

    optimizer.zero_grad()
    optimizer.step(total, phase=training_phase)

    return float(total.item()), training_phase
