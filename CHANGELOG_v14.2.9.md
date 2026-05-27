# TOVAH v14.2.9 — Real Hybrid Split Gate

This patch fixes the v14.2.8 hybrid optimizer gate so the AdamW/ShadowHoTT split is a real adaptive weight rather than a fixed 0.5/0.5 blend.

## Fixed

- **Frozen hybrid gate:** v14.2.8 updated logits using `reward * (last_weight - 0.5)`. Since the gate initialized at 0.5/0.5, that update was exactly zero forever.
- **No real ShadowHoTT/AdamW attribution:** v14.2.9 computes first-order proposal quality for each optimizer candidate before mixing.

## Added

- `adamw_score`: normalized first-order descent quality for the AdamW proposal.
- `shadow_score`: normalized first-order descent quality for the ShadowHoTT proposal.
- `hybrid_score_diff`: positive favors AdamW; negative favors ShadowHoTT.
- `hybrid_score_diff_ema`: smoothed proposal advantage.
- `hybrid_gate_advantage`: actual scalar advantage applied to the gate logits.
- `hybrid_reward` and `hybrid_reward_ema`: previous-step loss-improvement stabilizers.

## Mechanism

Each optimizer proposes an update from the same accumulated gradients. For proposal displacement `delta`, the local first-order score is:

```text
score(delta) = - <grad, delta> / (||grad|| ||delta|| + eps)
```

Higher is better. This lets the gate move immediately on the first step without extra forward passes.

## Expected behavior

Metrics should no longer show constant:

```json
"adamw_weight": 0.5,
"shadow_weight": 0.5
```

Instead, weights should vary gradually while respecting the configured floor so both optimizers remain active.
