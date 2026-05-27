# UAP ShadowOptimizer v14.3.1

v14.3.1 turns `ShadowOptimizer` into a UAP/ShadowHoTT optimizer with AdamW-like optimizer hygiene treated as a **classicalization** of a richer bilateral update, not as an external replacement.

## Mathematical loyalty map

| UAP / ShadowHoTT idea | Optimizer object |
|---|---|
| Truth-support lane | `T_sup` |
| Falsity-support lane | `F_sup` |
| Manifestation order | `M_sup = T_sup + F_sup` |
| Determination contrast | `D_det = T_sup - F_sup` |
| Glut aperture | `K_glut = min(T_sup, F_sup)` |
| Gap aperture | `G_gap = exp(-(T_sup + F_sup))` |
| Obstruction residue | `R_obs`, current descent conflicting with accumulated determination support |
| Collapse pressure | `C_collapse`, conflict concentrated on high-glut coordinates |
| Classicalization | bias-corrected AdamW-style first/second moment projection |
| ShadowHoTT correction | damped determination support, residue-aware and gap/glut-aware |
| Neuronal balance across geometries | 8-float `UAPGeometryGate` |

## AdamW-class benefits now present inside ShadowOptimizer

- First moment momentum: `m = beta1*m + (1-beta1)*g`
- Second moment scaling: `rms2 = beta2*rms2 + (1-beta2)*g^2`
- Bias correction for first, second, and bilateral support moments
- Decoupled weight decay
- Global gradient clipping
- Stable update magnitude through trust-ratio / update-norm control
- Warmup + cosine schedule already supported by the v14 training loop
- Classicalization floor via `uap_classical_weight`, bounded by `--uap-classical-floor` and `--uap-classical-ceiling`

## Why this is not merely AdamW

AdamW is the classical projection:

```text
classical_step = -m_hat / (sqrt(v_hat) + eps)
```

The ShadowHoTT correction is computed from the bilateral support geometry:

```text
D = T - F
K = min(T, F)
G = exp(-(T + F))
R = obstruction residue
shadow_step = D_hat / sqrt(v_hat) * damping(K, G, R) + small classical anchor
```

The applied update is:

```text
update = c * classical_step + s * shadow_step
```

where `c` and `s` are learned by the UAP geometry gate. The classical term is therefore a floor/fallback/classicalization, not the ontology of the optimizer.

## New CLI controls

```powershell
--uap-classical-floor 0.15
--uap-classical-ceiling 0.85
--uap-geometry-lr 0.01
--uap-weight-decay 0.1
--hybrid-gate-lr 0.02
--hybrid-min-adamw-weight 0.15
```

Use these flags inside the `python .\run_tovah.py ...` command, not as standalone PowerShell commands.

## Metrics to watch

For `--optimizer shadow`:

- `uap_classical_weight`
- `uap_shadow_weight`
- `uap_obstruction`
- `uap_obstruction_ema`
- `uap_residue_mass`
- `uap_collapse_pressure`
- `uap_trust_ratio_mean`

For `--optimizer hybrid`, the external AdamW/UAPShadow split is still logged as:

- `adamw_weight`
- `shadow_weight`
- `adamw_score`
- `shadow_score`

and the internal UAPShadow geometry is logged as:

- `shadow_uap_classical_weight`
- `shadow_uap_shadow_weight`
- `shadow_uap_obstruction`
- `shadow_uap_residue_mass`
- `shadow_uap_collapse_pressure`
