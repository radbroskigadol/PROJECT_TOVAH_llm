# TOVAH v14.3.4 Frontier Hardening

This patch addresses the frontier-engineer P0/P1 findings against the v14.3.3 loop/support build.

## P0 fixes

### 1. Frontier initialization

`ScalableBilateralCore` now uses GPT/LLaMA-style initialization:

- `nn.Linear` weights: `N(0, 0.02)`
- `nn.Embedding` weights: `N(0, 0.02)`
- residual output projections `o_T`, `o_F`, `down_T`, `down_F`: scaled by `1/sqrt(2*n_blocks)`

This removes the PyTorch `nn.Embedding` default `N(0,1)` failure mode that produced huge untrained logits.

### 2. Propositional/positional bilateral semantics

The frontier model now treats full-vocabulary next-token prediction as one shared classical vocabulary head. Bilateral support is represented by compact per-position semantic heads:

```text
semantic_T : hidden_T -> B x L x 1
semantic_F : hidden_F -> B x L x 1
```

`F_logits` remains available as a diagnostic view of the F hidden stream, but it shares the same vocabulary projection as `T_logits`. It is not the primary source of paraconsistent semantics.

The semantic loss was changed from a hard ceiling against contradiction/gap mass into a calibration objective. Metadata-weighted training now targets real K/G mass instead of forcing classicalization.

### 3. Compact ShadowOptimizer state

`ShadowOptimizer` now stores four persistent buffers per parameter:

```text
m
rms2
K_glut_q   uint8 EMA in [0,1]
R_obs_q    uint8 EMA in [0,1]
```

`M`, `D`, `G`, and collapse pressure are derived when needed. Old checkpoints with `T_sup/F_sup/K_glut/R_obs` are tolerated and compacted on load.

## P1 fixes

### Training repetition penalty removed

The logits concentration penalty is now a compatibility no-op. It was an anti-confidence regularizer that fought next-token CE. Loop control belongs in decode/eval diagnostics or sequence-level verifier rewards.

### Lane D protected

`lane_mixture()` excludes lane D unless `include_d=True`. Lane D remains a forced-totalization diagnostic and no longer participates accidentally in runtime lane mixtures.

### Bilateral OR stabilized

`bilateral_or()` now clamps inputs and accumulates support using a stable probabilistic-OR form, preserving order-insensitivity up to floating-point roundoff for finite inputs.

### Differentiable K/G surrogates

`differentiable_paraconsistent_surrogates()` was added for training-time uses. The old `compute_paraconsistent_invariants()` remains a detached diagnostic.

## Verifier-grounded reward scaffold

`training/formal_hott_rl.py` adds a small Π/Σ/Id task suite and a binary `FormalHoTTChecker` reward:

```python
from tovah_v14.training.formal_hott_rl import smoke_score_suite
print(smoke_score_suite())
```

This is the first grounded reward surface for proof-shaped training. It is intentionally AST-level; parser/model-output integration is a downstream layer.

## QLoRA / DoRA adapter scaffold

`training/bilateral_lora_adapter.py` provides an optional dependency-gated scaffold for starting from an open base model such as `Qwen/Qwen3-7B` instead of random-init frontier pretraining.

## Muon optimizer path

`neural/muon.py` adds a Muon-style wrapper. Frontier profile construction now defaults to `kind="muon"` unless overridden.

AdamW remains available:

```powershell
python .\run_tovah.py --pretrain --profile frontier_dev --optimizer adamw ...
```

Muon can be selected explicitly:

```powershell
python .\run_tovah.py --pretrain --profile frontier_dev --optimizer muon ...
```

## Kernel decomposition scaffold

The monolithic `kernel/kernel.py` remains intact for compatibility, but role modules were added:

```text
kernel/core_runtime.py
kernel/patch_governance.py
kernel/telemetry.py
kernel/ecology.py
kernel/operator_surface.py
```

These are safe migration targets for splitting `ProtozoanKernel` into auditable subsystems without breaking current imports.
