# TOVAH v14.3.1 — ShadowHoTT Autonomous AI Kernel


## v14.3.4 Frontier Hardening

This build includes the v14.3.4 frontier-hardening pass: safe scalable-model initialization, shared vocabulary projection with per-token bilateral semantic supports, compact four-buffer ShadowOptimizer state, Muon optimizer support, FormalHoTT verifier-reward scaffolding, and QLoRA/DoRA adapter scaffolding. See `docs/V14_3_4_FRONTIER_HARDENING.md`.


TOVAH is a research-grade autonomous AI kernel built around **bilateral paraconsistent semantics** and a **specialized formal HoTT substrate** for identity-preserving transformation.

It is not a prompt pack, LangChain wrapper, or finished frontier foundation model. It is a source-available research prototype containing:

- a persistent autonomous kernel runtime,
- bilateral truth/falsity state machinery,
- four semantic lanes for classical, paraconsistent, paracomplete, and totalized interpretations,
- contradiction-preserving memory and corpus generation,
- formal HoTT-style identity/transport/J/coherence machinery,
- patch governance and promotion gates,
- a byte-level bilateral transformer research core,
- frontier-model scaffolding for 2B/7B/13B-class experiments,
- targeted regression tests covering the currently validated surfaces.

## Core idea

Most neural systems collapse belief into a single likelihood/confidence value. TOVAH keeps **truth support** and **falsity support** independent:

```text
T high, F low   -> affirmed / true-like
T low, F high   -> rejected / false-like
T high, F high  -> glut / contradiction
T low, F low    -> gap / underdetermination
```

The system carries that bilateral state through runtime state, memory, training data, neural outputs, semantic lane routing, patch promotion, and formal HoTT-style verification.

## Current release status

This package is **v14.3.1**, the frontier-readiness/documented handoff package.

What is implemented:

- full Python source tree,
- test suite,
- closed-loop corpus exporter,
- specialized formal HoTT substrate with a bounded dependent type checker,
- patch-promotion ladder with fail-closed HoTT certification,
- metadata-aware high-glut / high-gap training loss,
- explicit lane B/C semantic matching losses,
- scalable bilateral transformer profiles,
- hidden-state semantic heads for frontier mode,
- FSDP/DDP scaffolding,
- resumable checkpoint surfaces,
- buyer-facing documentation and handoff notes.

What is not claimed:

- no trained frontier weights are included,
- no 13B run has been completed or certified,
- no production safety certification is claimed,
- no independent third-party benchmark is included.

## Repository layout

```text
tovah_v14/
  core/          bilateral primitives, lanes, runtime state, cache
  hott/          formal HoTT checker plus paths, transport, patch/memory/module/obstruction verifiers
  kernel/        autonomous kernel, hub/subkernel ecology, packet runtime
  memory/        conflict-preserving memory and provenance
  mutation/      patch staging, quarantine, promotion ladder
  neural/        byte-level and scalable bilateral transformer components
  training/      corpus export, pretraining entry points, semantic losses
  tools/         tool contracts and builtin tool surfaces
  tests/         regression and integration tests
```

## Quick start

See [`INSTALL.md`](INSTALL.md) and [`QUICKSTART.md`](QUICKSTART.md).

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md), [`FORMAL_HOTT.md`](FORMAL_HOTT.md), and [`BUYER_TECHNICAL_SUMMARY.md`](BUYER_TECHNICAL_SUMMARY.md).

## Buyer handoff

See [`HANDOFF.md`](HANDOFF.md), [`PRESALE_AUDIT.md`](PRESALE_AUDIT.md), [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md), [`TEST_REPORT.md`](TEST_REPORT.md), and [`LICENSE_OPTIONS.md`](LICENSE_OPTIONS.md).

## v14.3.1 scale handoff

v14.3.1 adds buyer execution materials: `SCALE_HANDOFF.md`, `SCALE_READINESS.md`, `SCALING_LADDER.md`, `FSDP_RUNBOOK.md`, reference configs, scripts, a lightweight eval harness, and security runbooks. Use these materials to evaluate scale readiness before any 7B/13B training attempt.


## v14.3.2 Shadow-depth update

v14.3.2 shifts evaluation from optimizer horse-racing to UAP token ontology. AdamW is treated as the classicalized floor/projection; ShadowOptimizer is evaluated by whether it preserves contradiction, gap, obstruction residue, collapse pressure, local-to-global noncollapse, and classicalization depth.

Generate the new corpus:

```powershell
python .\tools\generate_uap_shadow_corpus.py --out .\tovah_corpus\uap_shadow_depth_v14_3_2 --n 20000 --shard-size 5000
```

Run a short training pass with the auxiliary UAP profile objective:

```powershell
python .\run_tovah.py --pretrain --shard-dir .\tovah_corpus\uap_shadow_depth_v14_3_2 --profile heavy --optimizer shadow --tokenizer bpe --train-bpe-if-missing --bpe-save-path .\tokenizer.json --epochs 1 --batch-size 1 --grad-accum-steps 4 --max-steps 100 --device cpu --uap-aux-weight 0.05 --save-path .\checkpoints\tovah_v14_3_2_heavy_uapshadow_bpe_0100.pt --metrics-path .\runs\tovah_v14_3_2_heavy_uapshadow_bpe_0100_metrics.jsonl
```

Run Shadow-depth eval:

```powershell
python .\training\shadow_depth_eval.py .\tovah_corpus\uap_shadow_depth_v14_3_2 --out .\runs\tovah_v14_3_2_shadow_depth_eval.json
```


## v14.3.3 loop/support hardening

The v14.3.3 update adds generated-continuation loop-stability diagnostics, support-profile hardening helpers, stalled-probe check/rerun tools, and held-out/adversarial family split tooling. See `docs/V14_3_3_LOOP_SUPPORT_HARDENING.md` and `docs/V14_3_3_RUNBOOK.md`.

## v14.3.5 verifier/locality hardening

v14.3.5 adds verifier-grounded FormalHoTT reward scaffolding, metadata-faithful K/G semantic losses, bilateral negation initialization, optional Belnap MoE FFN routing, differentiable sheaf-obstruction regularization, predictive-coding auxiliaries, Bilateral DoRA adapters, and a bilateral SSM prototype. Defaults remain conservative: the new sheaf loss is opt-in with `--uap-sheaf-weight`, and the MoE path is enabled only with `ffn_kind="belnap_moe"`.
