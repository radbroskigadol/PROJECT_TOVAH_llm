# TEST_REPORT.md — TOVAH v14.2.6

## Summary

The latest presale verification pass reported:

```text
pip install -e . --no-deps: passed
installed editable import of tovah_v14: passed
compileall: passed
all non-test modules import: passed
pytest collection: 482 tests collected, no testpath warning
fast buyer-facing suite: 138 passed
run_tovah.py --help: passed
frontier_13b memory-estimate CLI smoke: passed
tiny metadata-bearing CPU pretrain demo: passed
```

A full monolithic pytest run was not claimed because long-running autonomy/kernel research paths exceeded lightweight sandbox time limits.

## Targeted suites of interest

### Frontier readiness

```bash
python -m pytest tests/test_frontier_readiness_v14_2_4.py -q
```

Covers:

```text
hidden-state semantic heads
frontier semantic mode behavior
memory-estimate path
checkpoint/optimizer surfaces
CLI-adjacent frontier-readiness behavior
```

### High-glut / high-gap semantic training

```bash
python -m pytest tests/test_high_glut_training.py -q
```

Covers:

```text
metadata-aware loss
K/G phase detection
Lane B contradiction matching
Lane C gap matching
Lane D exclusion from ordinary training
AdamW collapse-resistant behavior
```

### HoTT verifier layer

```bash
python -m pytest \
  tests/test_hott_core.py \
  tests/test_hott_verifiers.py \
  tests/test_hott_promotion_wiring.py -q
```

Covers:

```text
path/transport/J-like primitives
patch certificate checks
memory identity checks
module equivalence checks
obstruction utilities
promotion wiring
fail-closed behavior
```

### Scaling/neural/training checks

```bash
python -m pytest \
  tests/test_scaling.py \
  tests/test_neural.py \
  tests/test_training_pipeline.py -q
```

Covers:

```text
scalable model construction
RoPE/GQA regression surfaces
training pipeline behavior
semantic loss surfaces
```

## Recommended CI split

The current test suite should be marked and split before commercial deployment:

```text
fast: deterministic unit tests, no network, no long kernel loop
slow: integration tests and longer kernel workflows
gpu: CUDA-dependent training/throughput tests
distributed: DDP/FSDP/multi-process tests
autonomy: live kernel/autonomous research-loop tests
security: patch/tool boundary tests
```

Suggested default buyer-facing verification command:

```bash
python -m pytest tests/test_frontier_readiness_v14_2_4.py \
                 tests/test_high_glut_training.py \
                 tests/test_hott_core.py \
                 tests/test_hott_verifiers.py \
                 tests/test_hott_promotion_wiring.py \
                 tests/test_scaling.py \
                 tests/test_training_pipeline.py \
                 tests/test_smoke.py -q
```

Latest presale result:

```text
138 passed in 6.11s
```

## Reproducibility notes

For CPU-only environments, set thread caps before running broader suites:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
```

## Current confidence statement

The changed and high-value surfaces have targeted passing tests. A buyer should independently reproduce the targeted suites and then decide whether to fund broader slow/GPU/distributed validation.


## v14.2.6 additional formal HoTT checker verification

```text
formal HoTT checker suite: 9 passed
```

Covered surfaces: universe typing, Π/lambda checking, beta reduction, Σ pairs/projections, Id/refl, J computation on refl, J endpoint rejection, transparent global definitions, alpha-equivalence, and bad-application rejection.

## v14.2.6 scale-handoff verification

Added buyer-facing scale docs/scripts/configs/evals plus tests for scale ladder, eval harness, checkpoint manifests, metrics logging, and CLI surface. Full monolithic pytest may remain long-running; targeted scale-handoff tests are intended for presale validation.
