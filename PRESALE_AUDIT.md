# PRESALE_AUDIT.md — v14.2.6 Buyer-Readiness Audit

## Audit scope

This presale audit checked the package as a buyer would receive it:

```text
source-tree integrity
buyer documentation completeness
install/editable packaging path
test collection
targeted regression suites
CLI smoke paths
tiny-corpus training demo
frontier memory-estimate path
claim calibration / overclaim scan
secret/API-key scan
```

## Corrections made during presale audit

The audit found and fixed three buyer-facing issues:

1. **Editable install packaging was broken.**
   - `pip install -e .` installed metadata but did not expose `tovah_v14` as an importable package.
   - Fixed `pyproject.toml` with explicit package mapping: `tovah_v14 = "."` and enumerated subpackages.

2. **Pytest config pointed to the wrong testpath for this source layout.**
   - `testpaths = ["tovah_v14/tests"]` generated a warning when run from the package root.
   - Fixed to `testpaths = ["tests"]` and added marker declarations for future CI splitting.

3. **A smoke test could hang by touching live research/tool orchestration.**
   - `test_research_topic_typed` now stubs tool execution and remains deterministic/fast.

Additional cleanup:

```text
requirements header updated to v14.2.6
pyproject optional extras now include tokenizers via bpe/all
historical v14.2.0 status docs marked as historical
old overclaim wording changed from “full HoTT side” to “HoTT-inspired verifier side”
```

## Verification after corrections

```text
pip install -e . --no-deps: passed
import tovah_v14 from installed editable package: passed
python -m pytest --collect-only -q: 482 tests collected, no testpath warning
fast buyer-facing suite: 138 passed
compileall: passed
all non-test modules import from source parent: passed
run_tovah.py --help: passed
frontier_13b memory-estimate CLI smoke: passed
tiny metadata-bearing CPU pretrain demo: passed
```

Fast buyer-facing suite command:

```bash
python -m pytest \
  tests/test_frontier_readiness_v14_2_4.py \
  tests/test_high_glut_training.py \
  tests/test_hott_core.py \
  tests/test_hott_verifiers.py \
  tests/test_hott_promotion_wiring.py \
  tests/test_scaling.py \
  tests/test_training_pipeline.py \
  tests/test_smoke.py -q
```

Result:

```text
138 passed in 6.11s
```

## Current presale posture

Accurate to say:

```text
Full source and test suite available, with architecture notes, changelogs,
frontier-readiness notes, buyer handoff materials, and a reproducible fast
buyer-facing verification suite.
```

Do not claim:

```text
production-ready autonomous AI
trained 13B frontier model
complete HoTT proof assistant
independent third-party benchmark certification
```

## Remaining buyer diligence items

A serious buyer should still perform:

```text
legal review of license/assignment terms
third-party dependency license review
security audit of tool/patch/credential surfaces
GPU benchmark on target hardware
multi-GPU/FSDP checkpoint save/restore validation
external benchmark/evaluation design
patent/trade-secret strategy review
```


## v14.2.6 formal HoTT update

Added `hott/formal.py`, a bounded formal dependent-type-checking kernel. Buyer language should now say “specialized formal HoTT substrate” rather than the weaker “HoTT-inspired verifier layer,” while still avoiding “complete Lean/Coq-class proof assistant.”
