# HANDOFF.md — Buyer / Maintainer Handoff

This document is a practical handoff checklist for a technical buyer, maintainer, or evaluator.

## Package contents

The package includes:

```text
full Python source
tests
changelogs
architecture documentation
installation guide
quickstart
demo instructions
known limitations
test report
presale audit report
license-options memo
```

## Recommended first 60 minutes

1. Read:

```text
README.md
BUYER_TECHNICAL_SUMMARY.md
PRESALE_AUDIT.md
KNOWN_LIMITATIONS.md
TEST_REPORT.md
```

2. Install locally:

```bash
cd tovah_v14
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

3. Run smoke checks:

```bash
python run_tovah.py --help
python run_tovah.py --pretrain --profile frontier_13b --estimate-frontier-memory
python -m pytest tests/test_frontier_readiness_v14_2_4.py -q
python -m pytest tests/test_high_glut_training.py -q
python -m pytest tests/test_hott_core.py tests/test_hott_verifiers.py -q
```

## Recommended technical review path

Review in this order:

```text
1. core/primitives.py
2. core/lanes.py
3. neural/training.py
4. neural/scaling.py
5. training/pretrain.py
6. hott/core.py
7. hott/patch_certificates.py
8. mutation/promotion_ladder.py
9. memory/conflict.py and hott/memory_identity.py
10. kernel/kernel.py
```

This gives the evaluator the shortest path from core semantic idea to actual runtime/training behavior.

## What to verify before acquisition or exclusive license

A serious buyer should independently verify:

```text
source ownership and provenance
license status of dependencies
ability to run targeted test suite
ability to reproduce test collection count
absence of hidden external service dependency for local tests
GPU memory estimates for intended frontier profile
model construction for small/dev profiles
checkpoint save/resume on local machine
behavior of HoTT fail-closed promotion gate
behavior of Lane B/C high-glut/high-gap semantic matching
```

## Recommended next engineering milestones

1. CI configuration for targeted fast suites.
2. Full-test split into fast, slow, autonomy, and research markers.
3. A reproducible tiny-corpus training demo.
4. A GPU benchmark notebook for `tiny`, `small`, and frontier-dev profiles.
5. Sharded checkpoint end-to-end test on multi-GPU hardware.
6. Tensor/pipeline parallel strategy decision for 7B/13B.
7. Independent security review of tool and patch surfaces.
8. External technical audit of the HoTT-inspired verifier semantics.

## Handoff caveats

The project is a sophisticated research prototype. It should not be treated as a production-safe autonomous agent without additional containment, audit, and deployment hardening.

No trained frontier weights are included. The value is in the architecture, source, verifier logic, training scaffolding, and research direction.

## Suggested buyer-facing phrasing

Accurate:

```text
Full source and test suite available, with architecture notes, changelogs, frontier-readiness notes, and buyer handoff materials.
```

Avoid unless independently proven:

```text
Fully production-ready.
13B-trained.
Complete HoTT proof assistant.
Certified safe autonomous AI.
```

## v14.2.6 scale handoff addendum

For buyer scale execution, start with `SCALE_HANDOFF.md`. It indexes the scale ladder, FSDP runbook, tokenizer/data strategy, eval harness, and security posture.
