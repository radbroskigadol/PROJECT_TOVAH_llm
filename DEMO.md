# DEMO.md — Suggested Demonstrations

This file gives demo paths for a technical evaluator. They are designed to show the architecture without requiring a frontier-scale run.

## Demo 1 — Package and CLI smoke

```bash
cd tovah_v14
python -m pip install -e '.[all]'
python run_tovah.py --help
```

What to show:

```text
package installs
CLI exposes live-kernel and pretraining paths
frontier-readiness flags are present
```

## Demo 2 — Frontier memory estimate without model allocation

```bash
python run_tovah.py --pretrain --profile frontier_13b --estimate-frontier-memory
```

What to show:

```text
13B planning surface exists
estimate path exits before allocating model
no claim of completed 13B training is made
```

## Demo 3 — High-glut / high-gap loss tests

```bash
python -m pytest tests/test_high_glut_training.py -q
```

What to show:

```text
K-heavy examples affect lane B / contradiction matching
G-heavy examples affect lane C / gap matching
Lane D is not ordinary training target
AdamWWrapper responds to collapse-resistant phase
```

## Demo 4 — HoTT verifier tests

```bash
python -m pytest tests/test_hott_core.py tests/test_hott_verifiers.py tests/test_hott_promotion_wiring.py -q
```

What to show:

```text
path/transport/verifier layer exists
memory identity and module equivalence checks are tested
patch certification is wired into promotion behavior
```

## Demo 5 — Tiny metadata-bearing training smoke

Create a tiny corpus:

```bash
mkdir -p /tmp/tovah_demo_corpus
cat > /tmp/tovah_demo_corpus/demo.jsonl <<'JSONL'
{"text":"This event is contradictory but informative.","bilateral_t":0.9,"bilateral_f":0.85}
{"text":"This event is underdetermined and should preserve a gap.","bilateral_t":0.1,"bilateral_f":0.1}
{"text":"This event is classically affirmed.","bilateral_t":0.95,"bilateral_f":0.05}
JSONL
```

Run one tiny CPU step:

```bash
python run_tovah.py --pretrain \
  --shard-dir /tmp/tovah_demo_corpus \
  --profile debug \
  --epochs 1 \
  --batch-size 2 \
  --max-steps 1 \
  --device cpu
```

What to show:

```text
metadata-bearing records can feed training
bilateral_t/f are part of the training path
small profile runs without frontier hardware
```

## Demo 6 — Source walkthrough

Recommended 15-minute walkthrough:

```text
core/primitives.py          BilateralValue
core/lanes.py               semantic lanes
neural/training.py          high-glut/high-gap losses
hott/core.py                path/transport/J-like surface
hott/patch_certificates.py  patch certificates
mutation/promotion_ladder.py fail-closed promotion gate
neural/scaling.py           ScalableBilateralCore
training/pretrain.py        frontier semantic mode / checkpoint path
```

## Optional demo recording script

A screen recording can show:

```text
1. Open README.md and ARCHITECTURE.md.
2. Run CLI help.
3. Run frontier memory estimate.
4. Run high-glut tests.
5. Run HoTT tests.
6. Show tiny-corpus one-step training smoke.
7. End on KNOWN_LIMITATIONS.md to demonstrate honest scope control.
```
