# QUICKSTART.md — TOVAH v14.2.6

This quickstart is intended to verify that the package imports, that tests collect, and that the frontier planning CLI works. It does not train a frontier model.

## 1. Install

```bash
cd tovah_v14
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

## 2. Confirm package import

```bash
python - <<'PY'
import tovah_v14
print(tovah_v14.__version__)
PY
```

Expected:

```text
14.2.6
```

## 3. Run CLI help

```bash
python run_tovah.py --help
```

The help output should include frontier-readiness flags:

```text
--frontier-semantic-mode auto|hidden|logits
--fsdp-mixed-precision fp32|bf16|fp16
--resume-from PATH
--save-sharded
--estimate-frontier-memory
```

## 4. Run frontier memory estimate

```bash
python run_tovah.py --pretrain --profile frontier_13b --estimate-frontier-memory
```

This path performs a planning estimate and exits before model allocation.

## 5. Run targeted tests

```bash
python -m pytest tests/test_frontier_readiness_v14_2_4.py -q
python -m pytest tests/test_high_glut_training.py -q
python -m pytest tests/test_hott_core.py tests/test_hott_verifiers.py -q
```

## 6. Inspect main architecture entry points

Good first files:

```text
core/primitives.py                BilateralValue and core T/F state
core/lanes.py                     four semantic lane projections
hott/core.py                      Type, Id, Path, refl, transport, J-like eliminator
hott/patch_certificates.py        patch transport/certification witnesses
hott/memory_identity.py           memory identity/conflict witness logic
hott/module_equivalence.py        module substitutability verifier
hott/obstruction.py               local-to-global obstruction utilities
neural/shadow_model.py            byte-level bilateral transformer
neural/scaling.py                 scalable bilateral transformer profiles
neural/training.py                semantic losses and live training helpers
training/pretrain.py              pretraining entry point
mutation/promotion_ladder.py      patch promotion gates
kernel/kernel.py                  main autonomous runtime
```

## 7. Safe first training smoke

Use a tiny/local corpus. Do not start with `frontier_13b`.

```bash
mkdir -p /tmp/tovah_corpus
cat > /tmp/tovah_corpus/demo.jsonl <<'JSONL'
{"text":"TOVAH preserves contradiction as structured evidence.","bilateral_t":0.9,"bilateral_f":0.8}
{"text":"A clean affirmed example routes classically.","bilateral_t":0.95,"bilateral_f":0.05}
JSONL

python run_tovah.py --pretrain \
  --shard-dir /tmp/tovah_corpus \
  --profile debug \
  --epochs 1 \
  --batch-size 2 \
  --max-steps 1 \
  --device cpu
```

This verifies the training entry point without committing to a serious run.
