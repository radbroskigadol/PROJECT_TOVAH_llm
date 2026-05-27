# INSTALL.md — TOVAH v14.2.6

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Linux/macOS/Windows supported at the Python package level
- CUDA recommended for neural training experiments

The package is intentionally lightweight. Core dependencies are declared in `pyproject.toml` and `requirements.txt`.

## Basic local install

From the parent directory containing `tovah_v14/`:

```bash
cd tovah_v14
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

If shell quoting is awkward on your platform:

```bash
python -m pip install -e .
python -m pip install pytest python-dotenv pypdf openai
```

## Minimal install

```bash
cd tovah_v14
python -m pip install -e .
```

This installs the core package and PyTorch dependency. Optional extras are only needed for tests, environment loading, advisor APIs, or PDF utilities.

## Environment variables

The live kernel can optionally use advisor APIs. No API key is required for static tests or local import checks.

Optional:

```bash
export GROK_API_KEY='...'
```

The launcher sets conservative CPU thread caps by default to reduce local startup stalls:

```text
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
VECLIB_MAXIMUM_THREADS=2
NUMEXPR_NUM_THREADS=2
TOKENIZERS_PARALLELISM=false
```

## Run tests

Targeted fast suite example:

```bash
python -m pytest \
  tests/test_frontier_readiness_v14_2_4.py \
  tests/test_high_glut_training.py \
  tests/test_hott_core.py \
  tests/test_hott_verifiers.py \
  tests/test_hott_promotion_wiring.py \
  tests/test_scaling.py \
  tests/test_training_pipeline.py -q
```

Full collection:

```bash
python -m pytest --collect-only -q
```

Full monolithic test execution may take longer than lightweight CI/sandbox limits because several autonomy/research paths perform live kernel work.

## CLI smoke tests

```bash
python run_tovah.py --help
python run_tovah.py --pretrain --profile frontier_13b --estimate-frontier-memory
```

The memory-estimate path does not allocate a 13B model. It prints a planning estimate and exits.

## Development note

For CPU-only development, keep tests targeted. For GPU training, use the smallest profiles first and scale progressively.
