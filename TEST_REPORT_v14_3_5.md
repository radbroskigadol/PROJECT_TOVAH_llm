# TOVAH v14.3.5 Test Report

Validation performed in the sandbox after patching the v14.3.4 project zip.

## Commands run

```bash
python -m compileall -q .
python -m pytest tests/test_v14_3_5_next_update.py -q
python -m pytest tests/test_v14_3_4_frontier_hardening.py::test_muon_factory_steps -q
python run_tovah.py --help
```

## Results

- `compileall`: passed.
- `tests/test_v14_3_5_next_update.py`: 4 passed.
- v14.3.4 Muon compatibility smoke: passed.
- `run_tovah.py --help`: passed.
- Direct smoke:
  - `smoke_score_suite()['reward_mean'] == 1.0`.
  - `embed_F == -embed_T` at initialization for `ScalableBilateralCore`.
  - Muon stats report `optimizer_family='muon_v14_3_5'`, `nesterov=True`, `ns_steps=3`, and matrix LR adjustment.

## Notes

The older full `test_v14_3_4_frontier_hardening.py` suite instantiates and forwards through the CPU `frontier_dev` model, which is slow in this sandbox. The targeted compatibility test for Muon passed; the new v14.3.5 targeted tests passed.
