# TEST REPORT v14.3.4

Environment: sandbox CPU, Python 3.13, PyTorch available.

Commands run:

```text
PYTHONPATH=/mnt/data/tovah_work_v1433b python -m compileall -q tovah_v14
# result: compile_ok

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/mnt/data/tovah_work_v1433b pytest -q tovah_v14/tests/test_frontier_readiness_v14_2_4.py
# result: 5 passed in 10.05s

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/mnt/data/tovah_work_v1433b pytest -q tovah_v14/tests/test_scaling.py::TestScalableBilateralCore
# result: 5 passed in 19.05s

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/mnt/data/tovah_work_v1433b pytest -q tovah_v14/tests/test_scaling.py::TestFrontierPretrainWiring
# result: 1 passed in 7.80s

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/mnt/data/tovah_work_v1433b pytest -q tovah_v14/tests/test_v14_2_8_training_speed_fixes.py::test_shadow_optimizer_state_restores_by_parameter_order tovah_v14/tests/test_v14_2_8_training_speed_fixes.py::test_uap_shadow_optimizer_exposes_adamw_classicalization_geometry
# result: 2 passed in 2.84s
```

Additional direct smoke:

```text
ScalableBilateralCore frontier_dev initialized embed_T.std ~= 0.0200 and untrained |logits|.mean ~= 0.38 on a small random batch.
```

Note: the full historical test suite was not run end-to-end in this sandbox. Some chained pytest invocations printed passing summaries but did not return cleanly before the sandbox timeout, so this report only claims the targeted validations above.
