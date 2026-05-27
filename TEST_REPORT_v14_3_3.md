# TOVAH v14.3.3 Patch Test Report

## Environment

Patched from `tovah_v14_3_2a_eval_cli_hardened(1).zip` into a complete replacement project zip.

## Checks run

```text
python -m compileall -q training tools tests run_tovah.py
python -m pytest tests/test_v14_3_3_helpers.py -q
python training/eval.py --help
python run_tovah.py --help
python training/shadow_depth_eval.py /tmp/tovah_smoke/smoke.jsonl
```

## Results

```text
pytest: 3 passed
training/eval.py --help: OK
run_tovah.py --help: OK; --uap-loop-penalty-weight is exposed
shadow_depth_eval smoke: schema_version=tovah-shadow-depth-eval-v14.3.3 and loop_stability_v14_3_3 emitted
```

## Scope

No long training/eval run was executed in this environment because the checkpoint/corpus runtime path from the user's Windows machine is not mounted here. The patch is therefore validated for syntax, importability, helper behavior, CLI exposure, and Shadow-depth smoke output.
