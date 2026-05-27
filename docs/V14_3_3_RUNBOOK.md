# TOVAH v14.3.3 Runbook

## 1. Apply patch

```powershell
cd C:\Users\akiva\Downloads\TOVAH\tovah_v14_3_2a_eval_cli_hardened\tovah_v14
python C:\path\to\tovah_v14_3_3_loop_support_patch\apply_v14_3_3_patch.py --project-root .
```

## 2. Compile/check helper files

```powershell
python -m compileall training tools tests
python -m pytest .\tests\test_v14_3_3_helpers.py -q
```

## 3. Check existing 48/96/160 probe files

```powershell
python .\tools\check_probe_outputs_v14_3_3.py `
  --runs-dir .\runs `
  --pattern "tovah_v14_3_2a_real_mixed_generated_probe_128_tok{tok}.json" `
  --tokens 48 96 160
```

Expected interpretation:

- `complete`: good file; parseable generated-continuation metrics.
- `missing`: that token length never wrote an output.
- `empty`: process created a file but did not write useful output.
- `invalid_json_or_partial_write`: process probably interrupted during JSON write.

## 4. Rerun incomplete token lengths safely

```powershell
python .\tools\run_shadow_probe_lengths_v14_3_3.py `
  --checkpoint .\checkpoints\tovah_v14_3_2a_real_mixed_uapshadow_bpe_2500.pt `
  --shard-dir .\tovah_corpus\paradox_real_mixed_v14_3_2a `
  --tokenizer .\tokenizer_real_mixed.json `
  --out-template ".\runs\tovah_v14_3_3_real_mixed_generated_probe_128_tok{tok}.json" `
  --tokens 96 160 `
  --max-examples-shadow-model 128 `
  --temperature 0.0 `
  --resume
```

If CPU time is painful, first reduce samples:

```powershell
python .\tools\run_shadow_probe_lengths_v14_3_3.py `
  --checkpoint .\checkpoints\tovah_v14_3_2a_real_mixed_uapshadow_bpe_2500.pt `
  --shard-dir .\tovah_corpus\paradox_real_mixed_v14_3_2a `
  --tokenizer .\tokenizer_real_mixed.json `
  --out-template ".\runs\tovah_v14_3_3_smoke_probe_32_tok{tok}.json" `
  --tokens 96 160 `
  --max-examples-shadow-model 32 `
  --temperature 0.0 `
  --resume
```

## 5. Create held-out/adversarial split

```powershell
python .\tools\generate_heldout_adversarial_split_v14_3_3.py `
  --in-dir .\tovah_corpus\paradox_real_mixed_v14_3_2a `
  --out-dir .\tovah_corpus\paradox_real_mixed_v14_3_3_family_holdout `
  --copy-tokenizer .\tokenizer_real_mixed.json
```

Then train against the new out-dir once profile/loop auxiliary terms are wired into the training loop.

## 6. Keep the known-good optimizer settings

```powershell
--uap-classical-floor 0.15 `
--uap-classical-ceiling 0.85 `
--uap-geometry-lr 0.01 `
--uap-max-update-rms 1.0 `
--uap-trust-clip 0.0 `
--uap-aux-weight 0.05
```

Do not change these until the 48/96/160 generated-continuation failure mode is isolated.
