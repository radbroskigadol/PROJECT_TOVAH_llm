param(
  [string]$ProjectRoot = ".",
  [int]$N = 512
)
$ErrorActionPreference = "Stop"
Push-Location $ProjectRoot
try {
  python .\tools\generate_uap_shadow_corpus.py --out .\tovah_corpus\uap_shadow_depth_v14_3_2_smoke --n $N --shard-size 128
  python .\training\shadow_depth_eval.py .\tovah_corpus\uap_shadow_depth_v14_3_2_smoke --out .\runs\tovah_v14_3_2_shadow_depth_smoke_eval.json
  if (Test-Path .\tokenizer.json) {
    python .\tools\tokenizer_shadow_diagnostics.py .\tovah_corpus\uap_shadow_depth_v14_3_2_smoke --tokenizer .\tokenizer.json --out .\runs\tovah_v14_3_2_tokenizer_shadow_diag_smoke.json
  } else {
    python .\tools\tokenizer_shadow_diagnostics.py .\tovah_corpus\uap_shadow_depth_v14_3_2_smoke --out .\runs\tovah_v14_3_2_tokenizer_shadow_diag_smoke.json
  }
  python -m pytest .\tests\test_v14_3_2_shadow_depth.py -q
} finally {
  Pop-Location
}
