# Run 128 x 100-step hybrid AdamW/ShadowOptimizer chunks.
# Usage from tovah_v14 project root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_hybrid_128.ps1

param(
  [string]$ShardDir = ".\tovah_corpus\mixed_step2",
  [string]$Profile = "heavy",
  [int]$Runs = 128,
  [int]$StepsPerRun = 100,
  [int]$BatchSize = 1,
  [int]$GradAccumSteps = 1,
  [string]$Device = "cpu"
)

New-Item -ItemType Directory -Force -Path .\checkpoints | Out-Null
New-Item -ItemType Directory -Force -Path .\runs | Out-Null

$prev = ""
for ($i = 1; $i -le $Runs; $i++) {
  $tag = "{0:D4}" -f $i
  $save = ".\checkpoints\tovah_${Profile}_hybrid_${tag}.pt"
  $metrics = ".\runs\tovah_${Profile}_hybrid_${tag}_metrics.jsonl"
  Write-Host "=== hybrid run $i / $Runs => $save ==="
  $args = @(
    ".\run_tovah.py", "--pretrain",
    "--shard-dir", $ShardDir,
    "--profile", $Profile,
    "--optimizer", "hybrid",
    "--epochs", "1",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--max-steps", "$StepsPerRun",
    "--device", $Device,
    "--save-path", $save,
    "--metrics-path", $metrics,
    "--log-every", "50"
  )
  if ($prev -ne "") { $args += @("--resume-from", $prev) }
  python @args
  if ($LASTEXITCODE -ne 0) { throw "hybrid run $i failed with exit code $LASTEXITCODE" }
  $prev = $save
}
