$ErrorActionPreference = 'Stop'

# Set default arguments
$SEEDS = if ($env:SEEDS) { $env:SEEDS.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) } else { @("42", "17", "1234") }
$DATASETS = if ($env:DATASETS) { $env:DATASETS.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) } else { @("configs/xes3g5m.yaml", "configs/assist2012.yaml") }
$OUTDIR = if ($env:OUTDIR) { $env:OUTDIR } else { "results/q1/ddr_downstream_gkt" }
$MAX_FOLDS = $env:MAX_FOLDS

$OPS = @("edge_drop", "node_drop", "prereq_preserve")
$PS = @("0.10", "0.20", "0.30")
$ANCHOR_OPS = @("edge_drop", "node_drop")
$ANCHOR_PS = @("0.90")

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# Navigate to repo root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir) {
    Set-Location "$ScriptDir\.."
}

New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

$MAX_FOLDS_ARG = @()
if ($MAX_FOLDS) {
    $MAX_FOLDS_ARG += "--max-folds"
    $MAX_FOLDS_ARG += $MAX_FOLDS
}

foreach ($SEED in $SEEDS) {
    $OUT = "$OUTDIR/ddr_downstream_gkt_seed${SEED}.csv"
    foreach ($cfg in $DATASETS) {
        Write-Host "=== seed=$SEED grid: $cfg -> $OUT ==="
        
        $args1 = @("-m", "scripts.ddr_downstream", "--config", $cfg, "--model", "gkt", "--operators") + $OPS + @("--ps") + $PS + @("--experiment-seed", $SEED, "--perturb-seed", $SEED, "--out", $OUT) + $MAX_FOLDS_ARG
        & $PYTHON $args1
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        Write-Host "=== seed=$SEED manipulation-check anchor: $cfg -> $OUT ==="
        $args2 = @("-m", "scripts.ddr_downstream", "--config", $cfg, "--model", "gkt", "--operators") + $ANCHOR_OPS + @("--ps") + $ANCHOR_PS + @("--experiment-seed", $SEED, "--perturb-seed", $SEED, "--out", $OUT) + $MAX_FOLDS_ARG
        & $PYTHON $args2
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

Write-Host "=== per-seed shard summary ==="
foreach ($SEED in $SEEDS) {
    $OUT = "$OUTDIR/ddr_downstream_gkt_seed${SEED}.csv"
    if (Test-Path $OUT) {
        & $PYTHON -c "
import sys, pandas as pd
p = sys.argv[1]
try:
    df = pd.read_csv(p)
    print(p)
    print(df.groupby(['dataset','operator']).size().to_string())
    print('total', len(df), '\n')
except Exception as e:
    print('Error parsing', p, e)
" $OUT
    }
}
