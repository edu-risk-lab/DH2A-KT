# A2.2b — Rerun DDR GKT baseline (operator=none) for XES3G5M fold 0, experiment seed 17.
#
# Usage (Windows GPU server, repo root):
#   .\scripts\run_a2_2b_ddr_seed17_baseline.ps1
#   .\scripts\run_a2_2b_ddr_seed17_baseline.ps1 -DryRun
#   .\scripts\run_a2_2b_ddr_seed17_baseline.ps1 -SkipGraphRebuild
#   .\scripts\run_a2_2b_ddr_seed17_baseline.ps1 -MergeOnly

param(
    [switch]$SkipGraphRebuild,
    [switch]$DryRun,
    [switch]$MergeOnly
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir) { Set-Location (Join-Path $ScriptDir '..') }

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$CONFIG_SPLIT = 'configs/xes3g5m_split17.yaml'
$CONFIG_DDR = 'configs/xes3g5m.yaml'
$SEED = 17
$FOLD = 0
$BATCH_SIZE = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { '8' }
$OUT = "results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed${SEED}.csv"
$MERGED = 'results/tables/ddr_downstream.csv'
$LOG_DIR = 'logs/q1'
$WORK_DIR = "results/pykt_work/xes3g5m/fold_${FOLD}_seed_${SEED}/ddr_downstream/gkt"
$PLACEHOLDER_AUC = '0.8422173236026003'

New-Item -ItemType Directory -Force -Path $LOG_DIR, (Split-Path $OUT) | Out-Null
$LOG = Join-Path $LOG_DIR 'a2_2b_ddr_seed17_baseline.log'

function Log([string]$Msg) {
    $line = "[$(Get-Date -Format o)] $Msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

function Check-Env {
    Log '=== Environment check ==='
    if ($DryRun) { Log 'DryRun: skipping CUDA check'; return }
    & $PYTHON -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
    & $PYTHON -c "import pykt"
    if (-not (Test-Path 'data/processed/xes3g5m.parquet')) { throw 'Missing data/processed/xes3g5m.parquet' }
    Log 'Environment OK'
}

function Remove-PlaceholderRow {
    Log "=== Removing placeholder fold-0 none row (AUC=$PLACEHOLDER_AUC) ==="
    if (-not (Test-Path $OUT)) { Log "No existing shard $OUT"; return }
    if ($DryRun) { Log "DryRun: would strip placeholder from $OUT"; return }
    & $PYTHON -c @"
import pandas as pd
from pathlib import Path
path = Path(r'$OUT')
placeholder = float('$PLACEHOLDER_AUC')
df = pd.read_csv(path)
mask = (df['dataset']=='xes3g5m') & (df['model']=='gkt') & (df['fold']==0) & (df['operator']=='none') & (df['p']==0.0)
removed = int(mask.sum())
if removed:
    df = df.loc[~mask]
    df.to_csv(path, index=False)
    print(f'Removed {removed} row(s) from {path}')
else:
    print('No fold-0 none row to remove')
"@
}

function Invoke-PythonModuleLogged {
    param(
        [string[]]$PythonArgs,
        [string]$LogPath
    )
    $argStr = ($PythonArgs | ForEach-Object { if ($_ -match '\s') { """$_""" } else { $_ } }) -join ' '
    $cmd = """$PYTHON"" $argStr 2>&1"
    cmd /c $cmd | Tee-Object -Append -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Rebuild-Graph {
    Log "=== Rebuild train-only graphs for split seed $SEED ==="
    if ($DryRun) { Log "DryRun: graph_builder $CONFIG_SPLIT"; return }
    Invoke-PythonModuleLogged -PythonArgs @('-m', 'src.graph_builder', '--config', $CONFIG_SPLIT) `
        -LogPath (Join-Path $LOG_DIR "a2_2b_graph_builder_seed${SEED}.log")
}

function Clear-Workdir {
    Log "=== Clear stale DDR workdir ==="
    if ($DryRun) { Log "DryRun: would remove $WORK_DIR"; return }
    if (Test-Path $WORK_DIR) {
        Remove-Item -Recurse -Force $WORK_DIR
        Log "Removed $WORK_DIR"
    }
}

function Run-Baseline {
    Log "=== Train GKT baseline only: fold=$FOLD seed=$SEED batch=$BATCH_SIZE ==="
    $PythonArgs = @(
        '-m', 'scripts.ddr_downstream',
        '--config', $CONFIG_DDR,
        '--model', 'gkt',
        '--baseline-only',
        '--only-fold', "$FOLD",
        '--experiment-seed', "$SEED",
        '--perturb-seed', "$SEED",
        '--batch-size', "$BATCH_SIZE",
        '--out', $OUT,
        '--log-level', 'INFO'
    )
    if ($DryRun) { Log ("DryRun: $PYTHON " + ($PythonArgs -join ' ')); return }
    Invoke-PythonModuleLogged -PythonArgs $PythonArgs -LogPath $LOG
}

function Verify-Result {
    Log '=== Verify new AUC !== placeholder ==='
    if ($DryRun) { return }
    & $PYTHON -c @"
import pandas as pd
df = pd.read_csv(r'$OUT')
row = df[(df['dataset']=='xes3g5m')&(df['model']=='gkt')&(df['fold']==0)&(df['operator']=='none')&(df['p']==0.0)]
if row.empty: raise SystemExit('ERROR: fold-0 none row missing')
auc = float(row['auc'].iloc[0])
placeholder = float('$PLACEHOLDER_AUC')
if abs(auc - placeholder) < 1e-6:
    raise SystemExit(f'ERROR: AUC still placeholder ({auc})')
print(f'OK: fold-0 none AUC = {auc:.6f}')
"@
}

function Merge-Tables {
    Log "=== Merge into $MERGED ==="
    if ($DryRun) { Log 'DryRun: merge + plot + tex'; return }
    & $PYTHON -m scripts.merge_ddr_downstream --base $MERGED --append $OUT --out $MERGED
    & $PYTHON -m scripts.plot_ddr_downstream
    & $PYTHON -m scripts.generate_ddr_downstream_gkt_tex
    Log 'Merged tables refreshed'
}

Log '=== A2.2b DDR seed-17 baseline rerun start ==='

if ($MergeOnly) {
    Merge-Tables
    Log '=== Done (merge only) ==='
    exit 0
}

Check-Env
Remove-PlaceholderRow
if (-not $SkipGraphRebuild) { Rebuild-Graph }
Clear-Workdir
Run-Baseline
Verify-Result
Merge-Tables
Log '=== A2.2b finished ==='
