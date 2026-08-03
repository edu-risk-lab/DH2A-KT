# A3.5 — Optional GPU rerun: cold-start strata with full pyKT predictions (no 5000 cap).
#
# Usage (Windows GPU server, repo root):
#   .\scripts\run_a3_cold_start_rerun.ps1
#   .\scripts\run_a3_cold_start_rerun.ps1 -Dataset xes3g5m
#   .\scripts\run_a3_cold_start_rerun.ps1 -Dataset all
#   .\scripts\run_a3_cold_start_rerun.ps1 -FoldIdx 0
#   .\scripts\run_a3_cold_start_rerun.ps1 -DryRun
#   .\scripts\run_a3_cold_start_rerun.ps1 -TablesOnly
#   $env:FORCE_CPU='1'; .\scripts\run_a3_cold_start_rerun.ps1

param(
    [ValidateSet('junyi', 'xes3g5m', 'assist2012', 'all')]
    [string]$Dataset = 'junyi',
    [int]$FoldIdx = -1,
    [string]$Models = '',
    [switch]$DryRun,
    [switch]$TablesOnly,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir) { Set-Location (Join-Path $ScriptDir '..') }

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$LOG_DIR = 'logs/q1'
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG = Join-Path $LOG_DIR 'a3_cold_start_rerun.log'

function Log([string]$Msg) {
    $line = "[$(Get-Date -Format o)] $Msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

function Get-ConfigFor([string]$ds) {
    switch ($ds) {
        'junyi' { return 'configs/junyi.yaml' }
        'xes3g5m' { return 'configs/xes3g5m.yaml' }
        'assist2012' { return 'configs/assist2012.yaml' }
        default { throw "Unknown dataset: $ds" }
    }
}

function Get-ParquetFor([string]$ds) {
    return "data/processed/${ds}.parquet"
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

function Test-Preflight {
    Log "=== Preflight (dataset=$Dataset) ==="
    & $PYTHON -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    & $PYTHON -c "import pykt"
    if ($Dataset -eq 'all') {
        foreach ($ds in @('junyi', 'xes3g5m', 'assist2012')) {
            if (-not (Test-Path (Get-ParquetFor $ds))) { throw "Missing $(Get-ParquetFor $ds)" }
        }
    } else {
        if (-not (Test-Path (Get-ParquetFor $Dataset))) { throw "Missing $(Get-ParquetFor $Dataset)" }
        $cfg = Get-ConfigFor $Dataset
        if (-not (Test-Path $cfg)) { throw "Missing $cfg" }
    }
    Log 'Environment OK'
}

function Clear-PredCache([string]$ds) {
    Log "=== Clear stale 5000-row prediction cache for $ds ==="
    if ($DryRun) {
        Log "DryRun: would remove results/cache/${ds}_fold_*_*_train_only_preds.csv"
        return
    }
    $files = Get-ChildItem -Path 'results/cache' -Filter "${ds}_fold_*_*_train_only_preds.csv" -ErrorAction SilentlyContinue
    $n = ($files | Measure-Object).Count
    foreach ($f in $files) { Remove-Item $f.FullName -Force }
    Log "Removed $n cached pred file(s) for $ds"
}

function Invoke-DatasetRun([string]$ds) {
    $cfg = Get-ConfigFor $ds
    Clear-PredCache $ds

    $PythonArgs = @(
        '-m', 'src.baseline_runner',
        '--config', $cfg,
        '--cold-start-only',
        '--force-cold-start',
        '--log-level', 'INFO'
    )
    if ($FoldIdx -ge 0) { $PythonArgs += @('--fold-idx', "$FoldIdx") }
    if ($Models) { $PythonArgs += @('--models', $Models) }

    Log "=== Run cold-start rerun: dataset=$ds ==="
    if ($DryRun) {
        Log "DryRun: $PYTHON $($PythonArgs -join ' ')"
        return
    }
    Invoke-PythonModuleLogged -PythonArgs $PythonArgs -LogPath (Join-Path $LOG_DIR "a3_cold_start_${ds}.log")
}

function Invoke-RegenerateTables {
    Log '=== Regenerate cold-start TeX tables ==='
    if ($DryRun) {
        Log 'DryRun: generate_phase_c_tables + generate_cold_start_comparison + cold-start TeX'
        return
    }
    Invoke-PythonModuleLogged -PythonArgs @('-m', 'scripts.generate_phase_c_tables') -LogPath $LOG
    Invoke-PythonModuleLogged -PythonArgs @('-m', 'scripts.generate_cold_start_comparison') -LogPath $LOG
    Invoke-PythonModuleLogged -PythonArgs @(
        '-c',
        "from scripts.generate_paper_artifacts import _write_cold_start_tex, _write_cold_start_by_stratum_tex; from pathlib import Path; _write_cold_start_by_stratum_tex(Path('results/tables/cold_start_by_stratum.tex')); _write_cold_start_tex(Path('results/tables/cold_start_metrics.tex'))"
    ) -LogPath $LOG
}

function Invoke-Verify([string]$ds) {
    if ($DryRun -or $SkipVerify) { return }
    Log "=== Verify very_cold strata for $ds ==="
    & $PYTHON -c @"
import pandas as pd
ds = '$ds'
df = pd.read_csv('results/tables/cold_start_metrics.csv')
sub = df[(df['dataset'] == ds) & (df['stratum'] == 'very_cold')]
if sub.empty:
    print(f'WARNING: no very_cold rows for {ds}')
else:
    print(f'\n--- {ds} very_cold (post A3 rerun) ---')
    for fold, part in sub.groupby('fold'):
        aucs = part.groupby('model')['auc'].first()
        n_val = part.groupby('model')['n'].first().iloc[0]
        print(f'fold {fold}: n={n_val:.0f}, models={len(aucs)}, unique_AUC={aucs.nunique()}')
        print(aucs.sort_index().to_string())
"@
}

if ($TablesOnly) {
    Invoke-RegenerateTables
    $verifyDs = if ($Dataset -eq 'all') { 'junyi' } else { $Dataset }
    Invoke-Verify $verifyDs
    Log '=== Done (tables-only) ==='
    exit 0
}

Test-Preflight

if ($Dataset -eq 'all') {
    foreach ($ds in @('junyi', 'xes3g5m', 'assist2012')) {
        Invoke-DatasetRun $ds
    }
} else {
    Invoke-DatasetRun $Dataset
}

Invoke-RegenerateTables
$verifyDs = if ($Dataset -eq 'all') { 'junyi' } else { $Dataset }
Invoke-Verify $verifyDs

Log '=== A3 cold-start rerun complete ==='
Log 'Next: sync results/tables/cold_start_* -> paper/submission_APIN/ and rebuild PDF'
