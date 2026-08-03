<#
.SYNOPSIS
    1-seed P0 pipeline: preprocess (optional skip), split, graphs, DAG probes, baselines (1 fold), cold-start merge, paper artefacts.

.PARAMETER ForceFull
    Sets FORCE_PREPROCESS=1 so every dataset rebuilds parquet from raw CSV (use after schema/raw updates).

.PARAMETER ServerProfile
    Sensible defaults for a ~32GB RAM host: caps BLAS/OpenMP threads so pandas/sklearn do not oversubscribe.
    Does not configure CUDA; set CUDA_VISIBLE_DEVICES yourself if a stage uses GPU.

.EXAMPLE
    .\scripts\run_all_datasets_1seed.ps1 -ForceFull -ServerProfile

.EXAMPLE
    $env:CUDA_VISIBLE_DEVICES = "0"
    .\scripts\run_all_datasets_1seed.ps1 -ServerProfile
#>

param(
    [switch]$ForceFull,
    [switch]$ServerProfile
)

$ErrorActionPreference = "Stop"

if ($ForceFull) {
    $env:FORCE_PREPROCESS = "1"
    Write-Host "[run_all_datasets_1seed] ForceFull: FORCE_PREPROCESS=1 (rebuild parquet from raw)."
}

if ($ServerProfile) {
    $threadVars = @(
        @{ Name = "OMP_NUM_THREADS"; Default = "8" },
        @{ Name = "MKL_NUM_THREADS"; Default = "8" },
        @{ Name = "OPENBLAS_NUM_THREADS"; Default = "8" },
        @{ Name = "NUMEXPR_NUM_THREADS"; Default = "8" }
    )
    foreach ($tv in $threadVars) {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($tv.Name, "Process"))) {
            Set-Item -Path "env:$($tv.Name)" -Value $tv.Default
        }
    }
    if ([string]::IsNullOrEmpty($env:PYTHONHASHSEED)) {
        $env:PYTHONHASHSEED = "0"
    }
    Write-Host "[run_all_datasets_1seed] ServerProfile: BLAS/thread caps (override by setting env vars before launch)."
    Write-Host "    OMP_NUM_THREADS=$($env:OMP_NUM_THREADS) MKL_NUM_THREADS=$($env:MKL_NUM_THREADS) OPENBLAS_NUM_THREADS=$($env:OPENBLAS_NUM_THREADS)"
}

if ($env:PYTHON) {
    $Python = $env:PYTHON
}
else {
    $Python = "python"
}

Write-Host "[run_all_datasets_1seed] Using interpreter: $Python"
Write-Host "[run_all_datasets_1seed] FORCE_PREPROCESS=$($env:FORCE_PREPROCESS) CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)"

function Invoke-P0Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $($Arguments -join ' ')"
    }
}

$Datasets = @(
    @{
        Config = "configs/assist2012.yaml"
        Processed = "data/processed/assist2012.parquet"
    },
    @{
        Config = "configs/xes3g5m.yaml"
        Processed = "data/processed/xes3g5m.parquet"
    },
    @{
        Config = "configs/synthetic_c2.yaml"
        Processed = "data/processed/synthetic_c2.parquet"
    },
    @{
        Config = "configs/synthetic_c5.yaml"
        Processed = "data/processed/synthetic_c5.parquet"
    },
    @{
        Config = "configs/junyi.yaml"
        Processed = "data/processed/junyi.parquet"
    }
)

foreach ($Dataset in $Datasets) {
    $Cfg = $Dataset.Config
    $Processed = $Dataset.Processed
    Write-Host "==> Running 1-seed P0 pipeline for $Cfg"
    if ((Test-Path $Processed) -and ($env:FORCE_PREPROCESS -ne "1")) {
        Write-Host "    Skipping preprocess; found $Processed"
    }
    else {
        Invoke-P0Step @("-m", "src.preprocess", "--config", $Cfg)
    }
    Invoke-P0Step @("-m", "src.split_checker", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.graph_builder", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.export_full_log_graph", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.dag_audit", "--config", $Cfg)
    Invoke-P0Step @("-m", "src.dag_disruption", "--config", $Cfg)
    
    # Run baseline runner with only 1 fold/seed
    $baselineArgs = @("-m", "src.baseline_runner", "--config", $Cfg, "--folds", "1")
    Invoke-P0Step -Arguments $baselineArgs
    
    Invoke-P0Step @("-m", "src.cold_start_report", "--config", $Cfg)
}

Write-Host "==> Generating paper artefacts"
Invoke-P0Step @("scripts/generate_paper_artifacts.py")
Invoke-P0Step @("-m", "src.report_generator", "--out", "results/reports/")

Write-Host "==> Done. Main report: results/reports/p0_diagnostic_report.md"
Write-Host "    Optional Junyi GT CV: python scripts/run_gt_cross_validation_junyi.py (needs kc_name_to_id.json)"
