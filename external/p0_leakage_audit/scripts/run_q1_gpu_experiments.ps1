# Q1 GPU experiment orchestrator (24GB VRAM target: XES3G5M primary trio).
#
# Usage (repo root, venv + pip install -e ".[pykt]" + CUDA PyTorch):
#   .\scripts\run_q1_gpu_experiments.ps1              # all phases
#   .\scripts\run_q1_gpu_experiments.ps1 phase1       # GKT epochs30, seed 42 only
#   .\scripts\run_q1_gpu_experiments.ps1 phase2       # + seeds 17, 1234
#   .\scripts\run_q1_gpu_experiments.ps1 phase3       # primary trio matched, 3 seeds
#   .\scripts\run_q1_gpu_experiments.ps1 summarize    # merge results/q1 → tables
#
# Results land in results/q1/<tag>/ (isolated; main paper tables untouched).

$ErrorActionPreference = 'Continue'

# Get current script parent directory and navigate to repository root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir) {
    Set-Location "$ScriptDir\.."
}

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$PHASE = if ($args[0]) { $args[0] } else { "all" }
$LOG_DIR = "logs/q1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


# Create directories if they do not exist
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}
if (-not (Test-Path "results/q1")) {
    New-Item -ItemType Directory -Force -Path "results/q1" | Out-Null
}

function Log-Message {
    param([string]$message)
    $timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    $log_line = "[$timestamp] $message"
    Write-Host $log_line
    Add-Content -Path "$LOG_DIR/run.log" -Value $log_line
}

function Check-Env {
    Log-Message "=== Environment check ==="
    & $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # Check pykt package
    try {
        & $PYTHON -c "import pykt" 2>$null
    } catch {
        Write-Error "Install pyKT: pip install -e '.[pykt]'"
        exit 1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Install pyKT: pip install -e '.[pykt]'"
        exit 1
    }
    
    if (-not (Test-Path "data/processed/xes3g5m.parquet")) {
        Write-Error "Missing data/processed/xes3g5m.parquet"
        exit 1
    }
    Log-Message "Parquet OK"
}

function Ensure-Graphs {
    Log-Message "=== Graph builder (skip if exports exist) ==="
    if (Test-Path "data/processed/xes3g5m/fold_0/e_pre_train_only.csv") {
        Log-Message "Fold exports present; skipping graph_builder"
        return
    }
    cmd /c """$PYTHON"" -m src.graph_builder --config configs/xes3g5m.yaml 2>&1" | Tee-Object -Append -FilePath "$LOG_DIR/graph_builder.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Run-GktEpochs30 {
    param([string]$baseSeed)
    $tag = "gkt_epochs30_s$baseSeed"
    Log-Message "=== GKT epochs30 | split_base_seed=$baseSeed | tag=$tag ==="
    
    # 1) Rebuild graph for the correct split seed
    Log-Message "Rebuilding graphs for split seed $baseSeed..."
    $config_file = "configs/xes3g5m_split$baseSeed.yaml"
    if (-not (Test-Path $config_file)) {
        $config_file = "configs/xes3g5m.yaml"
    }
    cmd /c """$PYTHON"" -m src.graph_builder --config $config_file 2>&1" | Tee-Object -Append -FilePath "$LOG_DIR/graph_builder_s$baseSeed.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # 2) Clear GKT cache to be safe
    & $PYTHON scripts/clear_baseline_cache.py --dataset xes3g5m --models gkt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # 3) Train GKT 30ep
    $cmd_str = """$PYTHON"" -m src.baseline_runner --config configs/xes3g5m_gkt_epochs30.yaml --split-base-seed ""$baseSeed"" --isolated-results ""$tag"" --log-level INFO 2>&1"
    cmd /c $cmd_str | Tee-Object -Append -FilePath "$LOG_DIR/$tag.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Run-PrimaryTrio {
    param([string]$baseSeed)
    $tag = "trio_matched_s$baseSeed"
    Log-Message "=== Primary trio (GKT 30ep) | split_base_seed=$baseSeed | tag=$tag ==="
    
    # 1) Rebuild graph for the correct split seed
    Log-Message "Rebuilding graphs for split seed $baseSeed..."
    $config_file = "configs/xes3g5m_split$baseSeed.yaml"
    if (-not (Test-Path $config_file)) {
        $config_file = "configs/xes3g5m.yaml"
    }
    cmd /c """$PYTHON"" -m src.graph_builder --config $config_file 2>&1" | Tee-Object -Append -FilePath "$LOG_DIR/graph_builder_s$baseSeed.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # 2) Clear baseline cache
    & $PYTHON scripts/clear_baseline_cache.py --dataset xes3g5m --models gkt,simplekt,gikt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    
    # 3) Run baseline_runner
    $cmd_str = """$PYTHON"" -m src.baseline_runner --config configs/experiments/xes3g5m_primary_trio_matched.yaml --split-base-seed ""$baseSeed"" --isolated-results ""$tag"" --log-level INFO 2>&1"
    cmd /c $cmd_str | Tee-Object -Append -FilePath "$LOG_DIR/$tag.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Run-Phase1 {
    Check-Env
    Ensure-Graphs
    Run-GktEpochs30 42
    Run-Summarize
}

function Run-Phase2 {
    Check-Env
    Ensure-Graphs
    foreach ($S in 17, 1234) {
        Run-GktEpochs30 $S
    }
    Run-Summarize
}

function Run-Phase3 {
    Check-Env
    Ensure-Graphs
    foreach ($S in 42, 17, 1234) {
        Run-PrimaryTrio $S
    }
    Run-Summarize
}

function Run-Summarize {
    Log-Message "=== Summarize Q1 results ==="
    & $PYTHON scripts/summarize_q1_experiments.py
    Log-Message "Done. Inspect results/tables/q1_gkt_epochs30_ablation.tex"
}

switch ($PHASE) {
    "phase1" { Run-Phase1 }
    "phase2" { Check-Env; Ensure-Graphs; Run-GktEpochs30 17; Run-GktEpochs30 1234; Run-Summarize }
    "phase3" { Run-Phase3 }
    "summarize" { Run-Summarize }
    "all" {
        Run-Phase1
        Run-Phase2
        Run-Phase3
    }
    Default {
        Write-Error "Unknown phase: $PHASE (phase1|phase2|phase3|summarize|all)"
        exit 1
    }
}

Log-Message "=== Finished phase: $PHASE ==="
