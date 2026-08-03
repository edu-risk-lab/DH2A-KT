# Injection downstream AUC batch — XES3G5M fold 0, split seed 42 (Table S18).
# Runs 6 GPU jobs (inject05/inject20 × simpleKT, GKT, GIKT).
# Column 0% uses fold-0 train_only via scripts/collect_injection_auc.py (no inject00 GPU job).
#
# Usage (repo root, venv + pip install -e ".[pykt]" + CUDA PyTorch):
#   .\scripts\run_injection_downstream_batch.ps1
#   .\scripts\run_injection_downstream_batch.ps1 -SkipGraphs
#   .\scripts\run_injection_downstream_batch.ps1 -CollectOnly
#   .\scripts\run_injection_downstream_batch.ps1 -ClearCache
#   .\scripts\run_injection_downstream_batch.ps1 -DryRun
#
# Outputs:
#   results/cache/xes3g5m_fold_0_{model}_s42_inject{05,20}_result.json
#   results/tables/downstream_auc_injection.csv + .tex
#   logs/q1/injection_downstream_{arm}_{model}.log

param(
    [switch]$SkipGraphs,
    [switch]$CollectOnly,
    [switch]$ClearCache,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir) {
    Set-Location (Join-Path $ScriptDir '..')
}

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$CONFIG = 'configs/xes3g5m.yaml'
$DATASET = 'xes3g5m'
$FOLD = 0
$SPLIT_SEED = 42
$LOG_DIR = 'logs/q1'
$CACHE_DIR = 'results/cache'

# Order: simpleKT → GIKT → GKT (GKT is slowest)
$JOBS = @(
    @{ Arm = 'inject05'; Model = 'simplekt' },
    @{ Arm = 'inject20'; Model = 'simplekt' },
    @{ Arm = 'inject05'; Model = 'gikt' },
    @{ Arm = 'inject20'; Model = 'gikt' },
    @{ Arm = 'inject05'; Model = 'gkt' },
    @{ Arm = 'inject20'; Model = 'gkt' }
)

$LOCK_FILE = Join-Path $LOG_DIR 'injection_downstream_batch.lock'

function Enter-BatchLock {
    if ($DryRun) { return }
    if (Test-Path $LOCK_FILE) {
        $lockPid = Get-Content $LOCK_FILE -ErrorAction SilentlyContinue
        if ($lockPid -and (Get-Process -Id ([int]$lockPid) -ErrorAction SilentlyContinue)) {
            Write-Error "Another injection batch is running (PID $lockPid). Stop it first or wait for completion."
            exit 1
        }
        Remove-Item $LOCK_FILE -Force -ErrorAction SilentlyContinue
    }
    $PID | Out-File -FilePath $LOCK_FILE -Encoding ascii
}

function Exit-BatchLock {
    if ($DryRun) { return }
    if (Test-Path $LOCK_FILE) { Remove-Item $LOCK_FILE -Force -ErrorAction SilentlyContinue }
}

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK')] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $LOG_DIR 'injection_downstream_batch.log') -Value $line
}

function Test-ConfigHyperparams {
    Write-Log '=== Config preflight (Table S15 / 619c02cf) ==='
    $giktBlock = Select-String -Path $CONFIG -Pattern 'name: gikt' -Context 0,5 | Select-Object -First 1
    $gktBlock = Select-String -Path $CONFIG -Pattern 'name: gkt' -Context 0,5 | Select-Object -First 1
    $fail = $false
    if (-not ($giktBlock -and ($giktBlock.Context.PostContext -join "`n") -match 'batch_size:\s*8\b')) {
        Write-Error 'configs/xes3g5m.yaml: baselines.gikt.hyperparams.batch_size must be 8 (not 16)'
        $fail = $true
    }
    if (-not ($gktBlock -and ($gktBlock.Context.PostContext -join "`n") -match 'batch_size:\s*4\b')) {
        Write-Error 'configs/xes3g5m.yaml: baselines.gkt.hyperparams.batch_size must be 4'
        $fail = $true
    }
    if ($fail) { exit 1 }
    Write-Log 'Config OK: GKT batch=4, GIKT batch=8'
}

function Test-CudaEnv {
    Write-Log '=== Environment check ==='
    if ($DryRun) {
        Write-Log 'DryRun: skipping CUDA check'
        return
    }
    & $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('GPU:', torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $PYTHON -c "import pykt" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'Install pyKT: pip install -e ''.[pykt]'''
        exit 1
    }
    if (-not (Test-Path 'data/processed/xes3g5m.parquet')) {
        Write-Error 'Missing data/processed/xes3g5m.parquet'
        exit 1
    }
    Write-Log 'Environment OK'
}

function Invoke-PythonLogged {
    param(
        [string]$Command,
        [string]$LogPath
    )
    $cmd = """$PYTHON"" -c ""$Command"" 2>&1"
    cmd /c $cmd | Tee-Object -Append -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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

function Build-InjectGraphs {
    Write-Log '=== Building injected graphs (CPU) ==='
    if ($DryRun) {
        Write-Log 'DryRun: would run build_injected_graphs()'
        return
    }
    Invoke-PythonLogged `
        -Command "from scripts.run_injection_auc import build_injected_graphs; build_injected_graphs()" `
        -LogPath (Join-Path $LOG_DIR 'injection_downstream_graphs.log')

    Write-Log '=== S17 sanity (run_leak_injection) ==='
    Invoke-PythonModuleLogged `
        -PythonArgs @('-m', 'scripts.run_leak_injection') `
        -LogPath (Join-Path $LOG_DIR 'injection_downstream_s17.log')
    Write-Log 'Expect |E_pre|: 1162 / 1378 / 1417 in results/tables/leak_injection.csv'
}

function Clear-InjectCaches {
    Write-Log '=== Clearing inject05/inject20 caches (primary trio only) ==='
    foreach ($job in $JOBS) {
        $arm = $job.Arm
        $model = $job.Model
        $base = "${DATASET}_fold_${FOLD}_${model}_s${SPLIT_SEED}_${arm}"
        foreach ($suffix in @('_result.json', '_preds.csv')) {
            $path = Join-Path $CACHE_DIR ($base + $suffix)
            if (Test-Path $path) {
                if ($DryRun) {
                    Write-Log "DryRun: would remove $path"
                } else {
                    Remove-Item $path -Force
                    Write-Log "Removed $path"
                }
            }
        }
        $workDir = "results/pykt_work/xes3g5m/fold_${FOLD}_seed_${SPLIT_SEED}/$arm"
        if ((Test-Path $workDir) -and $ClearCache) {
            if ($DryRun) {
                Write-Log "DryRun: would remove $workDir"
            } else {
                Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
                Write-Log "Removed workdir $workDir"
            }
        }
    }
}

function Invoke-InjectionJob {
    param(
        [string]$Arm,
        [string]$Model
    )
    $tag = "${Arm}_${Model}"
    $logPath = Join-Path $LOG_DIR "injection_downstream_${tag}.log"
    Write-Log "=== Job: fold=$FOLD arm=$Arm model=$Model ==="

    $runnerArgs = @(
        '-m', 'src.baseline_runner',
        '--config', $CONFIG,
        '--baseline-backend', 'pykt',
        '--fold-idx', "$FOLD",
        '--seed', "$SPLIT_SEED",
        '--split-base-seed', "$SPLIT_SEED",
        '--graph-construction', $Arm,
        '--models', $Model,
        '--log-level', 'INFO'
    )

    if ($DryRun) {
        Write-Log "DryRun: $PYTHON $($runnerArgs -join ' ') 2>&1 | Tee-Object $logPath"
        return
    }

    $cmd = """$PYTHON"" $($runnerArgs -join ' ') 2>&1"
    cmd /c $cmd | Tee-Object -Append -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed: arm=$Arm model=$Model (see $logPath)"
        exit $LASTEXITCODE
    }

    $cacheJson = Join-Path $CACHE_DIR "${DATASET}_fold_${FOLD}_${Model}_s${SPLIT_SEED}_${Arm}_result.json"
    if (-not (Test-Path $cacheJson)) {
        Write-Error "Expected cache missing after job: $cacheJson"
        exit 1
    }
    Write-Log "OK: $cacheJson"
}

function Collect-InjectionTable {
    Write-Log '=== Collecting Table S18 (collect_injection_auc) ==='
    if ($DryRun) {
        Write-Log 'DryRun: would run python -m scripts.collect_injection_auc'
        return
    }
    Invoke-PythonModuleLogged `
        -PythonArgs @('-m', 'scripts.collect_injection_auc') `
        -LogPath (Join-Path $LOG_DIR 'injection_downstream_collect.log')
    Write-Log 'Wrote results/tables/downstream_auc_injection.csv and .tex'
}

# --- main ---
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

Enter-BatchLock
try {
Write-Log '=== Injection downstream batch start ==='
Write-Log "Jobs: $($JOBS.Count) | fold=$FOLD split_seed=$SPLIT_SEED"

Test-ConfigHyperparams
Test-CudaEnv

if ($CollectOnly) {
    Collect-InjectionTable
    Write-Log '=== Done (collect only) ==='
    return
}

if (-not $SkipGraphs) {
    Build-InjectGraphs
}

if ($ClearCache) {
    Clear-InjectCaches
}

$jobNum = 0
foreach ($job in $JOBS) {
    $jobNum++
    Write-Log "--- Progress $jobNum/$($JOBS.Count) ---"
    Invoke-InjectionJob -Arm $job.Arm -Model $job.Model
}

Collect-InjectionTable
Write-Log '=== Injection downstream batch finished ==='
} finally {
    Exit-BatchLock
}
