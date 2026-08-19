$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$log = Join-Path (Get-Location) "results\tables\m_stage.log"

function Write-Log([string]$message) {
    Add-Content -Path $log -Value $message -Encoding utf8
    Write-Host $message
}

function Invoke-PythonTrain([string]$name, [string[]]$arglist) {
    Write-Log "=== START $name $(Get-Date -Format o) ==="
    $out = Join-Path (Get-Location) "results\tables\_m_$name.out.txt"
    $err = Join-Path (Get-Location) "results\tables\_m_$name.err.txt"
    $proc = Start-Process -FilePath "python" -ArgumentList (@("-u") + $arglist) `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $out -RedirectStandardError $err
    if (Test-Path $out) { Get-Content $out | Add-Content $log -Encoding utf8 }
    if (Test-Path $err) { Get-Content $err | Add-Content $log -Encoding utf8 }
    Write-Log "=== EXIT $($proc.ExitCode) $name $(Get-Date -Format o) ==="
    if ($proc.ExitCode -ne 0) {
        exit $proc.ExitCode
    }
}

$fa = @(
    "scripts\03_train_tier1.py", "configs\foundational_assist.yaml",
    "--fold", "0", "--device", "cuda", "--architecture", "v4",
    "--use-questions", "--question-graph", "--no-graph", "--no-session",
    "--mask-repeats", "--window-mode", "chunked", "--max-seq-len", "200",
    "--batch-size", "16", "--epochs", "30", "--val-frac", "0.1",
    "--early-stop-patience", "5", "--seed", "42",
    "--graph-dropout", "0", "--graph-sensitivity-weight", "0"
)

Write-Log "=== RESUME m1_on..m5 $(Get-Date -Format o) ==="

Invoke-PythonTrain "m1_qmat_on" ($fa + @(
    "--skill-csv", "data/FoundationalASSIST/Skills.csv", "--full-qmatrix",
    "--tag", "m1_qmat_on", "--output", "results/tables/dh2_kt_m1_qmat_on_vs_p0.csv"
))
Invoke-PythonTrain "m2_dt_on" ($fa + @(
    "--time-gap", "--tag", "m2_dt_on", "--output", "results/tables/dh2_kt_m2_dt_on_vs_p0.csv"
))
Invoke-PythonTrain "m3_saw_on" ($fa + @(
    "--saw-input", "--tag", "m3_saw_on", "--output", "results/tables/dh2_kt_m3_saw_on_vs_p0.csv"
))

$assist = @(
    "scripts\03_train_tier1.py", "configs\assist2012.yaml",
    "--fold", "0", "--device", "cuda", "--architecture", "v4",
    "--use-questions", "--question-graph", "--no-graph", "--no-session",
    "--mask-repeats", "--window-mode", "chunked", "--max-seq-len", "200",
    "--batch-size", "16", "--epochs", "30", "--val-frac", "0.1",
    "--early-stop-patience", "5", "--seed", "42",
    "--graph-dropout", "0", "--graph-sensitivity-weight", "0"
)
Invoke-PythonTrain "m4_group_off" ($assist + @(
    "--tag", "m4_group_off", "--output", "results/tables/dh2_kt_m4_group_off_vs_p0.csv"
))
Invoke-PythonTrain "m4_group_on" ($assist + @(
    "--group-embed", "--tag", "m4_group_on", "--output", "results/tables/dh2_kt_m4_group_on_vs_p0.csv"
))

$junyi = @(
    "scripts\03_train_tier1.py", "configs\junyi.yaml",
    "--fold", "0", "--device", "cuda", "--architecture", "v4",
    "--use-questions", "--question-graph", "--no-graph", "--no-session",
    "--mask-repeats", "--window-mode", "chunked", "--max-seq-len", "200",
    "--batch-size", "16", "--epochs", "30", "--val-frac", "0.1",
    "--early-stop-patience", "5", "--seed", "42",
    "--graph-dropout", "0", "--graph-sensitivity-weight", "0",
    "--max-users", "5000",
    "--expert-dag", "external/p0_leakage_audit/data/raw/junyi/relationship_annotation_training.csv"
)
Invoke-PythonTrain "m5_expert_off" ($junyi + @(
    "--tag", "m5_expert_off", "--output", "results/tables/dh2_kt_m5_expert_off_vs_p0.csv"
))
Invoke-PythonTrain "m5_expert_on" ($junyi + @(
    "--expert-graph", "--tag", "m5_expert_on", "--output", "results/tables/dh2_kt_m5_expert_on_vs_p0.csv"
))
Write-Log "=== DONE GIAI DOAN M $(Get-Date -Format o) ==="
