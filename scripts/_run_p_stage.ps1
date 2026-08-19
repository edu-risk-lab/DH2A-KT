$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$log = Join-Path (Get-Location) "results\tables\p_stage.log"

function Write-Log([string]$message) {
    Add-Content -Path $log -Value $message -Encoding utf8
    Write-Host $message
}

function Invoke-PythonTrain([string]$name, [string[]]$arglist) {
    Write-Log "=== START $name $(Get-Date -Format o) ==="
    $out = Join-Path (Get-Location) "results\tables\_p_$name.out.txt"
    $err = Join-Path (Get-Location) "results\tables\_p_$name.err.txt"
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

# P0: time-gap on primary corpus (XES hg_qkc_on L=400). Twin val 0.825982.
$xes = @(
    "scripts\03_train_tier1.py", "configs\xes3g5m.yaml",
    "--fold", "0", "--device", "cuda", "--architecture", "v4",
    "--use-questions", "--question-graph", "--no-graph", "--no-session",
    "--mask-repeats", "--window-mode", "chunked", "--max-seq-len", "400",
    "--batch-size", "16", "--epochs", "30", "--val-frac", "0.1",
    "--early-stop-patience", "5", "--seed", "42",
    "--graph-dropout", "0", "--graph-sensitivity-weight", "0"
)

# P1/P2: FoundationalASSIST, same protocol as giai doan M.
$fa = @(
    "scripts\03_train_tier1.py", "configs\foundational_assist.yaml",
    "--fold", "0", "--device", "cuda", "--architecture", "v4",
    "--use-questions", "--question-graph", "--no-graph", "--no-session",
    "--mask-repeats", "--window-mode", "chunked", "--max-seq-len", "200",
    "--batch-size", "16", "--epochs", "30", "--val-frac", "0.1",
    "--early-stop-patience", "5", "--seed", "42",
    "--graph-dropout", "0", "--graph-sensitivity-weight", "0"
)

$assist = @(
    "scripts\03_train_tier1.py", "configs\assist2012.yaml",
    "--fold", "0", "--device", "cuda", "--architecture", "v4",
    "--use-questions", "--question-graph", "--no-graph", "--no-session",
    "--mask-repeats", "--window-mode", "chunked", "--max-seq-len", "200",
    "--batch-size", "16", "--epochs", "30", "--val-frac", "0.1",
    "--early-stop-patience", "5", "--seed", "42",
    "--graph-dropout", "0", "--graph-sensitivity-weight", "0"
)

Write-Log "=== START P0-P3 $(Get-Date -Format o) ==="

Invoke-PythonTrain "p0_dt_on" ($xes + @(
    "--time-gap", "--time-gap-mode", "both",
    "--tag", "p0_dt_on", "--output", "results/tables/dh2_kt_p0_dt_on_vs_p0.csv"
))

Invoke-PythonTrain "p1_dt_lstm" ($fa + @(
    "--time-gap", "--time-gap-mode", "lstm",
    "--tag", "p1_dt_lstm", "--output", "results/tables/dh2_kt_p1_dt_lstm_vs_p0.csv"
))
Invoke-PythonTrain "p1_dt_query" ($fa + @(
    "--time-gap", "--time-gap-mode", "query",
    "--tag", "p1_dt_query", "--output", "results/tables/dh2_kt_p1_dt_query_vs_p0.csv"
))
Invoke-PythonTrain "p2_dt_saw" ($fa + @(
    "--time-gap", "--saw-input",
    "--tag", "p2_dt_saw", "--output", "results/tables/dh2_kt_p2_dt_saw_vs_p0.csv"
))
Invoke-PythonTrain "p3_split_on" ($assist + @(
    "--time-split",
    "--tag", "p3_split_on", "--output", "results/tables/dh2_kt_p3_split_on_vs_p0.csv"
))

Write-Log "=== DONE P0-P3 $(Get-Date -Format o) ==="
Write-Log "P4 (--concept-forget) is gated: run only if P0 PASS or P1 lstm carries AUC."
