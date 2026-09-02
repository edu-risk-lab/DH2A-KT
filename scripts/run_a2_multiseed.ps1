# GS Hau A2 — launch multi-seed matrix (30 runs, ~2-3 days GPU).
# Run from PowerShell OUTSIDE Cursor if WDAC blocks torch in agent shells.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Optional: point to a Python where `import torch` works
# $env:DH2A_PYTHON = "C:\Path\To\python.exe"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$stdout = Join-Path (Get-Location) "results\tables\a2_multiseed_stdout.log"
Write-Host "Starting A2 matrix; stdout -> $stdout"
python -u scripts/27_a2_multiseed_matrix.py --device cuda *> $stdout
exit $LASTEXITCODE
