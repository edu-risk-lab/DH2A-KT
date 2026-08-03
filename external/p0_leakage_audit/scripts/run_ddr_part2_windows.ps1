$OUT = "results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed1234_part2.csv"
$SEED = "1234"
$CFG = "configs/xes3g5m.yaml"
$dir = Split-Path $OUT
if (!(Test-Path $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "=== PART 2: node_drop + prereq_preserve (ps 0.1, 0.2, 0.3) ==="
python -m scripts.ddr_downstream --config $CFG --model gkt --operators node_drop prereq_preserve --ps 0.10 0.20 0.30 --experiment-seed $SEED --perturb-seed $SEED --out $OUT --max-folds 2

Write-Host "=== PART 2: Anchor node_drop (p=0.90) ==="
python -m scripts.ddr_downstream --config $CFG --model gkt --operators node_drop --ps 0.90 --experiment-seed $SEED --perturb-seed $SEED --out $OUT --max-folds 2
