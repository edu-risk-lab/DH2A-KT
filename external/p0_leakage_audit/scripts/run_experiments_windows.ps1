$ErrorActionPreference = "Stop"

$seeds = @(17, 1234)

foreach ($seed in $seeds) {
    Write-Host "=== seed=$seed grid: xes3g5m ==="
    python -m scripts.ddr_downstream `
      --config configs/xes3g5m.yaml --model gkt `
      --operators edge_drop node_drop prereq_preserve --ps 0.10 0.20 0.30 `
      --experiment-seed $seed --perturb-seed $seed `
      --out results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed${seed}.csv

    Write-Host "=== seed=$seed anchor: xes3g5m ==="
    python -m scripts.ddr_downstream `
      --config configs/xes3g5m.yaml --model gkt `
      --operators edge_drop node_drop --ps 0.90 `
      --experiment-seed $seed --perturb-seed $seed `
      --out results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed${seed}.csv
}

$all_seeds = @(42, 17, 1234)
foreach ($s in $all_seeds) {
    Write-Host "=== A6 reachability disruption seed=$s ==="
    python -m scripts.reachability_disruption `
      --results results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed${s}.csv `
      --perturb-seed $s
}
