#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed1234_part1.csv"
SEED="1234"
CFG="configs/xes3g5m.yaml"
mkdir -p "$(dirname "$OUT")"

echo "=== PART 1: Baseline + edge_drop (ps 0.1, 0.2, 0.3) ==="
python -m scripts.ddr_downstream \
  --config "$CFG" --model gkt \
  --operators edge_drop \
  --ps 0.10 0.20 0.30 \
  --experiment-seed "$SEED" --perturb-seed "$SEED" \
  --out "$OUT"

echo "=== PART 1: Anchor edge_drop (p=0.90) ==="
python -m scripts.ddr_downstream \
  --config "$CFG" --model gkt \
  --operators edge_drop \
  --ps 0.90 \
  --experiment-seed "$SEED" --perturb-seed "$SEED" \
  --out "$OUT"
