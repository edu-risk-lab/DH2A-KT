#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed1234_part2.csv"
SEED="1234"
CFG="configs/xes3g5m.yaml"
mkdir -p "$(dirname "$OUT")"

echo "=== PART 2: node_drop + prereq_preserve (ps 0.1, 0.2, 0.3) FOLD 2 ONLY ==="
python -m scripts.ddr_downstream \
  --config "$CFG" --model gkt \
  --operators node_drop prereq_preserve \
  --ps 0.10 0.20 0.30 \
  --only-fold 2 \
  --experiment-seed "$SEED" --perturb-seed "$SEED" \
  --out "$OUT"

echo "=== PART 2: Anchor node_drop (p=0.90) FOLD 2 ONLY ==="
python -m scripts.ddr_downstream \
  --config "$CFG" --model gkt \
  --operators node_drop \
  --ps 0.90 \
  --only-fold 2 \
  --experiment-seed "$SEED" --perturb-seed "$SEED" \
  --out "$OUT"
