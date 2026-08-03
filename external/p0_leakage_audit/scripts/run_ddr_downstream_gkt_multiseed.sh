#!/usr/bin/env bash
# DDR -> downstream GKT sweep, AMENDED for the two advisors' requirements:
#   (1) manipulation check / positive control: a near-empty-graph anchor
#       (edge_drop & node_drop at p=0.90) so a null result can be told apart
#       from a graph-inert backbone (the DGEKT problem);
#   (2) >=3 seeds for CI / noise-floor + ANOVA power.
#
# IMPORTANT (footgun): scripts/ddr_downstream.py and merge_ddr_downstream.py
# dedupe by (dataset, model, fold, operator, p) WITHOUT seed. Different seeds
# MUST therefore go to SEPARATE output CSVs, or later seeds get skipped /
# collapsed. This script writes one shard per seed.
#
# Usage (GPU server):
#   bash scripts/run_ddr_downstream_gkt_multiseed.sh                 # full
#   DATASETS="configs/xes3g5m.yaml" bash scripts/...                 # primary only
#   SEEDS="42 17" MAX_FOLDS=1 bash scripts/...                       # calibrate
#
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 17 1234}"
DATASETS="${DATASETS:-configs/xes3g5m.yaml configs/assist2012.yaml}"
OUTDIR="${OUTDIR:-results/q1/ddr_downstream_gkt}"
OPS="edge_drop node_drop prereq_preserve"
PS="0.10 0.20 0.30"
ANCHOR_OPS="edge_drop node_drop"   # near-empty E_pre = "does the graph matter at all?"
ANCHOR_PS="0.90"
mkdir -p "$OUTDIR"

MAX_FOLDS_ARG=()
[[ -n "${MAX_FOLDS:-}" ]] && MAX_FOLDS_ARG=(--max-folds "$MAX_FOLDS")

for SEED in $SEEDS; do
  OUT="$OUTDIR/ddr_downstream_gkt_seed${SEED}.csv"
  for cfg in $DATASETS; do
    echo "=== seed=$SEED grid: $cfg -> $OUT ==="
    python -m scripts.ddr_downstream \
      --config "$cfg" --model gkt \
      --operators $OPS --ps $PS \
      --experiment-seed "$SEED" --perturb-seed "$SEED" \
      --out "$OUT" "${MAX_FOLDS_ARG[@]}"

    echo "=== seed=$SEED manipulation-check anchor: $cfg -> $OUT ==="
    python -m scripts.ddr_downstream \
      --config "$cfg" --model gkt \
      --operators $ANCHOR_OPS --ps $ANCHOR_PS \
      --experiment-seed "$SEED" --perturb-seed "$SEED" \
      --out "$OUT" "${MAX_FOLDS_ARG[@]}"
  done
done

echo "=== per-seed shard summary ==="
for SEED in $SEEDS; do
  OUT="$OUTDIR/ddr_downstream_gkt_seed${SEED}.csv"
  [[ -f "$OUT" ]] || continue
  python - "$OUT" <<'PY'
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_csv(p)
print(p)
print(df.groupby(["dataset","operator"]).size().to_string())
print("total", len(df), "\n")
PY
done

cat <<'NEXT'
Next steps (offline, no GPU):
  # A6 reachability disruption + DDR-vs-reach correlation, per seed shard:
  for s in 42 17 1234; do
    python -m scripts.reachability_disruption \
      --results results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed${s}.csv \
      --perturb-seed ${s}
  done
  # Inspect the p=0.90 anchor rows: if GKT AUC barely moves there, GKT is
  # graph-inert (Outcome C) -> do NOT interpret the DDR<->AUC correlation.
NEXT
