#!/usr/bin/env bash
# DDR -> downstream GKT sweep (ASSISTments + XES3G5M).
# Resumable; matches paper DGEKT operator grid.
#
# Usage (GPU server):
#   bash scripts/run_ddr_downstream_gkt.sh
#   bash scripts/run_ddr_downstream_gkt.sh --max-folds 1   # calibrate only
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/q1/ddr_downstream_gkt/ddr_downstream_gkt.csv}"
mkdir -p "$(dirname "$OUT")"

MAX_FOLDS=()
if [[ "${1:-}" == "--max-folds" && -n "${2:-}" ]]; then
  MAX_FOLDS=(--max-folds "$2")
  shift 2
fi

OPS=(edge_drop node_drop prereq_preserve)
PS=(0.10 0.20 0.30)

run_one() {
  local cfg="$1"
  echo "=== DDR downstream GKT: $cfg -> $OUT ==="
  python -m scripts.ddr_downstream \
    --config "$cfg" \
    --model gkt \
    --operators "${OPS[@]}" \
    --ps "${PS[@]}" \
    --experiment-seed 42 \
    --perturb-seed 42 \
    --out "$OUT" \
    "${MAX_FOLDS[@]}"
}

run_one configs/assist2012.yaml
run_one configs/xes3g5m.yaml

echo "Done. Rows in $OUT:"
python - <<PY
import pandas as pd
from pathlib import Path
p = Path("$OUT")
if not p.exists():
    raise SystemExit("missing output")
df = pd.read_csv(p)
print(df.groupby(["dataset", "model"]).size())
print("total", len(df))
PY
