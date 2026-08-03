#!/usr/bin/env bash
# A2.2b — Rerun DDR GKT baseline (operator=none) for XES3G5M fold 0, experiment seed 17.
#
# Replaces the invalid legacy placeholder copied from a GKT30 run.
#
# Usage (GPU server, repo root):
#   bash scripts/run_a2_2b_ddr_seed17_baseline.sh
#   bash scripts/run_a2_2b_ddr_seed17_baseline.sh --dry-run
#   bash scripts/run_a2_2b_ddr_seed17_baseline.sh --skip-graph-rebuild
#   bash scripts/run_a2_2b_ddr_seed17_baseline.sh --merge-only
#
# After success on server, sync shard back to dev and run --merge-only locally if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="${PYTHON:-python}"
CONFIG_SPLIT="configs/xes3g5m_split17.yaml"
CONFIG_DDR="configs/xes3g5m.yaml"
SEED=17
FOLD=0
BATCH_SIZE="${BATCH_SIZE:-8}"   # attested DDR multiseed (docs/GKT_Hyperparams_Review_Report.md)
OUT="results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed${SEED}.csv"
MERGED="results/tables/ddr_downstream.csv"
LOG_DIR="logs/q1"
WORK_DIR="results/pykt_work/xes3g5m/fold_${FOLD}_seed_${SEED}/ddr_downstream/gkt"
PLACEHOLDER_AUC="0.8422173236026003"

SKIP_GRAPH=0
DRY_RUN=0
MERGE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --skip-graph-rebuild) SKIP_GRAPH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --merge-only) MERGE_ONLY=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$(dirname "$OUT")"
LOG="$LOG_DIR/a2_2b_ddr_seed17_baseline.log"

log() {
  echo "[$(date -Iseconds)] $*"
  echo "[$(date -Iseconds)] $*" >> "$LOG"
}

check_env() {
  log "=== Environment check ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: skipping CUDA check"
    return
  fi
  "$PYTHON" -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
  "$PYTHON" -c "import pykt"
  test -f data/processed/xes3g5m.parquet
  log "Environment OK"
}

remove_placeholder_row() {
  log "=== Removing placeholder fold-0 none row (AUC=${PLACEHOLDER_AUC}) ==="
  if [[ ! -f "$OUT" ]]; then
    log "No existing shard $OUT — will create fresh"
    return
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: would strip placeholder from $OUT"
    return
  fi
  "$PYTHON" - "$OUT" "$PLACEHOLDER_AUC" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
placeholder = float(sys.argv[2])
df = pd.read_csv(path)
mask = (
    (df["dataset"] == "xes3g5m")
    & (df["model"] == "gkt")
    & (df["fold"] == 0)
    & (df["operator"] == "none")
    & (df["p"] == 0.0)
)
removed = int(mask.sum())
if removed:
    bad = df.loc[mask, "auc"].astype(float)
    if not (bad - placeholder).abs().lt(1e-6).all():
        print(f"WARNING: fold-0 none AUC is {bad.tolist()}, not placeholder — still removing for rerun")
    df = df.loc[~mask]
    df.to_csv(path, index=False)
    print(f"Removed {removed} placeholder row(s) from {path}")
else:
    print(f"No fold-0 none row in {path} (already clean or missing)")
PY
}

rebuild_graph() {
  log "=== Rebuild train-only graphs for split seed ${SEED} ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: $PYTHON -m src.graph_builder --config $CONFIG_SPLIT"
    return
  fi
  "$PYTHON" -m src.graph_builder --config "$CONFIG_SPLIT" \
    2>&1 | tee -a "$LOG_DIR/a2_2b_graph_builder_seed${SEED}.log"
}

clear_workdir() {
  log "=== Clear stale DDR workdir for fold ${FOLD} seed ${SEED} ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: would remove $WORK_DIR"
    return
  fi
  if [[ -d "$WORK_DIR" ]]; then
    rm -rf "$WORK_DIR"
    log "Removed $WORK_DIR"
  fi
}

run_baseline() {
  log "=== Train GKT baseline only: fold=${FOLD} seed=${SEED} batch=${BATCH_SIZE} ==="
  local args=(
    -m scripts.ddr_downstream
    --config "$CONFIG_DDR"
    --model gkt
    --baseline-only
    --only-fold "$FOLD"
    --experiment-seed "$SEED"
    --perturb-seed "$SEED"
    --batch-size "$BATCH_SIZE"
    --out "$OUT"
    --log-level INFO
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: $PYTHON ${args[*]}"
    return
  fi
  "$PYTHON" "${args[@]}" 2>&1 | tee -a "$LOG"
}

verify_result() {
  log "=== Verify new fold-0 none AUC !== placeholder ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi
  "$PYTHON" - "$OUT" "$PLACEHOLDER_AUC" <<'PY'
import sys
import pandas as pd

path, placeholder = sys.argv[1], float(sys.argv[2])
df = pd.read_csv(path)
row = df[
    (df["dataset"] == "xes3g5m")
    & (df["model"] == "gkt")
    & (df["fold"] == 0)
    & (df["operator"] == "none")
    & (df["p"] == 0.0)
]
if row.empty:
    raise SystemExit("ERROR: fold-0 none row missing after rerun")
auc = float(row["auc"].iloc[0])
if abs(auc - placeholder) < 1e-6:
    raise SystemExit(f"ERROR: AUC still placeholder ({auc})")
print(f"OK: fold-0 none AUC = {auc:.6f} (was placeholder {placeholder})")
PY
}

merge_tables() {
  log "=== Merge seed-17 shard into $MERGED ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: merge_ddr_downstream + plot + tex"
    return
  fi
  "$PYTHON" -m scripts.merge_ddr_downstream \
    --base "$MERGED" \
    --append "$OUT" \
    --out "$MERGED"
  "$PYTHON" -m scripts.plot_ddr_downstream
  "$PYTHON" -m scripts.generate_ddr_downstream_gkt_tex
  log "Merged tables refreshed"
}

log "=== A2.2b DDR seed-17 baseline rerun start ==="

if [[ "$MERGE_ONLY" -eq 1 ]]; then
  merge_tables
  log "=== Done (merge only) ==="
  exit 0
fi

check_env
remove_placeholder_row
if [[ "$SKIP_GRAPH" -eq 0 ]]; then
  rebuild_graph
fi
clear_workdir
run_baseline
verify_result
merge_tables

log "=== A2.2b finished ==="
log "Next: copy $OUT and $MERGED to dev; sync paper/submission_APIN/ddr_downstream*.tex if changed"
