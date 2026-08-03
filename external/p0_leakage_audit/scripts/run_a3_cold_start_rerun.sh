#!/usr/bin/env bash
# A3.5 — Optional GPU rerun: cold-start strata with full pyKT predictions (no 5000 cap).
#
# Regenerates uncapped prediction inputs for cold-start diagnostics.
# Degenerate Junyi strata can still have undefined AUC when class support is insufficient.
#
# Usage (GPU server, repo root):
#   bash scripts/run_a3_cold_start_rerun.sh                    # Junyi (default)
#   bash scripts/run_a3_cold_start_rerun.sh --dataset xes3g5m
#   bash scripts/run_a3_cold_start_rerun.sh --dataset all
#   bash scripts/run_a3_cold_start_rerun.sh --fold-idx 0       # debug single fold
#   bash scripts/run_a3_cold_start_rerun.sh --dry-run
#   bash scripts/run_a3_cold_start_rerun.sh --tables-only     # regenerate TeX after manual CSV sync
#   FORCE_CPU=1 bash scripts/run_a3_cold_start_rerun.sh       # CPU fallback (slow)
#
# After success on server, sync cold_start_metrics.csv + tables to dev and rebuild PDF.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/q1"
DATASET="junyi"
FOLD_IDX=""
MODELS=""
DRY_RUN=0
TABLES_ONLY=0
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET="${1#*=}" ;;
    --dataset)
      shift
      DATASET="${1:?--dataset requires value}"
      ;;
    --fold-idx=*) FOLD_IDX="${1#*=}" ;;
    --fold-idx)
      shift
      FOLD_IDX="${1:?--fold-idx requires value}"
      ;;
    --models=*) MODELS="${1#*=}" ;;
    --models)
      shift
      MODELS="${1:?--models requires value}"
      ;;
    --dry-run) DRY_RUN=1 ;;
    --tables-only) TABLES_ONLY=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    --help|-h)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/a3_cold_start_rerun.log"

log() {
  echo "[$(date -Iseconds)] $*"
  echo "[$(date -Iseconds)] $*" >> "$LOG"
}

config_for() {
  case "$1" in
    junyi) echo "configs/junyi.yaml" ;;
    xes3g5m) echo "configs/xes3g5m.yaml" ;;
    assist2012) echo "configs/assist2012.yaml" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

parquet_for() {
  echo "data/processed/$1.parquet"
}

preflight() {
  log "=== Preflight (dataset=${DATASET}) ==="
  "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  "$PYTHON" -c "import pykt"
  local cfg
  cfg="$(config_for "$DATASET")"
  if [[ "$DATASET" == "all" ]]; then
    for ds in junyi xes3g5m assist2012; do
      test -f "$(parquet_for "$ds")" || { echo "Missing $(parquet_for "$ds")"; exit 1; }
    done
  else
    test -f "$(parquet_for "$DATASET")" || { echo "Missing $(parquet_for "$DATASET")"; exit 1; }
    test -f "$cfg" || { echo "Missing $cfg"; exit 1; }
  fi
  log "Environment OK"
}

clear_pred_cache() {
  local ds="$1"
  log "=== Clear stale 5000-row prediction cache for ${ds} ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: would remove results/cache/${ds}_fold_*_*_train_only_preds.csv"
    return
  fi
  local n=0
  shopt -s nullglob
  for f in results/cache/"${ds}"_fold_*_*_train_only_preds.csv; do
    rm -f "$f"
    n=$((n + 1))
  done
  shopt -u nullglob
  log "Removed ${n} cached pred file(s) for ${ds}"
}

run_dataset() {
  local ds="$1"
  local cfg
  cfg="$(config_for "$ds")"
  clear_pred_cache "$ds"

  local -a args=(
    -m src.baseline_runner
    --config "$cfg"
    --cold-start-only
    --force-cold-start
    --log-level INFO
  )
  if [[ -n "$FOLD_IDX" ]]; then
    args+=(--fold-idx "$FOLD_IDX")
  fi
  if [[ -n "$MODELS" ]]; then
    args+=(--models "$MODELS")
  fi

  log "=== Run cold-start rerun: dataset=${ds} ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: $PYTHON ${args[*]}"
    return
  fi
  "$PYTHON" "${args[@]}" 2>&1 | tee -a "$LOG_DIR/a3_cold_start_${ds}.log"
}

regenerate_tables() {
  log "=== Regenerate cold-start TeX tables ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: generate_phase_c_tables + generate_cold_start_comparison + generate_paper_artifacts (cold-start)"
    return
  fi
  "$PYTHON" -m scripts.generate_phase_c_tables
  "$PYTHON" -m scripts.generate_cold_start_comparison
  "$PYTHON" -c "from scripts.generate_paper_artifacts import _write_cold_start_tex, _write_cold_start_by_stratum_tex; from pathlib import Path; _write_cold_start_by_stratum_tex(Path('results/tables/cold_start_by_stratum.tex')); _write_cold_start_tex(Path('results/tables/cold_start_metrics.tex'))"
}

verify_results() {
  local ds="${1:-$DATASET}"
  if [[ "$ds" == "all" ]]; then ds="junyi"; fi
  log "=== Verify very_cold strata for ${ds} ==="
  if [[ "$DRY_RUN" -eq 1 || "$SKIP_VERIFY" -eq 1 ]]; then
    return
  fi
  "$PYTHON" - "$ds" <<'PY'
import sys
import pandas as pd
import numpy as np

ds = sys.argv[1]
path = "results/tables/cold_start_metrics.csv"
df = pd.read_csv(path)
sub = df[(df["dataset"] == ds) & (df["stratum"] == "very_cold")].copy()
if sub.empty:
    print(f"WARNING: no very_cold rows for {ds}")
    sys.exit(0)

print(f"\n--- {ds} very_cold (post A3 rerun) ---")
for fold, part in sub.groupby("fold"):
    aucs = part.groupby("model")["auc"].first()
    n_vals = part.groupby("model")["n"].first()
    n_unique_auc = aucs.nunique(dropna=True)
    print(f"fold {fold}: n={n_vals.iloc[0]:.0f}, models={len(aucs)}, unique_AUC={n_unique_auc}")
    print(aucs.sort_index().to_string())
    if n_unique_auc <= 1 and len(aucs) > 1 and n_vals.iloc[0] >= 10:
        print("WARNING: all models share identical very_cold AUC — check full predictions")

if "n_discordant" in sub.columns:
    low = sub[sub["n_discordant"].fillna(0) < 10]
    if not low.empty:
        print(f"\nRows with n_discordant < 10 (AUC suppressed in TeX): {len(low)}")
PY
}

if [[ "$TABLES_ONLY" -eq 1 ]]; then
  regenerate_tables
  verify_results
  log "=== Done (tables-only) ==="
  exit 0
fi

preflight

if [[ "$DATASET" == "all" ]]; then
  for ds in junyi xes3g5m assist2012; do
    run_dataset "$ds"
  done
else
  run_dataset "$DATASET"
fi

regenerate_tables
verify_results

log "=== A3 cold-start rerun complete ==="
log "Next: sync results/tables/cold_start_* → paper/submission_APIN/ and rebuild PDF"
