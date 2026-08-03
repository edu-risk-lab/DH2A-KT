#!/usr/bin/env bash
# Injection downstream AUC batch — XES3G5M fold 0, split seed 42 (Table S18).
# Usage:
#   bash scripts/run_injection_downstream_batch.sh
#   bash scripts/run_injection_downstream_batch.sh --skip-graphs
#   bash scripts/run_injection_downstream_batch.sh --collect-only
#   bash scripts/run_injection_downstream_batch.sh --clear-cache
#   bash scripts/run_injection_downstream_batch.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="${PYTHON:-python}"
CONFIG="configs/xes3g5m.yaml"
DATASET="xes3g5m"
FOLD=0
SPLIT_SEED=42
LOG_DIR="logs/q1"
CACHE_DIR="results/cache"

SKIP_GRAPHS=0
COLLECT_ONLY=0
CLEAR_CACHE=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --skip-graphs) SKIP_GRAPHS=1 ;;
    --collect-only) COLLECT_ONLY=1 ;;
    --clear-cache) CLEAR_CACHE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -Iseconds)] $*"
  echo "[$(date -Iseconds)] $*" >> "$LOG_DIR/injection_downstream_batch.log"
}

check_config() {
  log "=== Config preflight (Table S15 / 619c02cf) ==="
  grep -A3 'name: gikt' "$CONFIG" | grep -q 'batch_size: 8' || {
    echo "ERROR: baselines.gikt.hyperparams.batch_size must be 8"
    exit 1
  }
  grep -A3 'name: gkt' "$CONFIG" | grep -q 'batch_size: 4' || {
    echo "ERROR: baselines.gkt.hyperparams.batch_size must be 4"
    exit 1
  }
  log "Config OK: GKT batch=4, GIKT batch=8"
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

build_graphs() {
  log "=== Building injected graphs (CPU) ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: would run build_injected_graphs()"
    return
  fi
  "$PYTHON" -c "from scripts.run_injection_auc import build_injected_graphs; build_injected_graphs()" \
    2>&1 | tee -a "$LOG_DIR/injection_downstream_graphs.log"
  log "=== S17 sanity (run_leak_injection) ==="
  "$PYTHON" -m scripts.run_leak_injection 2>&1 | tee -a "$LOG_DIR/injection_downstream_s17.log"
}

clear_caches() {
  log "=== Clearing inject05/inject20 caches ==="
  for spec in \
    "inject05 simplekt" "inject20 simplekt" \
    "inject05 gikt" "inject20 gikt" \
    "inject05 gkt" "inject20 gkt"
  do
    read -r arm model <<< "$spec"
    base="${DATASET}_fold_${FOLD}_${model}_s${SPLIT_SEED}_${arm}"
    for suffix in _result.json _preds.csv; do
      path="$CACHE_DIR/${base}${suffix}"
      if [[ -f "$path" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
          log "DryRun: would remove $path"
        else
          rm -f "$path"
          log "Removed $path"
        fi
      fi
    done
    workdir="results/pykt_work/xes3g5m/fold_${FOLD}_seed_${SPLIT_SEED}/${arm}"
    if [[ "$CLEAR_CACHE" -eq 1 && -d "$workdir" ]]; then
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "DryRun: would remove $workdir"
      else
        rm -rf "$workdir"
        log "Removed workdir $workdir"
      fi
    fi
  done
}

run_job() {
  local arm="$1"
  local model="$2"
  local tag="${arm}_${model}"
  local log_path="$LOG_DIR/injection_downstream_${tag}.log"
  log "=== Job: fold=$FOLD arm=$arm model=$model ==="

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: $PYTHON -m src.baseline_runner --config $CONFIG --baseline-backend pykt --fold-idx $FOLD --seed $SPLIT_SEED --split-base-seed $SPLIT_SEED --graph-construction $arm --models $model"
    return
  fi

  "$PYTHON" -m src.baseline_runner \
    --config "$CONFIG" \
    --baseline-backend pykt \
    --fold-idx "$FOLD" \
    --seed "$SPLIT_SEED" \
    --split-base-seed "$SPLIT_SEED" \
    --graph-construction "$arm" \
    --models "$model" \
    --log-level INFO \
    2>&1 | tee -a "$log_path"

  local cache_json="$CACHE_DIR/${DATASET}_fold_${FOLD}_${model}_s${SPLIT_SEED}_${arm}_result.json"
  test -f "$cache_json"
  log "OK: $cache_json"
}

collect_table() {
  log "=== Collecting Table S18 (collect_injection_auc) ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DryRun: would run python -m scripts.collect_injection_auc"
    return
  fi
  "$PYTHON" -m scripts.collect_injection_auc 2>&1 | tee -a "$LOG_DIR/injection_downstream_collect.log"
}

log "=== Injection downstream batch start ==="
check_config
check_env

if [[ "$COLLECT_ONLY" -eq 1 ]]; then
  collect_table
  log "=== Done (collect only) ==="
  exit 0
fi

if [[ "$SKIP_GRAPHS" -eq 0 ]]; then
  build_graphs
fi

if [[ "$CLEAR_CACHE" -eq 1 ]]; then
  clear_caches
fi

job=0
for spec in \
  "inject05 simplekt" "inject20 simplekt" \
  "inject05 gikt" "inject20 gikt" \
  "inject05 gkt" "inject20 gkt"
do
  job=$((job + 1))
  read -r arm model <<< "$spec"
  log "--- Progress $job/6 ---"
  run_job "$arm" "$model"
done

collect_table
log "=== Injection downstream batch finished ==="
