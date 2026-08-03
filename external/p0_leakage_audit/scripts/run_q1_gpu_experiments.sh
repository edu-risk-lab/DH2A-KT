#!/usr/bin/env bash
# Q1 GPU experiment orchestrator (24GB VRAM target: XES3G5M primary trio).
#
# Usage (repo root, venv + pip install -e ".[pykt]" + CUDA PyTorch):
#   chmod +x scripts/run_q1_gpu_experiments.sh
#   ./scripts/run_q1_gpu_experiments.sh              # all phases
#   ./scripts/run_q1_gpu_experiments.sh phase1       # GKT epochs30, seed 42 only
#   ./scripts/run_q1_gpu_experiments.sh phase2       # + seeds 17, 1234
#   ./scripts/run_q1_gpu_experiments.sh phase3       # primary trio matched, 3 seeds
#   ./scripts/run_q1_gpu_experiments.sh summarize    # merge results/q1 → tables
#
# Results land in results/q1/<tag>/ (isolated; main paper tables untouched).

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
PHASE="${1:-all}"
LOG_DIR="logs/q1"
mkdir -p "$LOG_DIR" results/q1

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_DIR/run.log"; }

check_env() {
  log "=== Environment check ==="
  $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"
  $PYTHON -c "import pykt" 2>/dev/null || { echo "Install pyKT: pip install -e '.[pykt]'"; exit 1; }
  test -f data/processed/xes3g5m.parquet || { echo "Missing data/processed/xes3g5m.parquet"; exit 1; }
  log "Parquet OK"
}

ensure_graphs() {
  log "=== Graph builder (skip if exports exist) ==="
  if test -f data/processed/xes3g5m/fold_0/e_pre_train_only.csv; then
    log "Fold exports present; skipping graph_builder"
    return
  fi
  $PYTHON -m src.graph_builder --config configs/xes3g5m.yaml 2>&1 | tee -a "$LOG_DIR/graph_builder.log"
}

run_gkt_epochs30() {
  local BASE_SEED="$1"
  shift
  local TAG="gkt_epochs30_s${BASE_SEED}"
  log "=== GKT epochs30 | split_base_seed=${BASE_SEED} | tag=${TAG} ==="
  if [ "${SKIP_CACHE_CLEAR:-0}" -ne 1 ]; then
    $PYTHON scripts/clear_baseline_cache.py --dataset xes3g5m --models gkt
  fi
  $PYTHON -m src.baseline_runner \
    --config configs/xes3g5m_gkt_epochs30.yaml \
    --split-base-seed "$BASE_SEED" \
    --isolated-results "$TAG" \
    --log-level INFO \
    "$@" \
    2>&1 | tee -a "$LOG_DIR/${TAG}.log"
}

run_primary_trio() {
  local BASE_SEED="$1"
  local TAG="trio_matched_s${BASE_SEED}"
  log "=== Primary trio (GKT+simpleKT+GIKT 30ep) | split_base_seed=${BASE_SEED} | tag=${TAG} ==="
  if [ "${SKIP_CACHE_CLEAR:-0}" -ne 1 ]; then
    $PYTHON scripts/clear_baseline_cache.py --dataset xes3g5m --models gkt,simplekt,gikt
  fi
  $PYTHON -m src.baseline_runner \
    --config configs/experiments/xes3g5m_primary_trio_matched.yaml \
    --split-base-seed "$BASE_SEED" \
    --isolated-results "$TAG" \
    --log-level INFO \
    2>&1 | tee -a "$LOG_DIR/${TAG}.log"
}

phase1() {
  check_env
  ensure_graphs
  run_gkt_epochs30 42
  summarize
}

phase2() {
  check_env
  ensure_graphs
  log "Running Phase 2 sequentially (16GB VRAM insufficient for parallel GKT)..."
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  log "Running seed 17 sequentially..."
  SKIP_CACHE_CLEAR=1 run_gkt_epochs30 17 "$@"

  log "Running seed 1234 sequentially..."
  SKIP_CACHE_CLEAR=1 run_gkt_epochs30 1234 "$@"

  log "Phase 2 sequential runs completed."
  summarize
}

phase3() {
  check_env
  ensure_graphs
  log "Running Phase 3 sequentially (16GB VRAM — one seed at a time)..."
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for S in 42 17 1234; do
    SKIP_CACHE_CLEAR=0 run_primary_trio "$S"
  done
  summarize
}

summarize() {
  log "=== Summarize Q1 results ==="
  $PYTHON scripts/summarize_q1_experiments.py
  log "Done. Inspect results/tables/q1_gkt_epochs30_ablation.tex"
}

case "$PHASE" in
  phase1) phase1 ;;
  phase2) phase2 ;;
  phase3) phase3 ;;
  summarize) summarize ;;
  all)
    phase1
    phase2
    phase3
    ;;
  *)
    echo "Unknown phase: $PHASE (phase1|phase2|phase3|summarize|all)"
    exit 1
    ;;
esac

log "=== Finished phase: $PHASE ==="
