#!/usr/bin/env bash
# GPU server dispatcher — pending revision jobs (APIN 2026-07-23).
#
# Usage:
#   bash scripts/run_gpu_server_pending.sh              # list jobs
#   bash scripts/run_gpu_server_pending.sh a2_2b        # DDR seed-17 baseline rerun
#   bash scripts/run_gpu_server_pending.sh a3_coldstart # A3.5 cold-start full preds (Junyi)
#   bash scripts/run_gpu_server_pending.sh injection    # Table S18 (if cache missing)
#   bash scripts/run_gpu_server_pending.sh crossref     # CPU: verify AUC consistency
#   bash scripts/run_gpu_server_pending.sh all          # a2_2b then crossref (injection skipped if done)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
JOB="${1:-help}"

log() { echo "[$(date -Iseconds)] $*"; }

preflight() {
  log "=== Preflight ==="
  "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  test -f data/processed/xes3g5m.parquet || { echo "Missing xes3g5m.parquet — see README §3"; exit 1; }
}

job_a2_2b() {
  log "=== Job A2.2b: DDR GKT seed-17 fold-0 baseline ==="
  bash scripts/run_a2_2b_ddr_seed17_baseline.sh
}

job_injection() {
  log "=== Job A1: injection downstream (Table S18) ==="
  local missing=0
  for arm in inject05 inject20; do
    for model in simplekt gkt gikt; do
      f="results/cache/xes3g5m_fold_0_${model}_s42_${arm}_result.json"
      if [[ ! -f "$f" ]]; then
        log "Missing: $f"
        missing=1
      fi
    done
  done
  if [[ "$missing" -eq 0 ]]; then
    log "All inject05/20 caches present — collect only"
    bash scripts/run_injection_downstream_batch.sh --collect-only
    return
  fi
  bash scripts/run_injection_downstream_batch.sh
}

job_crossref() {
  log "=== Job A1.3: cross-reference AUC numbers (CPU) ==="
  "$PYTHON" -m scripts.crossref_auc_numbers
}

job_a3_coldstart() {
  log "=== Job A3.5: cold-start rerun (full predictions, Junyi default) ==="
  bash scripts/run_a3_cold_start_rerun.sh --dataset junyi
}

case "$JOB" in
  help|-h|--help)
    cat <<'EOF'
GPU server pending jobs (APIN revision):

  a2_2b       Rerun DDR GKT baseline fold-0 seed-17 (fixes placeholder AUC)
  a3_coldstart  A3.5 cold-start rerun with full pyKT preds (Junyi; optional GPU)
  injection   Run/collect Table S18 injection downstream AUC
  crossref    Verify baseline vs S18 vs cache (CPU)
  all         a2_2b + crossref (injection only if caches missing)

Examples:
  bash scripts/run_gpu_server_pending.sh a2_2b
  nohup bash scripts/run_gpu_server_pending.sh a3_coldstart > logs/q1/nohup_a3_coldstart.log 2>&1 &
EOF
    ;;
  a2_2b)
    preflight
    job_a2_2b
    ;;
  a3_coldstart|a3)
    preflight
    test -f data/processed/junyi.parquet || { echo "Missing junyi.parquet — see README §3"; exit 1; }
    job_a3_coldstart
    ;;
  injection)
    preflight
    job_injection
    ;;
  crossref)
    job_crossref
    ;;
  all)
    preflight
    job_a2_2b
    job_crossref
    ;;
  *)
    echo "Unknown job: $JOB (try: help, a2_2b, a3_coldstart, injection, crossref, all)"
    exit 1
    ;;
esac
