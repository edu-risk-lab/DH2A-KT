#!/usr/bin/env bash
# Set up P0 dependency + Python environment.
#
# P0 is currently vendored as a static, pinned snapshot at
# external/p0_leakage_audit (see .p0_vendored_commit.txt inside it) rather
# than a live git submodule — the authoring environment's mounted filesystem
# did not support the unlink/relink operations git submodule needs. On a
# normal machine you can convert it to a live submodule instead:
#
#   rm -rf external/p0_leakage_audit
#   git submodule add https://github.com/edu-risk-lab/leakage-controlled-kt-audit.git external/p0_leakage_audit
#   git submodule update --init --recursive
#
# Either way, after P0's source is present at external/p0_leakage_audit:

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Python version check (P0 + DH2A-KT both require >=3.10) =="
python3 --version

python -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r external/p0_leakage_audit/requirements.txt

echo ""
echo "== pykt-toolkit (P0's own baseline dependency, not vendored — see"
echo "   docs/execution-plan.md M0) =="
if [ ! -d external/p0_leakage_audit/third_party/pykt-toolkit ]; then
  git clone https://github.com/pykt-team/pykt-toolkit.git \
    external/p0_leakage_audit/third_party/pykt-toolkit
fi
pip install -e external/p0_leakage_audit/third_party/pykt-toolkit

echo ""
echo "== Output directory skeleton (results/, checkpoints/, logs/) =="
mkdir -p results/tables results/figures checkpoints logs
echo "  results/tables, results/figures, checkpoints, logs ready."

echo ""
echo "== Tier 1 GPU stack (NOT installed by this script — CUDA-version"
echo "   specific) =="
echo "  Run 'nvidia-smi' to confirm the GPU/driver, then install torch"
echo "  matching your CUDA build, e.g. for CUDA 12.1:"
echo "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo "  then torch_geometric per https://pytorch-geometric.readthedocs.io"
echo "  (install instructions depend on the exact torch+CUDA combo — do not"
echo "  blindly 'pip install torch_geometric' without matching wheels)."
echo "  Uncomment the Tier 1 lines in requirements.txt once installed so"
echo "  future 'pip install -r requirements.txt' runs stay reproducible."

echo ""
echo "Next: follow external/p0_leakage_audit/README.md section 3 (Data"
echo "download and preparation) to place XES3G5M / ASSISTments2012 / Junyi"
echo "raw files under external/p0_leakage_audit/data/raw/<dataset>/, then run:"
echo "  python -m src.preprocess --config external/p0_leakage_audit/configs/xes3g5m.yaml"
echo "  (from inside external/p0_leakage_audit/, per its own README)"
echo ""
echo "Then verify provenance before trusting any baseline reuse:"
echo "  python scripts/01_verify_p0_preprocessing_match.py configs/xes3g5m.yaml"
echo "  python -m pytest tests/ -v"
echo "See docs/REPRODUCIBILITY.md for the full checklist."
