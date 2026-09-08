#!/usr/bin/env python3
"""Hau B8: ECE / Brier / NLL for Zero+Δt (no train).

Appends a Zero+Δt row to the existing DH² calibration CSV. Does not
overwrite QKC-T / Q←KC / pathway-off rows. Does not temperature-scale.

    python scripts/41_b8_zero_dt_calibration.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(__import__("os").environ.get("DH2A_PYTHON", sys.executable))
CKPT = REPO_ROOT / "results" / "checkpoints" / "xes3g5m_fold0_qkc_cm_zero_time_s42.pt"
EXISTING = REPO_ROOT / "results" / "tables" / "a6_calibration_xes3g5m_fold0.csv"
TMP = REPO_ROOT / "results" / "tables" / "a6_calibration_zero_dt_only.csv"
REL_JSON = REPO_ROOT / "results" / "tables" / "a6_reliability_zero_dt.json"
NAME = "Zero+Δt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    args = parser.parse_args()
    ckpt = args.checkpoint
    if not ckpt.is_file():
        print(f"SKIP: missing {ckpt}", flush=True)
        return 2
    cmd = [
        str(PYTHON),
        "scripts/30_a6_calibration_metrics.py",
        "configs/xes3g5m.yaml",
        "--fold",
        "0",
        "--device",
        args.device,
        "--checkpoint",
        f"{NAME}={ckpt.relative_to(REPO_ROOT) if str(ckpt).startswith(str(REPO_ROOT)) else ckpt}",
        "--max-seq-len",
        "400",
        "--window-mode",
        "chunked",
        "--output",
        str(TMP.relative_to(REPO_ROOT)),
        "--reliability-json",
        str(REL_JSON.relative_to(REPO_ROOT)),
    ]
    print("==>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode:
        return proc.returncode
    new = pd.read_csv(TMP)
    if EXISTING.exists():
        old = pd.read_csv(EXISTING)
        old = old[old["model"] != NAME]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(EXISTING, index=False)
    print(f"Appended {NAME} to {EXISTING}")
    if REL_JSON.exists() and (REPO_ROOT / "results/tables/a6_reliability_xes3g5m_fold0.json").exists():
        main_json = REPO_ROOT / "results" / "tables" / "a6_reliability_xes3g5m_fold0.json"
        blob = json.loads(main_json.read_text(encoding="utf-8"))
        extra = json.loads(REL_JSON.read_text(encoding="utf-8"))
        blob.setdefault("models", {}).update(extra.get("models", {}))
        main_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
