#!/usr/bin/env python3
"""GS Hau A7: query-side vs LSTM-side Linear Δt ablation on XES3G5M fold 0.

Trains three time-gap injection modes under the clean L=400 protocol:
  * both   — LSTM + next-step query (recommended p0_dt_on)
  * lstm   — state / forgetting branch only
  * query  — gap to the scored next-step query only

Twin for the registered gate is hg_qkc_on (same seed, no time-gap).
Skips arms already present in ``results/tables/a7_query_dt_ablation.csv``.

Usage:
    python scripts/31_a7_query_dt_ablation.py --device cuda
    python scripts/31_a7_query_dt_ablation.py --device cuda --force
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "results" / "tables" / "a7_query_dt_ablation.csv"
LOG = REPO_ROOT / "results" / "tables" / "a7_query_dt_ablation.log"
RUN_LOG_DIR = REPO_ROOT / "results" / "tables" / "a7_run_logs"
PYTHON = Path(os.environ.get("DH2A_PYTHON", sys.executable))
DEFAULT_SEED = 42
VAL_GATE = 0.002

ARMS: dict[str, dict[str, str | list[str]]] = {
    "dt_both": {
        "tag": "a7_dt_both",
        "extra": ["--time-gap", "--time-gap-mode", "both"],
    },
    "dt_lstm": {
        "tag": "a7_dt_lstm",
        "extra": ["--time-gap", "--time-gap-mode", "lstm"],
    },
    "dt_query": {
        "tag": "a7_dt_query",
        "extra": ["--time-gap", "--time-gap-mode", "query"],
    },
}

BASE = [
    "scripts/03_train_tier1.py",
    "configs/xes3g5m.yaml",
    "--fold",
    "0",
    "--architecture",
    "v4",
    "--use-questions",
    "--question-graph",
    "--no-graph",
    "--no-session",
    "--mask-repeats",
    "--window-mode",
    "chunked",
    "--max-seq-len",
    "400",
    "--batch-size",
    "16",
    "--epochs",
    "30",
    "--val-frac",
    "0.1",
    "--early-stop-patience",
    "5",
    "--graph-dropout",
    "0",
    "--graph-sensitivity-weight",
    "0",
]


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _already_done(arm: str, force: bool) -> bool:
    if force or not OUTPUT.exists():
        return False
    df = pd.read_csv(OUTPUT)
    return (df["arm"] == arm).any()


def _append_row(row: dict) -> None:
    table = pd.DataFrame([row])
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        dup = existing["arm"] == row["arm"]
        if dup.any():
            existing = existing.loc[~dup]
        table = pd.concat([existing, table], ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)


def _twin_val(seed: int) -> float | None:
    twin_csv = REPO_ROOT / "results" / "tables" / f"a2_dh2_hg_qkc_on_s{seed}.csv"
    if not twin_csv.exists():
        legacy = REPO_ROOT / "results" / "tables" / "dh2_kt_hg_qkc_on_vs_p0.csv"
        if legacy.exists():
            df = pd.read_csv(legacy)
            row = _fold0_row(df)
            return float(row["val_auc"])
        return None
    df = pd.read_csv(twin_csv)
    row = _fold0_row(df)
    return float(row["val_auc"])


def _fold0_row(df: pd.DataFrame) -> pd.Series:
    sub = df[df["fold"].astype(str) == "0"]
    if sub.empty:
        sub = df[df["fold"] == 0]
    if sub.empty:
        raise IndexError("no fold-0 row")
    return sub.iloc[0]


def _import_both_from_a2(seed: int) -> bool:
    """Reuse A2 p0_dt_on row when config matches (both mode)."""
    a2_csv = REPO_ROOT / "results" / "tables" / f"a2_dh2_p0_dt_on_s{seed}.csv"
    if not a2_csv.exists():
        return False
    res = pd.read_csv(a2_csv)
    row = _fold0_row(res)
    val_auc = float(row["val_auc"])
    test_auc = float(row.get("dh2_kt_auc", row.get("auc", float("nan"))))
    twin_val = _twin_val(seed)
    delta_val = val_auc - twin_val if twin_val is not None else float("nan")
    gate_pass = delta_val >= VAL_GATE if twin_val is not None else None
    _append_row(
        {
            "arm": "dt_both",
            "time_gap_mode": "both",
            "seed": seed,
            "fold": 0,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "twin_val_auc": twin_val,
            "delta_val": delta_val,
            "gate_pass": gate_pass,
            "minutes": 0.0,
            "status": "imported_a2",
        }
    )
    _log(f"IMPORT dt_both from A2 seed={seed} val_auc={val_auc} test_auc={test_auc}")
    return True


def run_arm(arm: str, seed: int, device: str, force: bool, dry_run: bool) -> int:
    if _already_done(arm, force):
        _log(f"SKIP {arm} (already in {OUTPUT.name})")
        return 0
    if arm == "dt_both" and not force and _import_both_from_a2(seed):
        return 0
    spec = ARMS[arm]
    tag = str(spec["tag"])
    out_csv = REPO_ROOT / "results" / "tables" / f"a7_{arm}_s{seed}.csv"
    cmd = [
        str(PYTHON),
        *BASE,
        "--device",
        device,
        "--seed",
        str(seed),
        *spec["extra"],
        "--tag",
        tag,
        "--output",
        str(out_csv.relative_to(REPO_ROOT)),
    ]
    run_log = RUN_LOG_DIR / f"{arm}_s{seed}.log"
    _log(f"START {arm} seed={seed} -> {run_log.name}")
    _log(f"  cmd: {' '.join(cmd)}")
    if dry_run:
        return 0
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        _log(f"FAIL {arm} exit={proc.returncode} (see {run_log.name})")
        return proc.returncode
    res = pd.read_csv(out_csv)
    row = _fold0_row(res)
    val_auc = float(row["val_auc"])
    test_auc = float(row.get("dh2_kt_auc", row.get("auc", float("nan"))))
    twin_val = _twin_val(seed)
    delta_val = val_auc - twin_val if twin_val is not None else float("nan")
    gate_pass = delta_val >= VAL_GATE if twin_val is not None else None
    _append_row(
        {
            "arm": arm,
            "time_gap_mode": arm.replace("dt_", ""),
            "seed": seed,
            "fold": 0,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "twin_val_auc": twin_val,
            "delta_val": delta_val,
            "gate_pass": gate_pass,
            "minutes": round((time.time() - started) / 60.0, 1),
            "status": "ok",
        }
    )
    _log(f"DONE {arm} val_auc={val_auc} test_auc={test_auc} delta_val={delta_val} pass={gate_pass}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _log(f"=== A7 query-dt ablation seed={args.seed} force={args.force} ===")
    for arm in ARMS:
        rc = run_arm(arm, args.seed, args.device, args.force, args.dry_run)
        if rc != 0:
            return rc
    _log("=== A7 complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
