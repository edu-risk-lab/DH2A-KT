#!/usr/bin/env python3
"""GS Hau A2: multi-seed stability matrix on XES3G5M fold 0 (clean L=400).

Runs PASS/FAIL arms and pyKT baselines at five seeds (42, 17, 1234, 0, 2024)
and appends rows to ``results/tables/a2_multiseed_matrix.csv``.

Do not Tee-Object into ``a2_multiseed_matrix.log`` while this script runs;
Python appends that file and Windows will raise PermissionError on lock clash.
Redirect stdout elsewhere instead, e.g. ``a2_multiseed_stdout.log``.

Usage:
    python scripts/27_a2_multiseed_matrix.py --device cuda
    python scripts/27_a2_multiseed_matrix.py --device cuda --arms simplekt,p0_dt_on
    python scripts/27_a2_multiseed_matrix.py --device cuda --force --seeds 42
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
DEFAULT_SEEDS = [42, 17, 1234, 0, 2024]
OUTPUT = REPO_ROOT / "results" / "tables" / "a2_multiseed_matrix.csv"
LOG = REPO_ROOT / "results" / "tables" / "a2_multiseed_matrix.log"
RUN_LOG_DIR = REPO_ROOT / "results" / "tables" / "a2_run_logs"
PYTHON = Path(os.environ.get("DH2A_PYTHON", sys.executable))

# pyKT baselines via 24_train_baselines_clean.py
PYKT_ARMS: dict[str, dict[str, str | int]] = {
    "simplekt": {"model": "simplekt", "tag_prefix": "C_simplekt_clean_L400"},
    "gikt": {
        "model": "gikt",
        "tag_prefix": "C_gikt_clean_L400_e30b16",
        "batch_size": 16,
        "epochs": 30,
        "patience": 5,
    },
    "akt": {"model": "akt", "tag_prefix": "C_akt_clean_L400"},
}

XES_DH2 = [
    "scripts/03_train_tier1.py",
    "configs/xes3g5m.yaml",
    "--fold",
    "0",
    "--architecture",
    "v4",
    "--use-questions",
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

DH2_ARM_NAMES = ("p0_dt_on", "hg_qkc_on", "hg_qkc_off")


def _dh2_extra_args(arm: str, seed: int) -> list[str]:
    """Per-arm CLI flags. Tags are seed-specific so checkpoints do not collide."""
    tag = f"{arm}_s{seed}"
    if arm == "p0_dt_on":
        return [
            "--question-graph",
            "--time-gap",
            "--time-gap-mode",
            "both",
            "--tag",
            tag,
        ]
    if arm == "hg_qkc_on":
        return ["--question-graph", "--tag", tag]
    if arm == "hg_qkc_off":
        return ["--tag", tag]
    raise KeyError(arm)


def _preflight_torch(device: str) -> None:
    """Fail fast when PyTorch/CUDA is unavailable (e.g. WDAC blocks c10.dll)."""
    probe = (
        "import torch; "
        f"assert torch.cuda.is_available() or '{device}'=='cpu', 'cuda unavailable'; "
        "print('torch_ok', torch.__version__)"
    )
    proc = subprocess.run(
        [str(PYTHON), "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(
            "PyTorch preflight failed — cannot start A2 GPU matrix.\n"
            f"  python: {PYTHON}\n"
            f"  error : {err}\n"
            "Fix: allow torch DLLs in Windows Application Control (WDAC), or set "
            "DH2A_PYTHON to a working interpreter, then rerun."
        )
    _log(f"preflight ok: {proc.stdout.strip()} via {PYTHON}")


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    for attempt in range(5):
        try:
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    print(f"WARN: could not append to {LOG}", flush=True)


def _already_done(arm: str, seed: int, backend: str, force: bool) -> bool:
    if force or not OUTPUT.exists():
        return False
    df = pd.read_csv(OUTPUT)
    mask = (df["arm"] == arm) & (df["seed"] == seed) & (df["backend"] == backend)
    if not mask.any():
        return False
    row = df.loc[mask].iloc[-1]
    # Skip reruns unless forced; reject legacy rows missing question-graph on p0_dt_on.
    if arm == "p0_dt_on" and bool(row.get("question_graph", True)) is False:
        return False
    return True


def _append_row(row: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        dup = (
            (existing["arm"] == row["arm"])
            & (existing["seed"] == row["seed"])
            & (existing["backend"] == row["backend"])
        )
        if dup.any():
            existing = existing.loc[~dup]
        table = pd.concat([existing, table], ignore_index=True)
    table.to_csv(OUTPUT, index=False)


def run_dh2(arm: str, seed: int, device: str, dry_run: bool, force: bool) -> int:
    if _already_done(arm, seed, "dh2", force):
        _log(f"SKIP dh2 {arm} seed={seed} (already in matrix)")
        return 0
    cmd = [
        str(PYTHON),
        *XES_DH2,
        "--device",
        device,
        "--seed",
        str(seed),
        *_dh2_extra_args(arm, seed),
        "--output",
        f"results/tables/a2_dh2_{arm}_s{seed}.csv",
    ]
    run_log = RUN_LOG_DIR / f"dh2_{arm}_s{seed}.log"
    _log(f"START dh2 {arm} seed={seed} -> {run_log.name}")
    _log(f"  cmd: {' '.join(cmd)}")
    if dry_run:
        return 0
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        _log(f"FAIL dh2 {arm} seed={seed} exit={proc.returncode} (see {run_log.name})")
        return proc.returncode
    out_csv = REPO_ROOT / "results" / "tables" / f"a2_dh2_{arm}_s{seed}.csv"
    auc = val_auc = None
    if out_csv.exists():
        res = pd.read_csv(out_csv)
        row = res[res["fold"] == 0].iloc[-1] if (res["fold"] == 0).any() else res.iloc[-1]
        auc = float(row.get("dh2_kt_auc", row.get("auc", float("nan"))))
        val_auc = float(row["val_auc"]) if pd.notna(row.get("val_auc")) else None
    question_graph = arm in ("p0_dt_on", "hg_qkc_on")
    _append_row(
        {
            "arm": arm,
            "backend": "dh2",
            "seed": seed,
            "fold": 0,
            "auc": auc,
            "val_auc": val_auc,
            "test_auc": auc,
            "question_graph": question_graph,
            "minutes": round((time.time() - started) / 60.0, 1),
            "status": "ok",
        }
    )
    _log(f"DONE dh2 {arm} seed={seed} val_auc={val_auc} auc={auc}")
    return 0


def run_pykt(arm: str, seed: int, device: str, dry_run: bool, force: bool) -> int:
    if _already_done(arm, seed, "pykt", force):
        _log(f"SKIP pykt {arm} seed={seed} (already in matrix)")
        return 0
    spec = PYKT_ARMS[arm]
    tag = f"{spec['tag_prefix']}_s{seed}"
    cmd = [
        str(PYTHON),
        "scripts/24_train_baselines_clean.py",
        "configs/xes3g5m.yaml",
        "--fold",
        "0",
        "--model",
        str(spec["model"]),
        "--window-mode",
        "chunked",
        "--mask-repeats",
        "--max-seq-len",
        "400",
        "--tag",
        tag,
        "--seed",
        str(seed),
        "--device",
        device,
        "--output",
        str(OUTPUT.with_name("a2_pykt_runs.csv")),
    ]
    if "batch_size" in spec:
        cmd.extend(["--batch-size", str(spec["batch_size"])])
    if "epochs" in spec:
        cmd.extend(["--epochs", str(spec["epochs"])])
    if "patience" in spec:
        cmd.extend(["--patience", str(spec["patience"])])
    run_log = RUN_LOG_DIR / f"pykt_{arm}_s{seed}.log"
    _log(f"START pykt {arm} seed={seed} -> {run_log.name}")
    if dry_run:
        return 0
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        _log(f"FAIL pykt {arm} seed={seed} exit={proc.returncode} (see {run_log.name})")
        return proc.returncode
    runs = pd.read_csv(OUTPUT.with_name("a2_pykt_runs.csv"))
    row = runs[runs["tag"] == tag].iloc[-1]
    _append_row(
        {
            "arm": arm,
            "backend": "pykt",
            "seed": seed,
            "fold": 0,
            "auc": float(row["auc"]),
            "val_auc": None,
            "test_auc": float(row["auc"]),
            "n_scored": int(row["n_scored"]),
            "question_graph": False,
            "minutes": round((time.time() - started) / 60.0, 1),
            "status": "ok",
        }
    )
    _log(f"DONE pykt {arm} seed={seed} auc={row['auc']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--arms",
        default=",".join(list(DH2_ARM_NAMES) + list(PYKT_ARMS)),
        help="Comma-separated arm names.",
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if the (arm, seed) pair is already in the matrix CSV.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not args.dry_run and args.device != "cpu":
        _preflight_torch(args.device)
    _log(f"=== A2 matrix arms={arms} seeds={seeds} force={args.force} python={PYTHON} ===")

    for arm in arms:
        for seed in seeds:
            if arm in DH2_ARM_NAMES:
                rc = run_dh2(arm, seed, args.device, args.dry_run, args.force)
            elif arm in PYKT_ARMS:
                rc = run_pykt(arm, seed, args.device, args.dry_run, args.force)
            else:
                _log(f"UNKNOWN arm {arm}")
                return 1
            if rc != 0:
                return rc
    _log("=== A2 matrix complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
