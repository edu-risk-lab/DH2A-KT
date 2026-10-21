"""Round-2 hyperparameter grid for AKT and simpleKT (docs/preregistration.md, Section 5).

Stage ``grid``: 12 configurations per model on seed 42, scored on the
validation split. Stage ``seeds``: the configuration with the highest
validation AUC, trained on the five seeds of the existing baseline rows.
Test AUC is written but never read for selection.

Each configuration gets its own pyKT work directory (the engine reloads a
checkpoint it finds there), and finished rows are skipped.

    python scripts/57_baseline_grid.py --stage grid --device cuda
    python scripts/57_baseline_grid.py --stage seeds --device cuda
    python scripts/57_baseline_grid.py --stage select
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "tables" / "baseline_grid_xes3g5m_fold0.csv"
MODELS = ("akt", "simplekt")
LRS = ("1e-3", "3e-4")
DROPOUTS = ("0.05", "0.10", "0.20")
D_MODELS = ("128", "256")
SELECT_SEED = 42
FINAL_SEEDS = (0, 17, 42, 1234, 2024)


def _configs() -> list[dict[str, str]]:
    return [
        {"lr": lr, "dropout": dr, "d_model": dm}
        for lr, dr, dm in itertools.product(LRS, DROPOUTS, D_MODELS)
    ]


def _tag(model: str, cfg: dict[str, str], seed: int) -> str:
    return f"G_{model}_lr{cfg['lr']}_do{cfg['dropout']}_dm{cfg['d_model']}_s{seed}"


def _done_tags() -> set[str]:
    if not OUT.exists():
        return set()
    return set(pd.read_csv(OUT)["tag"].astype(str))


def _run(model: str, cfg: dict[str, str], seed: int, device: str) -> int:
    tag = _tag(model, cfg, seed)
    if tag in _done_tags():
        print(f"SKIP {tag}")
        return 0
    cmd = [
        sys.executable,
        "scripts/24_train_baselines_clean.py",
        "configs/xes3g5m.yaml",
        "--fold", "0",
        "--model", model,
        "--window-mode", "chunked",
        "--mask-repeats",
        "--max-seq-len", "400",
        "--seed", str(seed),
        "--device", device,
        "--tag", tag,
        "--output", str(OUT.relative_to(REPO)),
        "--report-valid",
        "--hp", f"lr={cfg['lr']}",
        "--hp", f"dropout={cfg['dropout']}",
        "--hp", f"d_model={cfg['d_model']}",
    ]
    print("launch:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def _select() -> dict[str, dict[str, str]]:
    val = pd.read_csv(OUT).drop_duplicates("tag", keep="last").set_index("tag")["val_auc"]
    chosen = {}
    for model in MODELS:
        scored = {}
        for cfg in _configs():
            tag = _tag(model, cfg, SELECT_SEED)
            if tag not in val.index or pd.isna(val[tag]):
                raise SystemExit(f"{model}: grid row {tag} missing; finish the grid first")
            scored[tag] = (float(val[tag]), cfg)
        best_tag = max(scored, key=lambda t: scored[t][0])
        chosen[model] = scored[best_tag][1]
        print(f"{model}: selected {chosen[model]} val_auc={scored[best_tag][0]:.6f} ({best_tag})")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("grid", "seeds", "select"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]

    if args.stage == "select":
        _select()
        return 0
    if args.stage == "grid":
        for model in models:
            for cfg in _configs():
                rc = _run(model, cfg, SELECT_SEED, args.device)
                if rc:
                    return rc
        return 0
    chosen = _select()
    for model in models:
        for seed in FINAL_SEEDS:
            rc = _run(model, chosen[model], seed, args.device)
            if rc:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
