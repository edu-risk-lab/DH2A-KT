#!/usr/bin/env python3
"""ECE / Brier / NLL from saved pyKT ``test_predictions.npz`` (no train, CPU).

Each npz must contain ``ts`` (labels) and ``ps`` (probabilities), as written by
``scripts/24_train_baselines_clean.py``.

Usage:
    python scripts/35_pykt_calibration_from_npz.py \\
        --npz "GIKT=results/pykt_work_clean/xes3g5m/fold_0/C_gikt_clean_L400_e30b16_s42/test_predictions.npz" \\
        --npz "AKT=results/pykt_work_clean/xes3g5m/fold_0/C_akt_clean_L400_s42/test_predictions.npz" \\
        --npz "simpleKT=results/pykt_work_clean/xes3g5m/fold_0/C_simplekt_clean_L400_s42/test_predictions.npz" \\
        --output results/tables/a6_calibration_pykt_xes3g5m_fold0.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.diagnostics.black_confidence import (  # noqa: E402
    brier_score_np,
    expected_calibration_error_np,
    negative_log_likelihood_np,
)


def _parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"expected NAME=PATH, got {spec!r}")
    name, raw = spec.split("=", 1)
    return name, Path(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "a6_calibration_pykt_xes3g5m_fold0.csv",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for spec in args.npz:
        name, path = _parse_spec(spec)
        if not path.is_file():
            raise SystemExit(f"missing npz: {path}")
        data = np.load(path)
        missing = {"ts", "ps"} - set(data.files)
        if missing:
            raise SystemExit(f"{path} missing {sorted(missing)}; holds {data.files}")
        labels = np.asarray(data["ts"]).astype(np.int8).ravel()
        probs = np.asarray(data["ps"]).astype(np.float64).ravel()
        if labels.size != probs.size:
            raise SystemExit(f"{path}: ts/ps length mismatch {labels.size} vs {probs.size}")
        n_pos = float(labels.sum())
        n_neg = float(labels.size - n_pos)
        if n_pos == 0.0 or n_neg == 0.0:
            auc = float("nan")
        else:
            order = np.argsort(probs, kind="stable")
            ranks = np.empty(labels.size, dtype=np.float64)
            ranks[order] = np.arange(1, labels.size + 1, dtype=np.float64)
            auc = float(
                (ranks[labels == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
            )
        row = {
            "model": name,
            "n_scored": int(labels.size),
            "auc": auc,
            "brier": float(brier_score_np(probs, labels)),
            "nll": float(negative_log_likelihood_np(probs, labels)),
            "ece": float(expected_calibration_error_np(probs, labels, n_bins=args.ece_bins)),
            "ece_bins": args.ece_bins,
            "path": str(path),
        }
        rows.append(row)
        print(
            f"[{name}] n={row['n_scored']} AUC={row['auc']:.4f} "
            f"Brier={row['brier']:.4f} NLL={row['nll']:.4f} ECE={row['ece']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
