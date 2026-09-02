#!/usr/bin/env python3
"""GS Hau A6: ECE (15-bin), Brier, and NLL on clean-protocol test predictions.

Scores saved DH²-KT checkpoints without retraining. Uses the same clean
protocol as headline AUC (mask-repeats, chunked windows, test split).

Usage:
    python scripts/30_a6_calibration_metrics.py configs/xes3g5m.yaml --fold 0 --device cuda \\
        --checkpoint p0_dt_on=results/checkpoints/xes3g5m_fold0_p0_dt_on_s42.pt \\
        --max-seq-len 400 --window-mode chunked

Output: ``results/tables/a6_calibration_xes3g5m_fold0.csv``
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.diagnostics.black_confidence import (  # noqa: E402
    brier_score_np,
    expected_calibration_error_np,
    negative_log_likelihood_np,
)
from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids  # noqa: E402
from dh2a_kt.train.tier1 import collect_predictions, resolve_training_budget  # noqa: E402


def _reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0,
                    "mean_pred": None,
                    "empirical_rate": None,
                }
            )
            continue
        rows.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "mean_pred": float(probs[in_bin].mean()),
                "empirical_rate": float(labels[in_bin].mean()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--window-mode", default="chunked", choices=("first", "last", "chunked"))
    parser.add_argument("--max-seq-len", type=int, default=400)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "a6_calibration_xes3g5m_fold0.csv",
    )
    parser.add_argument(
        "--reliability-json",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "a6_reliability_xes3g5m_fold0.json",
    )
    args = parser.parse_args()

    from dh2a_kt.models.dh2_kt import empty_hyperedge_index  # noqa: WPS433
    from dh2a_kt.train.checkpoint import load_trained_fold  # noqa: WPS433
    from dh2a_kt.train.greykt import make_sequence_loader  # noqa: WPS433

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=args.max_seq_len,
    )
    budget = replace(budget, batch_size=int(args.eval_batch_size))

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    eval_df = splits["test"]

    rows: list[dict] = []
    reliability: dict[str, list[dict]] = {}
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        trained = load_trained_fold(REPO_ROOT / raw_path, device=args.device)
        index = trained.clean_hyperedge_index
        if not trained.clean_hyperedges:
            index = empty_hyperedge_index(
                trained.device, kinds=tuple(trained.model.config.hyperedge_kinds)
            )
        loader = make_sequence_loader(
            eval_df, trained, budget, shuffle=False, window_mode=args.window_mode
        )
        probs, labels = collect_predictions(
            trained.model, loader, index, trained.device, mask_repeats=True
        )
        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
        row = {
            "model": name,
            "fold": args.fold,
            "protocol": "clean_test_chunked",
            "n_scored": int(len(probs)),
            "auc": auc,
            "brier": brier_score_np(probs, labels),
            "nll": negative_log_likelihood_np(probs, labels),
            "ece": expected_calibration_error_np(probs, labels, n_bins=args.ece_bins),
            "ece_bins": args.ece_bins,
            "checkpoint": str(raw_path),
        }
        rows.append(row)
        reliability[name] = _reliability_bins(probs, labels, n_bins=args.ece_bins)
        print(
            f"[{name}] n={row['n_scored']} AUC={row['auc']:.4f} "
            f"Brier={row['brier']:.4f} NLL={row['nll']:.4f} ECE={row['ece']:.4f}"
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    args.reliability_json.write_text(
        json.dumps(
            {
                "fold": args.fold,
                "ece_bins": args.ece_bins,
                "protocol": "clean_test_chunked",
                "models": reliability,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.reliability_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
