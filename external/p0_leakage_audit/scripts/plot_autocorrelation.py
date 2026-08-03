#!/usr/bin/env python3
"""Bar chart: KC-repeat rate vs deep-KT and BKT AUC across benchmarks.

Reads the reproducible artifacts
- results/tables/autocorrelation_stats.csv  (KC-repeat rate)
- results/tables/baseline_results.csv       (simpleKT / BKT AUC, train-only)
and renders results/figures/fig_autocorr_vs_auc.pdf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Ordered by ascending KC-repeat rate so both repeat rate and deep AUC rise
# left-to-right.
DATASETS = ["XES3G5M", "ASSISTments 2012", "Junyi Academy"]
CSV_KEY = {"XES3G5M": "xes3g5m", "ASSISTments 2012": "assist2012", "Junyi Academy": "junyi"}
DEEP_MODEL = "simplekt"


def _load_auc(baseline: pd.DataFrame, dataset_key: str, model: str) -> float:
    sub = baseline[
        (baseline["dataset"] == dataset_key)
        & (baseline["model"] == model)
        & (baseline["graph_construction"] == "train_only")
    ]
    return float(sub["auc"].iloc[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autocorr", default="results/tables/autocorrelation_stats.csv")
    parser.add_argument("--baseline", default="results/tables/baseline_results.csv")
    parser.add_argument("--out", default="results/figures/fig_autocorr_vs_auc.pdf")
    args = parser.parse_args()

    autocorr = pd.read_csv(ROOT / args.autocorr).set_index("dataset")
    baseline = pd.read_csv(ROOT / args.baseline)

    kc_repeat, deep_auc, bkt_auc = [], [], []
    for label in DATASETS:
        key = CSV_KEY[label]
        kc_repeat.append(float(autocorr.loc[label, "kc_repeat_rate"]))
        deep_auc.append(_load_auc(baseline, key, DEEP_MODEL))
        bkt_auc.append(_load_auc(baseline, key, "bkt"))

    x = np.arange(len(DATASETS))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    b1 = ax.bar(x - width, kc_repeat, width, label="KC-repeat rate", color="#DD8452")
    b2 = ax.bar(x, bkt_auc, width, label="BKT AUC", color="#B0B0B0")
    b3 = ax.bar(x + width, deep_auc, width, label="simpleKT AUC", color="#4C72B0")

    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Rate / AUC")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=9)
    ax.set_title("Consecutive KC-repeat rate vs. KT performance", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    print(pd.DataFrame({"dataset": DATASETS, "kc_repeat": kc_repeat,
                        "bkt_auc": bkt_auc, "deep_auc": deep_auc}).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
