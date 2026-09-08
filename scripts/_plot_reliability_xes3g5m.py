"""Regenerate paper/figures/fig_reliability_xes3g5m.{pdf,png} from JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "tables" / "a6_reliability_xes3g5m_fold0.json"
OUT = ROOT / "paper" / "figures" / "fig_reliability_xes3g5m"

LABELS = {
    "p0_dt_on": "QKC-T",
    "hg_qkc_on": "Q to KC",
    "hg_qkc_off": "Pathway-off",
    "Zero+\u0394t": r"Zero+$\Delta t$",
}
COLORS = {
    "p0_dt_on": "#2A9D8F",
    "hg_qkc_on": "#E07A5F",
    "hg_qkc_off": "#6D597A",
    "Zero+\u0394t": "#B08900",
}
ORDER = ["p0_dt_on", "hg_qkc_on", "hg_qkc_off", "Zero+\u0394t"]


def main() -> None:
    blob = json.loads(SRC.read_text(encoding="utf-8"))
    models = blob["models"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#888888", label="perfect", zorder=0)
    counts = []
    for key in ORDER:
        counts.extend(b["count"] for b in models[key])
    cmax = max(counts)
    for key in ORDER:
        bins = models[key]
        xs = np.array([b["mean_pred"] for b in bins])
        ys = np.array([b["empirical_rate"] for b in bins])
        sizes = np.array([28.0 + 90.0 * (b["count"] / cmax) for b in bins])
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=COLORS[key],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            label=LABELS[key],
            zorder=2,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT.with_suffix(".pdf").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
