#!/usr/bin/env python3
"""Generate Fig. 2: manipulation-check ΔAUC sweep for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Documented in docs/execution-plan.md §0.3 and M6 JSON.
# Graph-inert (pairwise + exercise path): auc_drop ≈ 0.0003 (FAIL).
# Graph-only v2 (chain + no exercise embed): auc_drop = 0.0380 (PASS).
labels = [
    "Pairwise +\nexercise path\n(graph-inert)",
    "Chain +\ngraph-only\n(v2)",
]
auc_drops = [0.0003, 0.0380]
pass_flags = [False, True]

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(4.2, 3.2))
colors = ["#9e9e9e", "#2a6f97"]
bars = ax.bar(labels, auc_drops, color=colors, width=0.55, edgecolor="black", linewidth=0.6)

ax.axhline(0.003, color="#c1121f", linestyle="--", linewidth=1.0, label="Pass threshold (illustrative)")
ax.set_ylabel(r"Manipulation $\Delta$AUC ($p{=}0.9$ node-drop)")
ax.set_ylim(0, 0.045)
ax.set_title("XES3G5M fold 0 — graph reliance gate")

for bar, drop, ok in zip(bars, auc_drops, pass_flags):
    tag = "PASS" if ok else "FAIL"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        drop + 0.0015,
        f"{drop:.4f}\n{tag}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#1b4332" if ok else "#6c757d",
    )

ax.legend(loc="upper left", frameon=False, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

pdf = OUT / "fig_manipulation_sweep.pdf"
png = OUT / "fig_manipulation_sweep.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, bbox_inches="tight")
print(f"Wrote {pdf.relative_to(ROOT)}")
print(f"Wrote {png.relative_to(ROOT)}")
