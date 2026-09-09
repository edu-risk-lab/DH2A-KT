#!/usr/bin/env python3
"""Generate Fig. 2: manipulation-check ΔAUC sweep for the paper."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical fold-0 node-drop: results/tables/b9_same_ckpt_destruction.csv
# (shared-clean evaluator). Do not hard-code a second ΔAUC.

b9 = ROOT / "results" / "tables" / "b9_same_ckpt_destruction.csv"
fold0_node_drop = None
with b9.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["fold"] == "0" and row["operator"] == "node_drop":
            fold0_node_drop = float(row["auc_drop_from_shared"])
            break
if fold0_node_drop is None:
    raise SystemExit(f"missing fold-0 node_drop in {b9}")

# Historical graph-inert path is not in B9; keep the archived 0.0003 and
# label it historical so it cannot be read as the B9 evaluator.
labels = [
    "Pairwise +\nexercise path\n(historical inert)",
    "Chain +\ngraph-only\n(B9 shared-clean)",
]
auc_drops = [0.0003, fold0_node_drop]
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

ax.axhline(0.003, color="#c1121f", linestyle="--", linewidth=1.0,
           label=r"Illustrative floor ($\approx$10$\times$ inert noise)")
ax.set_ylabel(r"Manipulation $\Delta$AUC ($p{=}0.9$ node-drop)")
ax.set_ylim(0, max(0.045, fold0_node_drop + 0.012))
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
