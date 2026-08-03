#!/usr/bin/env python3
"""Analyse DDR -> downstream graph-KT accuracy (``small_downstream``).

Consumes results/tables/ddr_downstream.csv (produced by scripts/ddr_downstream.py)
and produces:
  - results/tables/ddr_downstream_summary.csv : per (dataset, model, operator, p) mean
    DDR, mean test AUC, and mean AUC drop vs the DDR=0 baseline graph.
  - results/tables/ddr_downstream.tex         : LaTeX summary table.
  - results/figures/fig_ddr_downstream.pdf     : scatter of DDR vs AUC drop with
    a pooled regression line and Pearson/Spearman correlations per dataset.

The headline number is the correlation between DDR and downstream AUC drop: a
positive, significant correlation shows that the structural DDR diagnostic
predicts accuracy degradation, and that the prerequisite-preserving operator
(low DDR at matched budget) degrades accuracy least.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# J05: color + marker (scatter has no linestyle) for grayscale separability.
_OP_STYLE = {
    "attr_mask": {"color": "#937860", "marker": "o"},
    "edge_drop": {"color": "#4C72B0", "marker": "s"},
    "node_drop": {"color": "#C44E52", "marker": "^"},
    "prereq_preserve": {"color": "#55A868", "marker": "D"},
    "subgraph": {"color": "#8172B2", "marker": "v"},
}
_OP_COLOR = {k: v["color"] for k, v in _OP_STYLE.items()}


def _baseline_auc(df: pd.DataFrame) -> dict[tuple, float]:
    """Per (dataset, model, fold[, split_seed]) AUC of the DDR=0 baseline graph."""
    base = df[df["operator"] == "none"]
    use_seed = "split_seed" in df.columns
    out: dict[tuple, float] = {}
    for r in base.itertuples():
        if pd.isna(r.auc):
            continue
        key: tuple
        if use_seed and pd.notna(getattr(r, "split_seed", np.nan)):
            key = (str(r.dataset), str(r.model), int(r.fold), int(r.split_seed))
        else:
            key = (str(r.dataset), str(r.model), int(r.fold))
        out[key] = float(r.auc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-csv", type=Path, default=ROOT / "results/tables/ddr_downstream.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "results/tables/ddr_downstream_summary.csv")
    parser.add_argument("--out-tex", type=Path, default=ROOT / "results/tables/ddr_downstream.tex")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "results/figures/fig_ddr_downstream.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df[pd.notna(df["auc"])].copy()
    if df.empty:
        raise SystemExit(f"No AUC rows in {args.in_csv}; run scripts/ddr_downstream.py on the GPU server first.")
    if "model" not in df.columns:
        df["model"] = "gkt"  # backward-compat with pre-multi-model CSVs

    base = _baseline_auc(df)
    use_seed = "split_seed" in df.columns

    def _lookup(d, m, f, s) -> float:
        if use_seed and pd.notna(s):
            return base.get((str(d), str(m), int(f), int(s)), np.nan)
        return base.get((str(d), str(m), int(f)), np.nan)

    seeds = df["split_seed"] if use_seed else [np.nan] * len(df)
    df["baseline_auc"] = [
        _lookup(d, m, f, s) for d, m, f, s in zip(df["dataset"], df["model"], df["fold"], seeds)
    ]
    df["auc_drop"] = df["baseline_auc"] - df["auc"]  # positive = degradation

    pert = df[df["operator"] != "none"].copy()

    # ---- per (dataset, model, operator, p) summary across folds ----
    summary = (
        pert.groupby(["dataset", "model", "operator", "p"])
        .agg(
            ddr_mean=("ddr", "mean"),
            auc_mean=("auc", "mean"),
            auc_drop_mean=("auc_drop", "mean"),
            auc_drop_std=("auc_drop", "std"),
            n=("auc", "size"),
        )
        .reset_index()
        .sort_values(["dataset", "model", "operator", "p"])
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    # ---- correlations per (dataset, model) (pooled over operators/p/folds) ----
    corr_lines = []
    for (ds, mdl), part in pert.groupby(["dataset", "model"]):
        x = part["ddr"].to_numpy(dtype=float)
        y = part["auc_drop"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3 and np.ptp(x[ok]) > 0:
            r, pr = stats.pearsonr(x[ok], y[ok])
            rho, prho = stats.spearmanr(x[ok], y[ok])
            corr_lines.append((f"{ds}/{mdl}", r, pr, rho, prho, int(ok.sum())))

    # ---- figure: DDR vs AUC drop ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    op_order = [a for a in _OP_STYLE if a in set(pert["operator"])]
    op_order += sorted(set(pert["operator"]) - set(op_order))
    for op in op_order:
        g = pert.loc[pert["operator"] == op]
        style = _OP_STYLE.get(op, {"color": "#555555", "marker": "o"})
        ax.scatter(
            g["ddr"],
            g["auc_drop"],
            s=42,
            alpha=0.85,
            label=op,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.4,
        )
    xall = pert["ddr"].to_numpy(dtype=float)
    yall = pert["auc_drop"].to_numpy(dtype=float)
    ok = np.isfinite(xall) & np.isfinite(yall)
    if ok.sum() >= 2 and np.ptp(xall[ok]) > 0:
        b, a = np.polyfit(xall[ok], yall[ok], 1)
        xs = np.linspace(xall[ok].min(), xall[ok].max(), 50)
        ax.plot(xs, b * xs + a, "k--", lw=1.2, alpha=0.7, label="pooled fit")
        r_all, p_all = stats.pearsonr(xall[ok], yall[ok])
        ax.annotate(f"pooled Pearson $r$={r_all:.2f} ($p$={p_all:.1e})", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=9, va="top")
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("DAG Disruption Rate (DDR)")
    ax.set_ylabel("Downstream AUC drop vs. baseline graph")
    ax.set_title("DDR vs. downstream graph-KT accuracy change", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, bbox_inches="tight")
    fig.savefig(args.out_fig.with_suffix(".png"), dpi=150, bbox_inches="tight")

    # ---- LaTeX summary table ----
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['dataset']} & \\texttt{{{str(r['model']).replace('_', chr(92)+'_')}}} & "
            f"\\texttt{{{r['operator'].replace('_', chr(92)+'_')}}} & {r['p']:.2f} & "
            f"{r['ddr_mean']:.3f} & {r['auc_mean']:.4f} & {r['auc_drop_mean']:.4f} \\\\"
        )
    body = "\n".join(rows)
    corr_str = "; ".join(
        f"{ds}: Pearson $r$={r:.2f} ($n$={n}), Spearman $\\rho$={rho:.2f}"
        for ds, r, pr, rho, _, n in corr_lines
    )
    xg = pert["ddr"].to_numpy(dtype=float)
    yg = pert["auc_drop"].to_numpy(dtype=float)
    n_global = int((np.isfinite(xg) & np.isfinite(yg)).sum())
    tex = (
        "% Auto-generated by scripts/plot_ddr_downstream.py\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{DDR vs.\\ downstream graph-KT accuracy. For each dataset and model, the prerequisite graph "
        "$\\Epre$ is perturbed by each operator at strength $p$; \\emph{AUC drop} is the mean "
        "decrease in test AUC relative to the unperturbed (DDR$=0$) baseline graph across folds. "
        "Per-(dataset, model) correlations are distinct estimands"
        + (": " + corr_str if corr_str else ".")
        + f" The global pooled Pearson across both datasets $\\times$ both models is annotated on "
        f"Fig.~S3 ($n$={n_global}) and must not be conflated with XES3G5M/GKT-only figures.}}\n"
        "\\label{tab:ddr-downstream}\n"
        "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabularx}{\\linewidth}{@{} >{\\RaggedRight\\arraybackslash}p{0.13\\linewidth} "
        ">{\\RaggedRight\\arraybackslash}p{0.10\\linewidth} >{\\RaggedRight\\arraybackslash}p{0.16\\linewidth} "
        "c *{3}{>{\\centering\\arraybackslash}X} @{}}\n\\toprule\n"
        "Dataset & Model & Operator & $p$ & Mean DDR & Mean AUC & AUC drop \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabularx}}\n\\end{{table}}\n"
    )
    args.out_tex.write_text(tex, encoding="utf-8")

    print(summary.to_string(index=False))
    print("\nCorrelations (DDR vs AUC drop):")
    for ds, r, pr, rho, prho, n in corr_lines:
        print(f"  {ds}: Pearson r={r:.3f} (p={pr:.2e}), Spearman rho={rho:.3f} (p={prho:.2e}), n={n}")
    print(f"\nWrote {args.out_summary}\n      {args.out_tex}\n      {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
