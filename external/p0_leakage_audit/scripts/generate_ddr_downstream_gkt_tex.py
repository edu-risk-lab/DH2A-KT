#!/usr/bin/env python3
"""Regenerate results/tables/ddr_downstream_gkt.tex from the merged paper CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DS_LABEL = {"xes3g5m": "XES3G5M", "assist2012": "ASSIST2012"}
OP_ORDER = {"edge_drop": 0, "node_drop": 1, "prereq_preserve": 2}
DS_ORDER = {"xes3g5m": 0, "assist2012": 1}


def _corr_stats(
    gkt: pd.DataFrame, dataset: str, p_max: float | None = None
) -> tuple[float, float, int]:
    part = gkt[gkt["dataset"] == dataset]
    if p_max is not None:
        part = part[part["p"] <= p_max + 1e-9]
    x = part["ddr"].to_numpy(dtype=float)
    y = part["auc_drop"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r = float(stats.pearsonr(x[ok], y[ok])[0])
    rho = float(stats.spearmanr(x[ok], y[ok])[0])
    return r, rho, n


def _row(r: pd.Series) -> str:
    op = str(r.operator).replace("_", r"\_")
    return (
        f"{DS_LABEL.get(str(r.dataset), r.dataset)} & \\texttt{{{op}}} & "
        f"{float(r.p):.2f} & {float(r.ddr_mean):.3f} & "
        f"{float(r.auc_mean):.4f} & {float(r.auc_drop_mean):.4f} \\\\"
    )


def main() -> int:
    df = pd.read_csv(ROOT / "results/tables/ddr_downstream.csv")
    gkt = df[(df["model"] == "gkt") & (df["operator"] != "none")].copy()
    gkt = gkt[pd.notna(gkt["auc_drop"])]
    gkt["p"] = gkt["p"].astype(float).round(4)

    summary = (
        gkt.groupby(["dataset", "operator", "p"])
        .agg(
            ddr_mean=("ddr", "mean"),
            auc_mean=("auc", "mean"),
            auc_drop_mean=("auc_drop", "mean"),
            n=("auc", "size"),
        )
        .reset_index()
    )
    summary["_ds"] = summary["dataset"].map(DS_ORDER)
    summary["_op"] = summary["operator"].map(OP_ORDER)
    summary = summary.sort_values(["_ds", "_op", "p"])

    base = df[(df["model"] == "gkt") & (df["operator"] == "none")]
    xes_base = float(base.loc[base["dataset"] == "xes3g5m", "auc"].mean())
    ast_base = float(base.loc[base["dataset"] == "assist2012", "auc"].mean())
    r_core, _, n_core = _corr_stats(gkt, "xes3g5m", 0.3)
    r_all, rho_all, n_all = _corr_stats(gkt, "xes3g5m", None)

    core = "\n".join(_row(r) for _, r in summary[summary["p"] <= 0.3 + 1e-9].iterrows())
    anchors = "\n".join(_row(r) for _, r in summary[summary["p"] >= 0.9 - 1e-9].iterrows())
    n_xes = int(summary.loc[(summary["dataset"] == "xes3g5m") & (summary["p"] == 0.1), "n"].max())

    tex = f"""% Auto-derived from results/tables/ddr_downstream.csv (GKT multi-seed merge)
% XES3G5M: seeds 42/17/1234 x 3 folds (n={n_xes}); ASSIST2012: seed-42 grid (+ anchors).
% AUC drop = mean decrease in test AUC vs. the unperturbed (DDR=0) baseline.
% Caption scopes: r_core n={n_core}; r_all/rho_all n={n_all}; global pooled r~0.75 is Fig S3 only.
\\begin{{table}}[t]
\\centering
\\caption{{\\DDR{{}}$\\to$downstream AUC for the graph-reliant backbone GKT, with a
positive control. For each dataset the prerequisite graph $\\Epre$ is perturbed
by each operator at strength $p$; \\emph{{AUC drop}} is the mean decrease in test
AUC relative to the unperturbed (\\DDR{{}}$=0$) baseline across folds (baselines:
XES3G5M ${xes_base:.4f}$, ASSISTments ${ast_base:.4f}$; XES3G5M pooled over
3 seeds $\\times$ 3 folds). The bottom block reports the
manipulation-check anchors ($p{{=}}0.90$, near-total graph destruction). On
XES3G5M/GKT, \\DDR{{}} tracks AUC drop with distinct estimands: Pearson
$r{{=}}{r_core:.2f}$ on the $p{{\\le}}0.3$ core ($n{{=}}{n_core}$); Pearson
$r{{=}}{r_all:.2f}$ on the full pool including anchors ($n{{=}}{n_all}$); Spearman
$\\rho{{=}}{rho_all:.2f}$ on that same full pool ($n{{=}}{n_all}$). These are not
interchangeable with the global pooled Pearson $r{{\\approx}}0.75$ ($n{{=}}186$)
annotated on Fig.~S3. At matched budget,
\\texttt{{prereq\\_preserve}} degrades AUC least and \\texttt{{node\\_drop}} most. On
ASSISTments the same total destruction moves GKT by ${{\\le}}0.003$, so this cell
(like DGEKT in Table~\\ref{{tab:ddr-downstream}}) is a \\emph{{low-reliance anchor}}
whose correlation is not practically meaningful.}}
\\label{{tab:ddr-downstream-gkt}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabularx}}{{\\linewidth}}{{@{{}} >{{\\RaggedRight\\arraybackslash}}p{{0.13\\linewidth}} >{{\\RaggedRight\\arraybackslash}}p{{0.20\\linewidth}} c *{{3}}{{>{{\\centering\\arraybackslash}}X}} @{{}}}}
\\toprule
Dataset & Operator & $p$ & Mean \\DDR{{}} & Mean AUC & AUC drop \\\\
\\midrule
{core}
\\midrule
\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{Manipulation-check anchors (near-total destruction)}}}}\\\\
{anchors}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
"""
    out = ROOT / "results/tables/ddr_downstream_gkt.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"XES baseline={xes_base:.4f} ASSIST baseline={ast_base:.4f}")
    print(f"Pearson XES core p<=0.3 r={r_core:.3f} n={n_core}; all r={r_all:.3f} rho={rho_all:.3f} n={n_all}")
    print(summary[summary["dataset"] == "xes3g5m"][["operator", "p", "ddr_mean", "auc_mean", "auc_drop_mean", "n"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
