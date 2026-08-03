"""Generate Phase-C statistics tables: mean±std summaries and inferential ANOVA.

Reads released CSV artefacts under results/tables/ and writes LaTeX fragments
consumed by paper/main_APIN.tex and paper/supplementary.tex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"

DATASET_LABELS = {
    "assist2012": "ASSISTments 2012",
    "junyi": "Junyi Academy",
    "xes3g5m": "XES3G5M",
    "synthetic_c2": "Synthetic C2",
    "synthetic_c5": "Synthetic C5",
}

DATASET_LABELS_TEX = {
    "assist2012": r"ASSIST'12",
    "junyi": r"\makecell[l]{Junyi\\Academy}",
    "xes3g5m": "XES3G5M",
    "synthetic_c2": r"\makecell{Synth\\C2}",
    "synthetic_c5": r"\makecell{Synth\\C5}",
}

MODEL_ORDER = ["bkt", "akt", "dkt", "simplekt", "gkt", "gikt", "skt", "dygkt", "dgekt"]
MODEL_DISPLAY = {
    "simplekt": r"\textit{simpleKT}",
    "gkt": "GKT",
    "gikt": "GIKT",
    "skt": "SKT",
    "dygkt": "DyGKT",
    "dgekt": "DGEKT",
    "akt": "AKT",
    "dkt": "DKT",
    "bkt": "BKT",
}

DDR_FAMILY_ORDER = [
    "attr_mask",
    "subgraph",
    "prereq_preserve",
    "edge_drop",
    "node_drop",
]

STRATA_ORDER = ["very_cold", "cold", "warm", "hot"]
NO_GRAPH = {"dkt", "simplekt"}
GRAPH = {"gkt", "gikt", "dgekt"}


def _fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    if np.isnan(mean):
        return r"$--.--- \pm --.---$"
    if np.isnan(std):
        std = 0.0
    return rf"${mean:.{decimals}f} \pm {std:.{decimals}f}$"


def _fmt_delta_pm(mean: float, std: float) -> str:
    if np.isnan(mean):
        return r"$--.--- \pm --.---$"
    if np.isnan(std):
        std = 0.0
    sign = "+" if mean >= 0 else ""
    return rf"${sign}{mean:.3f} \pm {std:.3f}$"


def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "---"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def _stratum_tex(name: str) -> str:
    return str(name).replace("_", r"\_")


def _eta_partial(ss_effect: float, ss_error: float) -> float:
    denom = ss_effect + ss_error
    return float(ss_effect / denom) if denom > 0 else float("nan")


def _legacy_eoc_to_rho_abs(value: float) -> float:
    """Convert legacy sqrt(2+2rho^2) exports to |rho|; pass through if already |rho|."""
    if value is None or np.isnan(value):
        return float("nan")
    if value <= 1.0:
        return float(value)
    return float(np.sqrt(max(0.0, (float(value) ** 2 - 2.0) / 2.0)))


def write_leakage_metrics_tex(df: pd.DataFrame, path: Path) -> None:
    share_path = path.parent / "edge_share_summary.csv"
    share_by_ds: dict[str, pd.DataFrame] = {}
    if share_path.exists():
        share_df = pd.read_csv(share_path)
        for dataset, part in share_df.groupby("dataset"):
            share_by_ds[str(dataset)] = part

    rows = []
    for dataset, part in df.groupby("dataset"):
        label = DATASET_LABELS.get(dataset, dataset)
        rho_series = part["eoc"].map(_legacy_eoc_to_rho_abs)
        share_part = share_by_ds.get(str(dataset))
        if share_part is not None and not share_part.empty:
            gt50 = _fmt_pm(
                share_part["frac_share_gt_50"].mean(),
                share_part["frac_share_gt_50"].std(ddof=0),
                3,
            )
            p90 = _fmt_pm(
                share_part["share_p90"].mean(),
                share_part["share_p90"].std(ddof=0),
                3,
            )
        else:
            gt50 = p90 = "---"
        rows.append(
            (
                label,
                _fmt_pm(part["ecr_flag"].mean(), part["ecr_flag"].std(ddof=0), 3),
                _fmt_pm(part["ecr_overlap"].mean(), part["ecr_overlap"].std(ddof=0), 3),
                _fmt_pm(rho_series.mean(), rho_series.std(ddof=0), 3),
                _fmt_pm(part["tbvr"].mean(), part["tbvr"].std(ddof=0), 3),
                gt50,
                p90,
            )
        )

    share_note = (
        r" Edge share $>50\%$: fraction of retained edges whose held-out transition "
        r"share exceeds $0.5$; share p90: 90th percentile of per-edge held-out shares "
        r"(Supplementary artefact \texttt{edge\_share\_summary.csv})."
        if share_by_ds
        else ""
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direct leakage diagnostics per dataset (mean~$\pm$~std over three folds). "
        r"\textsc{ECR}\textsubscript{flag}: learner-overlap indicator (Eq.~\ref{eq:ecr-flag}). "
        r"\textsc{ECR}\textsubscript{overlap}: held-out pattern overlap (Eq.~\ref{eq:ecr-overlap}). "
        r"$|\rho|$: edge--outcome Pearson correlation magnitude (Eq.~\ref{eq:rho-edge-outcome}). "
        r"\textsc{TBMR}: within-train temporal mixing "
        r"(Eq.~\ref{eq:tbvr}), not a train/test violation."
        + share_note
        + r"}",
        r"\label{tab:leakage-metrics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}p{0.17\linewidth} *{6}{>{\centering\arraybackslash}X} @{}}",
        r"\toprule",
        r"Dataset & \textsc{ECR}\textsubscript{flag} & \textsc{ECR}\textsubscript{overlap} & $|\rho|$ & \textsc{TBMR} & $>50\%$ & p90 \\",
        r"\midrule",
    ]
    for label, ecr_f, ecr_o, rho, tbvr, gt50, p90 in rows:
        lines.append(f"{label} & {ecr_f} & {ecr_o} & {rho} & {tbvr} & {gt50} & {p90} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_baseline_summary_tex(fold_df: pd.DataFrame, path: Path) -> None:
    df = fold_df[fold_df["graph_construction"].astype(str).str.strip() == "train_only"].copy()
    summary = df.groupby(["dataset", "model"])["auc"].agg(["mean", "std"]).reset_index()
    summary.rename(columns={"mean": "auc_mean", "std": "auc_std"}, inplace=True)
    summary["auc_std"] = summary["auc_std"].fillna(0.0)

    cols = ["xes3g5m", "assist2012", "junyi", "synthetic_c2", "synthetic_c5"]
    lines = [
        r"% Main-manuscript summary: cross-dataset AUC (three-fold means $\pm$ std).",
        r"% Auto-generated by scripts/generate_phase_c_tables.py",
        r"",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Diagnostic baseline AUC summary (train-only graphs; three-fold learner CV "
        r"means~$\pm$~std). XES3G5M is the primary model-comparison column; Junyi is a "
        r"saturated sanity corpus. Per-dataset ACC/NLL and 95\% bootstrap confidence "
        r"intervals are in Supplementary Tables~S1--S5; fold-level detail in "
        r"Table~\ref{tab:baseline-cv}.}",
        r"\label{tab:baseline-summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}p{0.20\linewidth} *{5}{>{\centering\arraybackslash}X} @{}}",
        r"\toprule",
        r"Model & \makecell{XES\\3G5M} & \makecell{ASSIST\\2012} & \makecell{Junyi} & \makecell{Synth\\C2} & \makecell{Synth\\C5} \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        cells = [MODEL_DISPLAY.get(model, model)]
        for ds in cols:
            row = summary[(summary["dataset"] == ds) & (summary["model"] == model)]
            if row.empty:
                cells.append("---")
            else:
                r0 = row.iloc[0]
                cells.append(_fmt_pm(r0["auc_mean"], r0["auc_std"]))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ddr_raw_tex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"% Auto-generated by scripts/generate_phase_c_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean~$\pm$~std \DDR{} across three seeds and three train folds for the five "
        r"augmentation families and four perturbation strengths ($9$ runs per cell). "
        r"Subgraph sampling is random-walk based (Section~\ref{sec:ddr}); "
        r"\texttt{prereq\_preserve} removes only transitively redundant edges at the same "
        r"per-edge budget as \texttt{edge\_drop}. Lower is less disruptive.}",
        r"\label{tab:ddr-raw}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}p{0.18\linewidth} >{\ttfamily\footnotesize\raggedright\arraybackslash}p{0.20\linewidth} *{4}{>{\centering\arraybackslash}X} @{}}",
        r"\toprule",
        r"Dataset & Family & \makecell{$p{=}0.05$} & \makecell{$p{=}0.10$} & \makecell{$p{=}0.20$} & \makecell{$p{=}0.30$} \\",
        r"\midrule",
    ]

    for ds_idx, dataset in enumerate(["junyi", "assist2012", "xes3g5m"]):
        ds_label = DATASET_LABELS_TEX[dataset]
        sub = summary[summary["dataset"] == dataset]
        n_fam = len(DDR_FAMILY_ORDER)
        for f_idx, family in enumerate(DDR_FAMILY_ORDER):
            fam_rows = sub[sub["augmentation"] == family].sort_values("p")
            ds_cell = rf"\multirow{{{n_fam}}}{{*}}{{{ds_label}}}" if f_idx == 0 else ""
            cells = [ds_cell, family.replace("_", r"\_")]
            for p in [0.05, 0.1, 0.2, 0.3]:
                row = fam_rows[np.isclose(fam_rows["p"], p)]
                if row.empty:
                    cells.append("---")
                else:
                    r0 = row.iloc[0]
                    cells.append(_fmt_pm(r0["ddr_mean"], r0["ddr_std"]))
            lines.append(" & ".join(cells) + r" \\")
        if ds_idx < 2:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _cold_start_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, stratum, fold), part in df.groupby(["dataset", "stratum", "fold"]):
        if stratum not in STRATA_ORDER:
            continue
        no_graph = part[part["model"].isin(NO_GRAPH)]["auc"].max()
        graph = part[part["model"].isin(GRAPH)]["auc"].max()
        simple = part[part["model"] == "simplekt"]["auc"]
        rows.append(
            {
                "dataset": dataset,
                "stratum": stratum,
                "fold": fold,
                "n": int(part[part["model"] == "simplekt"]["n"].iloc[0]) if not simple.empty else np.nan,
                "simplekt_auc": float(simple.iloc[0]) if not simple.empty else np.nan,
                "delta_max": float(graph - no_graph) if pd.notna(graph) and pd.notna(no_graph) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _suppress_summary_stratum(raw: pd.DataFrame, dataset: str, stratum: str) -> bool:
    """Suppress unreliable stratum means (tiny n and/or identical AUC across models)."""
    sub = raw[(raw["dataset"] == dataset) & (raw["stratum"] == stratum)]
    if sub.empty:
        return True
    compare_models = ["dkt", "simplekt", "gkt", "gikt", "dgekt"]
    fold0 = sub[sub["fold"] == 0]
    if not fold0.empty:
        n0 = int(fold0["n"].iloc[0])
        vals = fold0[fold0["model"].isin(compare_models)]["auc"].dropna()
        if n0 < 50 and len(vals) >= 3:
            if len(set(round(float(v), 6) for v in vals)) == 1:
                return True
    if dataset == "junyi" and stratum in ("very_cold", "cold"):
        return True
    return False


def write_cold_start_summary_tex(df: pd.DataFrame, path: Path) -> None:
    delta_df = _cold_start_deltas(df)
    agg = (
        delta_df.groupby(["dataset", "stratum"])
        .agg(
            n=("n", "first"),
            simple_mean=("simplekt_auc", "mean"),
            simple_std=("simplekt_auc", "std"),
            delta_mean=("delta_max", "mean"),
            delta_std=("delta_max", "std"),
        )
        .reset_index()
    )
    agg[["simple_std", "delta_std"]] = agg[["simple_std", "delta_std"]].fillna(0.0)

    lines = [
        r"% Auto-generated by scripts/generate_phase_c_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cold-start KC diagnostic summary. Column $n$ counts fold~0 "
        r"validation+test held-out interactions per KC-frequency stratum (strata "
        r"assigned from train-fold counts only; \textit{simpleKT}); \textit{simpleKT} AUC and "
        r"$\Delta_{\max}$ report three-fold means~$\pm$~std ($\Delta_{\max}$: best "
        r"graph-family AUC minus best sequence-only AUC). Cells marked ``---'' are "
        r"suppressed when fold~0 stratum counts are too small or stratum AUC is "
        r"undefined/unreliable (see Supplementary Tables~S12--S13). Per-model strata appear in "
        r"Supplementary Tables~S12--S13.}",
        r"\label{tab:cold-start-summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}p{0.15\linewidth} >{\RaggedRight\arraybackslash}p{0.12\linewidth} >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X @{}}",
        r"\toprule",
        r"Dataset & Stratum & $n$ & \textit{simpleKT} AUC & $\Delta_{\max}$ \\",
        r"\midrule",
    ]

    hot_n = {
        "junyi": r"$\approx 7.7 \times 10^{6}$",
        "assist2012": r"$\approx 8.1 \times 10^{5}$",
        "xes3g5m": r"$\approx 1.9 \times 10^{6}$",
    }

    for ds_idx, dataset in enumerate(["junyi", "assist2012", "xes3g5m", "synthetic_c2", "synthetic_c5"]):
        ds_label = DATASET_LABELS_TEX[dataset]
        if dataset in {"synthetic_c2", "synthetic_c5"}:
            row = agg[(agg["dataset"] == dataset) & (agg["stratum"] == "hot")]
            if row.empty:
                continue
            r0 = row.iloc[0]
            lines.append(
                f"{ds_label} & hot & 60{{,}}000 & "
                f"{_fmt_pm(r0['simple_mean'], r0['simple_std'])} & "
                f"{_fmt_delta_pm(r0['delta_mean'], r0['delta_std'])} \\\\"
            )
            if dataset != "synthetic_c5":
                lines.append(r"\midrule")
            continue

        sub = agg[agg["dataset"] == dataset]
        n_rows = len(STRATA_ORDER)
        for s_idx, stratum in enumerate(STRATA_ORDER):
            row = sub[sub["stratum"] == stratum]
            if row.empty:
                continue
            r0 = row.iloc[0]
            ds_cell = rf"\multirow{{{n_rows}}}{{*}}{{{ds_label}}}" if s_idx == 0 else ""
            n_cell = hot_n.get(dataset, str(int(r0["n"]))) if stratum == "hot" else f"{int(r0['n']):,}".replace(",", "{,}")
            delta_s = _fmt_delta_pm(r0["delta_mean"], r0["delta_std"])
            suppress = _suppress_summary_stratum(df, dataset, stratum)
            if pd.isna(r0["simple_mean"]) or pd.isna(r0["delta_mean"]):
                suppress = True
            auc_cell = r"---" if suppress else _fmt_pm(r0["simple_mean"], r0["simple_std"])
            if suppress:
                delta_s = r"---"
            lines.append(
                f"{ds_cell} & {_stratum_tex(stratum)} & {n_cell} & "
                f"{auc_cell} & {delta_s} \\\\"
            )
        if dataset != "synthetic_c5":
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _anova_row(name: str, ss: float, df: int, ms: float, f_val: float, p: float, eta: float) -> str:
    return (
        f"{name} & {ss:.4f} & {df} & {ms:.4f} & {_fmt_p(p)} & {eta:.3f} \\\\"
        if not np.isnan(f_val)
        else f"{name} & {ss:.4f} & {df} & {ms:.4f} & --- & {eta:.3f} \\\\"
    )


def write_baseline_anova_tex(fold_df: pd.DataFrame, path: Path) -> None:
    public = ["xes3g5m", "assist2012", "junyi"]
    models = ["simplekt", "gkt", "gikt", "skt", "dygkt", "dgekt"]
    df = fold_df[
        (fold_df["dataset"].isin(public))
        & (fold_df["model"].isin(models))
        & (fold_df["graph_construction"].astype(str).str.strip() == "train_only")
    ].copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=public, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)

    fit = ols("auc ~ C(dataset) + C(model) + C(dataset):C(model)", data=df).fit()
    table = anova_lm(fit, typ=2)
    resid_ss = float(table.loc["Residual", "sum_sq"])
    resid_df = int(table.loc["Residual", "df"])

    lines = [
        r"% Auto-generated by scripts/generate_phase_c_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Exploratory two-way ANOVA on fold-level AUC ($n{=}3$ folds per "
        r"dataset--model cell; public benchmarks only, train-only graphs; Table~S19). "
        r"Partial $\eta^2 = \mathrm{SS}_{\mathrm{effect}}/(\mathrm{SS}_{\mathrm{effect}}+\mathrm{SS}_{\mathrm{error}})$. "
        r"With only three folds per cell, $p$-values are underpowered; fold-level "
        r"$\Delta$AUC magnitudes and Table~S16 intervals are the primary evidence.}",
        r"\label{tab:anova-baseline}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}p{0.28\linewidth} r r r >{\centering\arraybackslash}X >{\centering\arraybackslash}X @{}}",
        r"\toprule",
        r"Source & SS & df & MS & $p$ & $\eta^2_p$ \\",
        r"\midrule",
    ]

    rename = {
        "C(dataset)": "Dataset",
        "C(model)": "Backbone",
        "C(dataset):C(model)": r"Dataset $\times$ Backbone",
        "Residual": "Residual",
    }
    for idx, row in table.iterrows():
        ss = float(row["sum_sq"])
        df_i = int(row["df"])
        ms = ss / df_i if df_i > 0 else float("nan")
        f_val = float(row["F"]) if "F" in row.index and pd.notna(row["F"]) else float("nan")
        p = float(row["PR(>F)"]) if "PR(>F)" in row.index and pd.notna(row["PR(>F)"]) else float("nan")
        eta = _eta_partial(ss, resid_ss) if idx != "Residual" else float("nan")
        label = rename.get(str(idx), str(idx))
        if idx == "Residual":
            lines.append(f"Residual & {ss:.4f} & {df_i} & {ms:.4f} & --- & --- \\\\")
        else:
            lines.append(_anova_row(label, ss, df_i, ms, f_val, p, eta))

    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ddr_downstream_anova_tex(ddr_df: pd.DataFrame, path: Path) -> None:
    # Prefer GKT multi-seed rows when present (revision powered ANOVA); else DGEKT.
    has_gkt = "model" in ddr_df.columns and (ddr_df["model"] == "gkt").any()
    model = "gkt" if has_gkt else "dgekt"
    df = ddr_df[
        (ddr_df["operator"] != "none")
        & (ddr_df["dataset"].isin(["assist2012", "xes3g5m"]))
    ].copy()
    if "model" in df.columns:
        df = df[df["model"] == model]
    # Core grid only (exclude p=0.90 manipulation-check anchors)
    if "p" in df.columns:
        df = df[df["p"].astype(float) <= 0.3 + 1e-9]
    df["operator"] = pd.Categorical(
        df["operator"], categories=["edge_drop", "node_drop", "prereq_preserve"], ordered=True
    )

    fit = ols("auc ~ C(dataset) + C(operator) + C(operator):C(p) + C(dataset):C(operator)", data=df).fit()
    table = anova_lm(fit, typ=2)
    resid_ss = float(table.loc["Residual", "sum_sq"])

    backbone = "GKT" if model == "gkt" else "DGEKT"
    n_note = (
        "GKT multi-seed sweep on XES3G5M (seeds 42/17/1234) plus ASSISTments seed-42 folds; "
        "core $p\\le 0.3$ grid (anchors $p{=}0.90$ excluded)"
        if model == "gkt"
        else "ASSISTments and XES3G5M; three folds per operator/strength"
    )
    lines = [
        r"% Auto-generated by scripts/generate_phase_c_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{ANOVA on {backbone} downstream AUC under graph perturbation "
        rf"({n_note}; Table~S20). Dependent variable: "
        r"test AUC after retraining on the perturbed graph. Partial $\eta^2$ as above. "
        r"This analysis supports the DDR$\to$downstream findings in "
        r"Section~\ref{sec:exp-ddr-downstream}.}",
        r"\label{tab:anova-ddr-downstream}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{@{} >{\RaggedRight\arraybackslash}p{0.34\linewidth} r r r >{\centering\arraybackslash}X >{\centering\arraybackslash}X @{}}",
        r"\toprule",
        r"Source & SS & df & MS & $p$ & $\eta^2_p$ \\",
        r"\midrule",
    ]

    rename = {
        "C(dataset)": "Dataset",
        "C(operator)": "Operator",
        "C(operator):C(p)": r"Operator $\times$ strength $p$",
        "C(dataset):C(operator)": r"Dataset $\times$ Operator",
        "Residual": "Residual",
    }
    for idx, row in table.iterrows():
        ss = float(row["sum_sq"])
        df_i = int(row["df"])
        ms = ss / df_i if df_i > 0 else float("nan")
        f_val = float(row["F"]) if "F" in row.index and pd.notna(row["F"]) else float("nan")
        p = float(row["PR(>F)"]) if "PR(>F)" in row.index and pd.notna(row["PR(>F)"]) else float("nan")
        eta = _eta_partial(ss, resid_ss) if idx != "Residual" else float("nan")
        label = rename.get(str(idx), str(idx))
        if idx == "Residual":
            lines.append(f"Residual & {ss:.4f} & {df_i} & {ms:.4f} & --- & --- \\\\")
        else:
            lines.append(_anova_row(label, ss, df_i, ms, f_val, p, eta))

    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    leakage = pd.read_csv(OUT / "leakage_metrics.csv")
    write_leakage_metrics_tex(leakage, OUT / "leakage_metrics.tex")

    fold_df = pd.read_csv(OUT / "baseline_fold_results.csv")
    write_baseline_summary_tex(fold_df, OUT / "baseline_results.tex")

    ddr_summary = pd.read_csv(OUT / "dag_disruption_summary.csv")
    write_ddr_raw_tex(ddr_summary, OUT / "ddr_raw.tex")

    cold_df = pd.read_csv(OUT / "cold_start_metrics.csv")
    write_cold_start_summary_tex(cold_df, OUT / "cold_start_summary.tex")

    write_baseline_anova_tex(fold_df, OUT / "anova_baseline.tex")

    ddr_down = pd.read_csv(OUT / "ddr_downstream.csv")
    write_ddr_downstream_anova_tex(ddr_down, OUT / "anova_ddr_downstream.tex")

    # Refresh CV table and significance CSV via existing helper.
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import significance_testing as sig

    train_only = sig._load_train_only(fold_df)
    sig_df = sig.run_significance_tests(train_only)
    if not sig_df.empty:
        sig_df.to_csv(OUT / "significance_tests.csv", index=False)
    sig.write_cv_tex(train_only, sig_df if not sig_df.empty else None, OUT / "baseline_cv_template.tex")

    import generate_cold_start_comparison as csc

    csc.main()

    import bootstrap_auc_ci as boot

    boot.synthesize_table()

    import generate_training_parity as tp

    tp.main()

    print("Phase-C tables written to", OUT)


if __name__ == "__main__":
    main()
