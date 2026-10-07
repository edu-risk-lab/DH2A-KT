"""Credit outcomes for the transfer ladders, the second encoder, and the padded arm.

Rows whose five seed CSVs are not all present are listed as pending and
left out of the table.

Reads the per-seed comparison CSVs that ``scripts/50_credit_ladder_plan.py``
jobs write, pairs on/control arms by seed, and applies two rules:

  seed-wise  every paired delta_s >= tau, tau = +0.002
  interval   mean(delta) - t_{0.975,4} * SD(delta) / sqrt(5) >= tau

SD(delta) is the sample SD of the paired per-seed deltas, so the interval
rule scales with the noise of the contrast itself and has no free
constant. It replaces an earlier max(0.002, K * SD_control) cut whose K
had been chosen to keep the XES3G5M map. The XES rows are recomputed here
from the frozen CSVs.

Writes results/tables/credit_ladder_transfer.json and
paper/tables/table_credit_transfer.tex. Does not retrain anything.

    python scripts/51_transfer_credit_tables.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
SEEDS = (0, 17, 42, 1234, 2024)
TAU = 0.002
# Two-sided 95% Student t quantile, df = len(SEEDS) - 1 = 4.
T_975 = 2.7764451051977987
OUT_JSON = TABLES / "credit_ladder_transfer.json"
OUT_TEX = REPO / "paper" / "tables" / "table_credit_transfer.tex"


def _ladder(arm: str) -> str:
    return f"dh2_kt_ladder_{arm}_s{{seed}}_vs_p0.csv"


def _frozen(prefix: str) -> str:
    return f"{prefix}_s{{seed}}.csv"


ASSIST_ARMS = (
    ("Q$\\leftarrow$KC incidence", "Capacity", "incidence-on", "incidence-off"),
    ("Package, observed incidence", "Package", "time-on-observed", "incidence-on"),
    ("Package, zero incidence", "Package", "time-on-zero", "incidence-off"),
    ("Aligned gap vs.\\ T-zero", "Architecture", "time-on-zero", "t-zero"),
    ("Aligned gap vs.\\ T-misaligned", "Architecture", "time-on-zero", "t-misaligned"),
)
ASSIST_1000S = "ASSIST2012 1{,}000\\,s"
ASSIST_GROUPS = (
    ("assist-sec", "ASSIST2012"),
    ("assist-sec-ref", "ASSIST2012 ref."),
    ("assist", ASSIST_1000S),
    ("assist-ref", ASSIST_1000S + " ref."),
)
OUT_TEX_RES = REPO / "paper" / "tables" / "table_credit_assist_resolution.tex"
OUT_HEAT = REPO / "paper" / "figures" / "fig_credit_heatmap.tex"

# (corpus, label, rung, on pattern, control pattern)
CONTRASTS = [
    ("XES3G5M", "Q$\\leftarrow$KC incidence", "Capacity",
     _frozen("qkc_cm_observed"), _frozen("qkc_cm_zero")),
    ("XES3G5M", "Package, observed incidence", "Package", None, None),
    ("XES3G5M", "Package, zero incidence", "Package",
     _frozen("qkc_cm_zero_time"), _frozen("qkc_cm_zero")),
    ("XES3G5M", "Aligned gap vs.\\ T-zero", "Architecture",
     _frozen("qkc_cm_zero_time"), _frozen("a1_t_zero")),
    ("XES3G5M", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _frozen("qkc_cm_zero_time"), _frozen("a1_t_misaligned")),
    ("XES3G5M", "Package vs.\\ param-padded no-time", "Package (param.\\ matched)",
     _frozen("qkc_cm_zero_time"), _ladder("package-matched_no-time-param-padded")),
    ("XES3G5M", "Aligned gap vs.\\ T-shuffled", "Architecture",
     _frozen("qkc_cm_zero_time"), _ladder("xes-gap-controls_t-shuffled")),
    ("XES3G5M", "Aligned gap vs.\\ T-boundary", "Architecture",
     _frozen("qkc_cm_zero_time"), _ladder("xes-gap-controls_t-boundary")),
    ("XES3G5M", "T-boundary vs.\\ T-zero", "Architecture",
     _ladder("xes-gap-controls_t-boundary"), _frozen("a1_t_zero")),
    ("XES3G5M fold 1", "Package, zero incidence", "Package",
     _ladder("xes-fold1_time-package"), _ladder("xes-fold1_no-time")),
    ("XES3G5M fold 1", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("xes-fold1_time-package"), _ladder("xes-fold1_t-zero")),
    ("XES3G5M fold 2", "Package, zero incidence", "Package",
     _ladder("xes-fold2_time-package"), _ladder("xes-fold2_no-time")),
    ("XES3G5M fold 2", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("xes-fold2_time-package"), _ladder("xes-fold2_t-zero")),
    ("XES3G5M attn.", "Package, zero incidence", "Package (backbone)",
     _ladder("backbone-attn_time-package"), _ladder("backbone-attn_no-time")),
    ("XES3G5M attn.", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("backbone-attn_time-package"), _ladder("backbone-attn_t-zero")),
    ("XES3G5M attn.", "Aligned gap vs.\\ T-misaligned", "Architecture",
     _ladder("backbone-attn_time-package"), _ladder("backbone-attn_t-misaligned")),
    # One-second export first (main map); the original 1,000 s export is the
    # resolution sensitivity table.
    *[
        (corpus, label, rung, _ladder(f"{group}_{on}"), _ladder(f"{group}_{ctrl}"))
        for group, corpus in ASSIST_GROUPS
        for label, rung, on, ctrl in ASSIST_ARMS
    ],
    *[
        (f"ASSIST2012 ref.\\ fold {fold}", label, rung,
         _ladder(f"assist-sec-fold{fold}_{on}"), _ladder(f"assist-sec-fold{fold}_{ctrl}"))
        for fold in (1, 2)
        for label, rung, on, ctrl in ASSIST_ARMS
        if ctrl in ("incidence-off", "t-zero") and on == "time-on-zero"
    ],
    ("Junyi (50k)", "Package, zero incidence", "Package",
     _ladder("junyi-partial_time-package"), _ladder("junyi-partial_no-time")),
    ("Junyi (50k)", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("junyi-partial_time-package"), _ladder("junyi-partial_t-zero")),
    ("Junyi (all)", "Package, zero incidence", "Package",
     _ladder("junyi-full_time-package"), _ladder("junyi-full_no-time")),
    ("Junyi (all)", "Aligned gap vs.\\ T-zero", "Architecture",
     _ladder("junyi-full_time-package"), _ladder("junyi-full_t-zero")),
]


def _available(pattern: str | None) -> bool:
    if pattern is None:
        return True
    return all((TABLES / pattern.format(seed=s)).exists() for s in SEEDS)


def _read(pattern: str) -> dict[int, dict]:
    out = {}
    for seed in SEEDS:
        path = TABLES / pattern.format(seed=seed)
        with path.open(encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        out[seed] = {
            "val_auc": float(row["val_auc"]),
            "auc": float(row["dh2_kt_auc"]),
            "n": int(row["n_predictions"]),
            "batch": int(row["batch_size"]),
            "epochs": int(row["epochs"]),
            "file": path.name,
        }
    return out


def _outcome(credited: bool, deltas: list[float]) -> str:
    if credited:
        return "Credited"
    if all(d > 0 for d in deltas):
        return "Consistent-positive, not credited"
    return "Unsupported"


def _a7_both() -> tuple[dict[int, float], dict[int, dict]]:
    # QKC-T (observed incidence, both-mode gap) and its no-time twin, as
    # printed in table_a7_query_dt_seeds.tex.
    path = TABLES / "a7_query_dt_ablation.csv"
    on: dict[int, float] = {}
    ctrl: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["time_gap_mode"] != "both":
                continue
            seed = int(row["seed"])
            on[seed] = float(row["val_auc"])
            ctrl[seed] = {"val_auc": float(row["twin_val_auc"]), "n": 0, "batch": 16, "epochs": 30}
    return on, ctrl


_HEAT_LABEL = {
    "Q$\\leftarrow$KC incidence": "Incidence",
    "Package, observed incidence": "Pkg, obs.",
    "Package, zero incidence": "Pkg, zero",
    "Aligned gap vs.\\ T-zero": "Gap vs 0",
    "Aligned gap vs.\\ T-misaligned": "Gap vs mis.",
    "Package vs.\\ param-padded no-time": "Pkg vs pad",
    "Aligned gap vs.\\ T-shuffled": "Gap vs shuf.",
    "Aligned gap vs.\\ T-boundary": "Gap vs bnd.",
    "T-boundary vs.\\ T-zero": "Bnd vs 0",
}
_HEAT_FILL = {
    "Credited": "cbBlue!35",
    "Consistent-positive, not credited": "cbOrange!50",
    "Unsupported": "black!16",
}
_HEAT_MARK = {
    "Credited": r"$\checkmark$",
    "Consistent-positive, not credited": r"$\circ$",
    "Unsupported": r"$\times$",
}


def _write_heatmap(rows: list[dict]) -> None:
    keep = [r for r in rows if not r["corpus"].startswith(ASSIST_1000S)]
    contrasts: list[str] = []
    corpora: list[str] = []
    for row in keep:
        if row["contrast"] not in contrasts:
            contrasts.append(row["contrast"])
        if row["corpus"] not in corpora:
            corpora.append(row["corpus"])
    lookup = {(r["corpus"], r["contrast"]): r for r in keep}
    col_w, row_h = 1.38, 0.46
    label_x = 0.0
    x0 = 0.55 + col_w / 2
    lines = [
        "% Generated by scripts/51_transfer_credit_tables.py",
        "\\begin{figure}[H]",
        "\\centering",
        "\\definecolor{cbBlue}{HTML}{0072B2}",
        "\\definecolor{cbOrange}{HTML}{E69F00}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tikzpicture}[font=\\scriptsize\\sffamily, every node/.style={inner sep=1.5pt}]",
    ]
    header_y = 0.28 + len(corpora) * row_h
    for j, contrast in enumerate(contrasts):
        label = _HEAT_LABEL.get(contrast, contrast)
        x = x0 + j * col_w
        lines.append(
            f"\\node[rotate=50, anchor=south west, inner sep=0pt] at ({x-0.28},{header_y}) {{{label}}};"
        )
    for i, corpus in enumerate(corpora):
        y = (len(corpora) - 1 - i) * row_h
        lines.append(
            f"\\node[anchor=east, font=\\footnotesize\\sffamily] at ({label_x},{y}) {{{corpus}}};"
        )
        for j, contrast in enumerate(contrasts):
            row = lookup.get((corpus, contrast))
            x = x0 + j * col_w
            if row is None:
                lines.append(
                    f"\\node[minimum width={col_w-0.08}cm, "
                    f"minimum height={row_h-0.06}cm] at ({x},{y}) {{}};"
                )
                continue
            fill = _HEAT_FILL[row["outcome_abs"]]
            mark = _HEAT_MARK[row["outcome_abs"]]
            text = _milli(row["mean_delta_val"])
            lines.append(
                f"\\node[draw, fill={fill}, minimum width={col_w-0.08}cm, "
                f"minimum height={row_h-0.06}cm] at ({x},{y}) {{{mark}\\,{text}}};"
            )
    lines += [
        "\\end{tikzpicture}}",
        "\\caption{Credit outcomes; row names match Table~\\ref{tab:credit-transfer}.",
        "A check on blue is credited, a circle on orange is consistent-positive",
        "(not credited), and a cross on grey is unsupported. Column labels:",
        "Incidence; Pkg, obs.\\ and Pkg, zero, the package with observed and with",
        "zeroed incidence; Gap vs 0, mis., shuf., and bnd., the aligned gap",
        "against T-zero, T-misaligned, T-shuffled, and T-boundary; Pkg vs pad,",
        "the package against the parameter-padded control; Bnd vs 0, T-boundary",
        "against T-zero. The number is the mean paired difference in units of",
        "$10^{-3}$ validation AUC. An empty cell was not run.",
        "The $1{,}000$\\,s export is Table~\\ref{tab:credit-assist-resolution}.}",
        "\\label{fig:credit-heatmap}",
        "\\end{figure}",
        "",
    ]
    OUT_HEAT.write_text("\n".join(lines), encoding="utf-8")


def _milli(value: float) -> str:
    return f"{1000 * value:+.2f}"


def _one_outcome(row: dict, short: dict[str, str]) -> str:
    if row["outcome_abs"] == row["outcome_ci"]:
        return f"{short[row['outcome_abs']]} ${row['pass_abs']}/5$"
    return (
        f"{short[row['outcome_abs']]}/{short[row['outcome_ci']]} "
        f"${row['pass_abs']}/5$"
    )


def _write_resolution_table(rows: list[dict], short: dict[str, str]) -> None:
    by_key = {(r["corpus"], r["contrast"]): r for r in rows}
    lines = [
        "\\begin{table}[H]",
        "\\caption{ASSISTments~2012 on the original export, whose start times",
        "are stored at 1{,}000\\,s resolution. The one-second re-export of the",
        "same rows is in Table~\\ref{tab:credit-transfer}. Fold~0, five paired",
        "seeds. Differences are in units of $10^{-3}$ validation AUC; the",
        "bracket is the lower 95\\% bound. The outcome is the same under both",
        "rules. Abbreviations as in Table~\\ref{tab:credit-transfer}.}",
        "\\label{tab:credit-assist-resolution}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrl@{}}",
        "\\toprule",
        "Recipe & Contrast & Mean [lower] & Outcome \\\\",
        "\\midrule",
    ]
    for recipe, _new, old in (
        ("Corpus", "ASSIST2012", ASSIST_1000S),
        ("Reference", "ASSIST2012 ref.", ASSIST_1000S + " ref."),
    ):
        first = True
        for label, _rung, _on, _ctrl in ASSIST_ARMS:
            row = by_key.get((old, label))
            if row is None:
                continue
            lines.append(
                f"{recipe if first else ''} & {label} & "
                f"${_milli(row['mean_delta_val'])}$ [${_milli(row['lcb95_delta_val'])}$] & "
                f"{_one_outcome(row, short)} \\\\"
            )
            first = False
        if recipe == "Corpus":
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX_RES.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = []
    pending = []
    for corpus, label, rung, on_pat, ctrl_pat in CONTRASTS:
        if not (_available(on_pat) and _available(ctrl_pat)):
            pending.append(f"{corpus}: {label}")
            continue
        if on_pat is None:
            on_val, ctrl = _a7_both()
        else:
            ctrl = _read(ctrl_pat)
            on_val = {s: v["val_auc"] for s, v in _read(on_pat).items()}
        deltas = [on_val[s] - ctrl[s]["val_auc"] for s in SEEDS]
        sd_ctrl = statistics.stdev(ctrl[s]["val_auc"] for s in SEEDS)
        mean = statistics.mean(deltas)
        sd_delta = statistics.stdev(deltas)
        lcb = mean - T_975 * sd_delta / math.sqrt(len(SEEDS))
        rows.append(
            {
                "corpus": corpus,
                "contrast": label,
                "rung": rung,
                "seeds": list(SEEDS),
                "delta_val": deltas,
                "mean_delta_val": mean,
                "min_delta_val": min(deltas),
                "sd_control_val": sd_ctrl,
                "sd_delta_val": sd_delta,
                "lcb95_delta_val": lcb,
                "tau_abs": TAU,
                "pass_abs": sum(d >= TAU for d in deltas),
                "outcome_abs": _outcome(all(d >= TAU for d in deltas), deltas),
                "outcome_ci": _outcome(lcb >= TAU, deltas),
                "control_batch": ctrl[SEEDS[0]]["batch"],
                "control_epochs": ctrl[SEEDS[0]]["epochs"],
                "n_scored": ctrl[SEEDS[0]]["n"],
            }
        )

    payload = {
        "tau_abs": TAU,
        "t_975_df4": T_975,
        "interval_rule": "mean - t_975 * SD(delta) / sqrt(5) >= tau",
        "rows": rows,
        "pending": pending,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    short = {"Credited": "Credited", "Consistent-positive, not credited": "Cons.-pos.", "Unsupported": "Unsupp."}
    lines = [
        "\\begin{table}[H]",
        "\\caption{Credit map. Differences are $10^{-3}$ validation AUC; the",
        "bracket is the lower 95\\% bound. The outcome is the same under both",
        "rules. Cons.-pos.:",
        "consistent-positive (not credited). Abbreviations are in",
        "Section~\\ref{sec:credit-gate}.}",
        "\\label{tab:credit-transfer}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{0.84}",
        "\\begin{tabular}{@{}llrl@{}}",
        "\\toprule",
        "Setting & Contrast & Mean [lower] & Outcome \\\\",
        "\\midrule",
    ]
    last = None
    for r in rows:
        if r["corpus"].startswith(ASSIST_1000S):
            continue
        corpus = r["corpus"] if r["corpus"] != last else ""
        if last is not None and r["corpus"] != last:
            lines.append("\\addlinespace")
        last = r["corpus"]
        lines.append(
            f"{corpus} & {r['contrast']} & ${_milli(r['mean_delta_val'])}$ "
            f"[${_milli(r['lcb95_delta_val'])}$] & {_one_outcome(r, short)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    _write_resolution_table(rows, short)
    _write_heatmap(rows)

    for r in rows:
        print(
            f"{r['corpus']:15} {r['contrast']:40} mean={r['mean_delta_val']:+.6f} "
            f"min={r['min_delta_val']:+.6f} sd_delta={r['sd_delta_val']:.6f} "
            f"lcb95={r['lcb95_delta_val']:+.6f} seed={r['pass_abs']}/5 {r['outcome_abs']} | "
            f"interval {r['outcome_ci']}"
        )
    for p in pending:
        print(f"PENDING {p}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
