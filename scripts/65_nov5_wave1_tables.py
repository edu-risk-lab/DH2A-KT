#!/usr/bin/env python3
"""TeX tables for the 5 Nov 2026 advisor comments (wave 1; no training).

Reads results/tables/learner_bootstrap.json and credit_primary_test.json.
Writes:

  paper/tables/table_learner_bootstrap.tex
  paper/tables/table_learner_bootstrap_all.tex
  paper/tables/table_availability.tex

    python scripts/65_nov5_wave1_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOT = json.loads((REPO / "results" / "tables" / "learner_bootstrap.json").read_text(encoding="utf-8"))
CREDIT = json.loads((REPO / "results" / "tables" / "credit_primary_test.json").read_text(encoding="utf-8"))
OUT = REPO / "paper" / "tables"

# Display names: (setting, fold, contrast) for the main-text subset.
MAIN_KEYS = [
    ("XES3G5M", 0, "Package, zero incidence"),
    ("XES3G5M", 1, "Package, zero incidence"),
    ("XES3G5M", 2, "Package, zero incidence"),
    ("XES3G5M", 0, "Aligned gap vs.\\ T-zero"),
    ("XES3G5M attn.", 0, "Package, zero incidence"),
    ("ASSIST2012", 0, "Package, zero incidence"),
    ("ASSIST2012 ref.", 1, "Package, zero incidence"),
    ("ASSIST2012 ref.", 2, "Package, zero incidence"),
]


def _credit_row(setting: str, fold: int, contrast: str) -> dict | None:
    for r in CREDIT["rows"]:
        if r["family"] != "C":
            continue
        if r["contrast"] != contrast:
            continue
        if int(r["fold"]) != int(fold):
            continue
        # credit JSON uses corpus labels like "XES3G5M attn.\\ fold 1"
        corpus = r["corpus"].replace("\\", "")
        if setting == "XES3G5M" and corpus.startswith("XES3G5M") and "attn" not in corpus:
            if fold == 0 and corpus == "XES3G5M":
                return r
            if fold != 0 and f"fold {fold}" in corpus:
                return r
        if setting == "XES3G5M attn." and "attn" in corpus:
            if fold == 0 and "fold" not in corpus:
                return r
            if fold != 0 and f"fold {fold}" in corpus:
                return r
        if setting == "ASSIST2012" and corpus.startswith("ASSIST2012") and "ref" not in corpus:
            if fold == 0 and corpus == "ASSIST2012":
                return r
            if fold != 0 and f"fold {fold}" in corpus:
                return r
        if setting == "ASSIST2012 ref." and "ref" in corpus:
            if fold == 0 and "fold" not in corpus:
                return r
            if fold != 0 and f"fold {fold}" in corpus:
                return r
    return None


def _boot_rows() -> list[dict]:
    return [r for r in BOOT["rows"] if "pending" not in r]


def _label(r: dict) -> str:
    if r["fold"] == 0:
        return r["setting"]
    return f"{r['setting']} fold {r['fold']}"


def _line(r: dict) -> str:
    cr = _credit_row(r["setting"], r["fold"], r["contrast"])
    sd_seed = 1e3 * cr["sd"] if cr else float("nan")
    lo, hi = r["boot_ci95"]
    flips = 100.0 * r["decision_flip_share"]
    acc = 1e3 * r["practical_on_minus_control"]["accuracy"]
    return (
        f"{_label(r)} & {r['contrast']} & "
        f"{1e3 * r['mean_delta_exact']:+.2f} "
        f"[{1e3 * lo:+.2f},{1e3 * hi:+.2f}] & "
        f"{sd_seed:.2f} & {1e3 * r['boot_sd_mean']:.2f} & "
        f"{acc:+.2f} & {flips:.1f} \\\\"
    )


def write_bootstrap_main() -> None:
    by = {(r["setting"], r["fold"], r["contrast"]): r for r in _boot_rows()}
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Learner-clustered bootstrap of family~C contrasts",
        r"(docs/preregistration\_primary\_rule.md Section~5;",
        r"$B{=}2000$ resamples of held-out learners, seed 20261023).",
        r"Mean $\Delta$ is the mean over ten seeds of pooled held-out",
        r"evaluation AUC (on minus control), $\times 10^{-3}$. The 95\%",
        r"percentile interval resamples learners, not training seeds.",
        r"SD$_{\mathrm{seed}}$ is the sample SD of the ten paired",
        r"$\Delta_s$; SD$_{\mathrm{learn}}$ is the bootstrap SD of that",
        r"mean. Acc.\ flip is the mean paired change in accuracy at $0.5$",
        r"($\times 10^{-3}$); Flip~\% is the share of scored positions",
        r"whose $0.5$ decision differs between arms. These intervals do",
        r"not decide credit. A credited LSTM package and an uncredited",
        r"attention package is not a test that the two effects differ.}",
        r"\label{tab:learner-bootstrap}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Setting & Contrast & Mean [learner 95\%] & SD$_{\mathrm{seed}}$ &"
        r" SD$_{\mathrm{learn}}$ & Acc.\ $\Delta$ & Flip~\% \\",
        r"\midrule",
    ]
    for key in MAIN_KEYS:
        r = by.get(key)
        if r is None:
            continue
        lines.append(_line(r))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (OUT / "table_learner_bootstrap.tex").write_text("\n".join(lines), encoding="utf-8")


def write_bootstrap_all() -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Learner-clustered bootstrap for every family~C contrast",
        r"(same method as Table~\ref{tab:learner-bootstrap}). Units",
        r"$10^{-3}$ AUC except Flip~\%.}",
        r"\label{tab:learner-bootstrap-all}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Setting & Contrast & Mean [learner 95\%] & SD$_{\mathrm{seed}}$ &"
        r" SD$_{\mathrm{learn}}$ & Acc.\ $\Delta$ & Flip~\% \\",
        r"\midrule",
    ]
    for r in _boot_rows():
        lines.append(_line(r))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_learner_bootstrap_all.tex").write_text("\n".join(lines), encoding="utf-8")


def write_availability() -> None:
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{When each time-related input is created, and which",
        r"decision it may enter. Prediction time is the onset of item",
        r"$t{+}1$, when the learner opens it",
        r"(Section~\ref{sec:protocol-threshold}). The credited XES3G5M",
        r"LSTM package uses a query-side gap and is not a claim about",
        r"choosing the next item before it opens. Junyi completion-time",
        r"query gaps are family~D diagnostics.}",
        r"\label{tab:availability}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.4cm}"
        r">{\raggedright\arraybackslash}p{4.2cm}X@{}}",
        r"\toprule",
        r"Variable & Created when & Usable for \\",
        r"\midrule",
        r"History gap $\log(1{+}\Delta t)$ on the recurrent input &"
        r" After item $t$ opens (start-to-start on XES3G5M and"
        r" ASSISTments) & Scoring, or choosing item $t{+}1$ before it"
        r" opens \\",
        r"Query-side gap, XES3G5M / ASSISTments start times &"
        r" After item $t{+}1$ has opened & Scoring the opened item;"
        r" \textbf{not} choosing the next item \\",
        r"Junyi completion gap (\texttt{time\_done}) &"
        r" After item $t{+}1$ completes (includes its duration) &"
        r" Leakage diagnostic only (family~D) \\",
        r"Boundary indicator $1[\Delta t{=}0]$ &"
        r" Same timestamp as the gap it is built from &"
        r" The same decision as that gap \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    (OUT / "table_availability.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    write_bootstrap_main()
    write_bootstrap_all()
    write_availability()
    print("wrote table_learner_bootstrap.tex, table_learner_bootstrap_all.tex, table_availability.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
