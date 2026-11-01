#!/usr/bin/env python3
"""TeX tables for the round-3 primary credit rule (docs/preregistration_round3.md).

No training. Reads results/tables/credit_primary_test.json and writes:

  paper/tables/table_credit_primary_c.tex   — family C map (main)
  paper/tables/table_credit_primary_changes.tex — rows that change vs old test-interval
  paper/tables/table_credit_primary_var.tex — fold variance components (appendix)

    python scripts/64_credit_primary_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "tables" / "credit_primary_test.json"
OUT = REPO / "paper" / "tables"

SHORT = {
    "Credited": "C",
    "Positive, not credited": "P",
    "Unsupported": "U",
    "Diagnostic": "D",
}


def _e(s: str) -> str:
    return s.replace("_", "\\_")


def _mean_cell(r: dict) -> str:
    m = 1e3 * r["mean"]
    k = r["seed_k"]
    n = len(r["deltas"])
    out = SHORT[r["outcome"]]
    if r.get("seed_variable"):
        out = f"{out}$^*$"
    return f"{m:+.2f} ({k}/{n}) {out}"


def write_family_c(rows: list[dict]) -> None:
    c_rows = [r for r in rows if r["family"] == "C"]
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Confirmatory credit map (family~C) under the primary rule:",
        r"held-out test AUC, one-sided paired $t$-tests of",
        r"$H_0{:}\,\mu\le\tau$ and $H_0{:}\,\mu\le 0$ with $\tau{=}{+}2.0\times 10^{-3}$,",
        r"Holm within the family, $\alpha{=}0.025$ one-sided. Mean difference is",
        r"in units of $10^{-3}$ test AUC; $k/10$ is the seed-stability descriptor",
        r"(not a decision rule). Outcome: C credited, P positive not credited, U",
        r"unsupported. A starred C is credited with $k{<}10$ (seed-variable).",
        r"Junyi completion-time rows are family~D and are omitted here;",
        r"history-only Junyi and ASSISTments attention contrasts are family~E.}",
        r"\label{tab:credit-primary-c}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Setting & Contrast & Mean & $p_{\tau}^{\mathrm{Holm}}$ & Outcome \\",
        r"\midrule",
    ]
    for r in c_rows:
        p = r["p_tau_adj"]
        p_s = f"{p:.2e}" if p is not None and p < 1e-3 else (f"{p:.3f}" if p is not None else "---")
        out = SHORT[r["outcome"]] + ("$^*$" if r.get("seed_variable") else "")
        lines.append(
            f"{r['corpus']} & {r['contrast']} & "
            f"{1e3 * r['mean']:+.2f} ({r['seed_k']}/10) & {p_s} & {out} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_credit_primary_c.tex").write_text("\n".join(lines), encoding="utf-8")


def write_changes(summary: dict) -> None:
    ch = summary["changed_vs_old_test_interval"]
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Family~C and~E rows whose primary outcome (test AUC)",
        r"differs from the earlier test-only interval-rule outcome. Empty if none.",
        r"Validation-interval disagreements are listed in the appendix.}",
        r"\label{tab:credit-primary-changes}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"Setting & Contrast & Family & Old test-interval $\to$ new \\",
        r"\midrule",
    ]
    if not ch:
        lines.append(r"\multicolumn{4}{c}{(none)} \\")
    for r in ch:
        lines.append(
            f"{r['corpus']} & {r['contrast']} & {r['family']} & "
            f"{r['old']} $\\to$ {r['new']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_credit_primary_changes.tex").write_text("\n".join(lines), encoding="utf-8")


def write_var(comps: list[dict]) -> None:
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{One-way random-effects variance components for family~C",
        r"contrasts run on three learner folds (method of moments;",
        r"between-fold variance truncated at 0). Pooled mean is the mean of",
        r"the three fold means with a $t$ interval on $\mathrm{df}{=}2$; units",
        r"$10^{-3}$ test AUC.}",
        r"\label{tab:credit-primary-var}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Setting & Contrast & Pooled & 95\% CI & SD$_{\mathrm{seed}}$ & SD$_{\mathrm{fold}}$ \\",
        r"\midrule",
    ]
    for c in comps:
        lo, hi = c["pooled_ci95_df2"]
        lines.append(
            f"{c['setting']} & {c['contrast']} & "
            f"{1e3 * c['pooled_mean']:+.2f} & "
            f"[{1e3 * lo:+.2f},{1e3 * hi:+.2f}] & "
            f"{1e3 * c['sd_seed_within_fold']:.2f} & "
            f"{1e3 * c['sd_between_fold']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_credit_primary_var.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    write_family_c(data["rows"])
    write_changes(data["summary"])
    write_var(data["variance_components"])
    print("wrote table_credit_primary_{c,changes,var}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
