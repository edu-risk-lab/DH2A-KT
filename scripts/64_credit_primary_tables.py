#!/usr/bin/env python3
"""TeX tables for the primary credit rule (docs/preregistration_primary_rule.md).

No training. Reads results/tables/credit_primary_test.json and writes:

  paper/tables/table_credit_primary_c.tex
  paper/tables/table_credit_primary_changes.tex
  paper/tables/table_credit_primary_var.tex
  paper/tables/table_credit_primary_mde.tex
  paper/tables/table_credit_primary_tau.tex

    python scripts/64_credit_primary_tables.py
"""

from __future__ import annotations

import json
import math
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

# One-sided Holm at α/|C| with |C|=29, df=9 (t critical ≈ 4.40).
T_HOLM29 = 4.40
ALPHA = 0.025
N_C = 29
N_SEED = 10


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"


def write_family_c(rows: list[dict]) -> None:
    c_rows = [r for r in rows if r["family"] == "C"]
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Primary credit map (family~C) under the primary rule:",
        r"pooled held-out evaluation AUC (original valid$\cup$test), one-sided",
        r"paired $t$-tests of $H_0{:}\,\mu\le\tau$ and $H_0{:}\,\mu\le 0$ with",
        r"$\tau{=}{+}2.0\times 10^{-3}$, Holm (FWER) within the family,",
        r"$\alpha{=}0.025$ one-sided. Mean and the lower end of the two-sided",
        r"95\% interval (LCB) are in units of $10^{-3}$ evaluation AUC; $k/10$",
        r"is the seed-stability descriptor (not a decision rule). Outcome: C",
        r"credited, P positive not credited, U unsupported. A starred C is",
        r"credited with $k{<}10$ (seed-variable). Junyi completion-time rows",
        r"are family~D and are omitted; history-only Junyi and ASSISTments",
        r"attention contrasts are family~E.}",
        r"\label{tab:credit-primary-c}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Setting & Contrast & Mean [LCB] ($k/10$) & $p_{\tau}^{\mathrm{H}}$ & $p_{0}^{\mathrm{H}}$ & Out. \\",
        r"\midrule",
    ]
    for r in c_rows:
        out = SHORT[r["outcome"]] + ("$^*$" if r.get("seed_variable") else "")
        lines.append(
            f"{r['corpus']} & {r['contrast']} & "
            f"{1e3 * r['mean']:+.2f} [{1e3 * r['lcb95_two_sided']:+.2f}] "
            f"({r['seed_k']}/10) & {_fmt_p(r['p_tau_adj'])} & "
            f"{_fmt_p(r['p_0_adj'])} & {out} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_credit_primary_c.tex").write_text("\n".join(lines), encoding="utf-8")


def write_changes(summary: dict, rows: list[dict]) -> None:
    ch = summary["changed_vs_old_test_interval"]
    by = {(r["corpus"], r["contrast"]): r for r in rows}
    n_c = sum(1 for r in rows if r["family"] == "C")
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Family~C rows whose primary-rule outcome on pooled held-out",
        r"evaluation AUC differs from the earlier test-interval rule on the",
        r"same field. Mean, two-sided LCB, and $k/10$ are in units of",
        r"$10^{-3}$ AUC under the primary metric. Old~=~interval rule",
        r"(Credited if LCB$\,\ge\tau$); New~=~primary Holm rule of",
        r"Section~\ref{sec:protocol-threshold}. The other "
        + str(n_c - len(ch))
        + r" family~C rows keep the same primary-rule outcome as that",
        r"earlier test-interval rule. This table is a reanalysis, not a",
        r"confirmatory test. Validation-interval disagreements are archived",
        r"in \ref{app:ten-seed-stats}.}",
        r"\label{tab:credit-primary-changes}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrll@{}}",
        r"\toprule",
        r"Setting & Contrast & Mean [LCB] ($k/10$) & Old interval & New primary \\",
        r"\midrule",
    ]
    if not ch:
        lines.append(r"\multicolumn{5}{c}{(none)} \\")
    for item in ch:
        r = by[(item["corpus"], item["contrast"])]
        lines.append(
            f"{r['corpus']} & {r['contrast']} & "
            f"{1e3 * r['mean']:+.2f} [{1e3 * r['lcb95_two_sided']:+.2f}] "
            f"({r['seed_k']}/10) & {item['old']} & {item['new']} \\\\"
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
        r"$10^{-3}$ test AUC. For every contrast shown,",
        r"$\mathrm{SD}_{\mathrm{fold}}\le\mathrm{SD}_{\mathrm{seed}}$.}",
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


def write_mde(rows: list[dict]) -> None:
    # Representative SDs from family-C package rows.
    pkgs = [
        r
        for r in rows
        if r["family"] == "C" and r["contrast"] == "Package, zero incidence"
    ]
    xes = next(r for r in pkgs if r["corpus"] == "XES3G5M")
    assist = next(r for r in pkgs if "ref.\\ fold 1" in r["corpus"] or r["corpus"].endswith("ref. fold 1"))
    # fallback keys
    assist = next(
        (r for r in pkgs if "ref" in r["corpus"] and "fold 1" in r["corpus"]),
        next(r for r in pkgs if "ASSIST2012 ref" in r["corpus"]),
    )
    sds = {
        "XES3G5M package (observed)": xes["sd"],
        "ASSIST2012 ref.\\ fold~1 package (observed)": assist["sd"],
        "Illustrative SD $0.50\\times 10^{-3}$": 0.0005,
    }
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Conservative sensitivity under the \emph{first}",
        r"Bonferroni step of family~C Holm ($m{=}29$, $|S|{=}10$, one-sided",
        r"$\alpha{=}0.025$; critical $t_{9,\alpha/m}\approx 4.40$). The mean",
        r"must clear $\tau{=}{+}2.0\times 10^{-3}$ by $t^\star s/\sqrt{10}$.",
        r"This is not the exact power of the full Holm procedure (which",
        r"depends on the other $p$-values). Units $10^{-3}$ evaluation AUC.",
        r"``Not credited'' on ASSISTments packages near $+2.5$ is lack of",
        r"evidence after adjustment, not absence of an effect.}",
        r"\label{tab:credit-primary-mde}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}p{5.6cm}rr@{}}",
        r"\toprule",
        r"Seed SD ($10^{-3}$) & Observed mean & Mean to clear first Holm step \\",
        r"\midrule",
    ]
    for label, sd in sds.items():
        margin = T_HOLM29 * sd / math.sqrt(N_SEED)
        m50 = 0.002 + margin
        if "Illustrative" in label:
            obs = "---"
        elif "XES" in label:
            obs = f"{1e3 * xes['mean']:+.2f}"
        else:
            obs = f"{1e3 * assist['mean']:+.2f}"
        lines.append(
            f"{label} (${1e3 * sd:.2f}$) & {obs} & {1e3 * m50:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_credit_primary_mde.tex").write_text("\n".join(lines), encoding="utf-8")


def write_tau(rows: list[dict]) -> None:
    taus = ["1.0", "1.5", "2.0", "2.5", "3.0"]
    c_rows = [r for r in rows if r["family"] == "C"]
    # Count outcomes at each tau
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Sensitivity of family~C primary outcomes to $\tau$ on",
        r"pooled held-out evaluation AUC (one-sided paired $t$, Holm within",
        r"family~C, $\alpha{=}0.025$). Counts are over the $29$ family~C contrasts.",
        r"The operating choice in the main text is $\tau{=}{+}2.0\times 10^{-3}$.",
        r"Earlier validation-AUC threshold recounts under the interval/seed",
        r"rules are archived as Tables~\ref{tab:threshold-all-10seed}",
        r"and~\ref{tab:threshold-all-10seed-b}.}",
        r"\label{tab:credit-primary-tau}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"$\tau$ ($10^{-3}$) & Credited & Positive, not credited & Unsupported & ASSIST ref.\ pkgs credited \\",
        r"\midrule",
    ]
    assist_pkgs = [
        r
        for r in c_rows
        if "ASSIST2012 ref" in r["corpus"] and r["contrast"] == "Package, zero incidence"
    ]
    for tau in taus:
        counts = {"Credited": 0, "Positive, not credited": 0, "Unsupported": 0}
        assist_c = 0
        for r in c_rows:
            lab = r["tau_curve"][tau]
            # normalize old naming if any
            if lab.startswith("Positive"):
                lab = "Positive, not credited"
            counts[lab] = counts.get(lab, 0) + 1
        for r in assist_pkgs:
            if r["tau_curve"][tau] == "Credited":
                assist_c += 1
        lines.append(
            f"${tau}$ & {counts['Credited']} & {counts['Positive, not credited']} & "
            f"{counts['Unsupported']} & {assist_c}/{len(assist_pkgs)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_credit_primary_tau.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    write_family_c(data["rows"])
    write_changes(data["summary"], data["rows"])
    write_var(data["variance_components"])
    write_mde(data["rows"])
    write_tau(data["rows"])
    print("wrote table_credit_primary_{c,changes,var,mde,tau}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
