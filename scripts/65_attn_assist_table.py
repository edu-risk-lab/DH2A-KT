#!/usr/bin/env python3
"""Rewrite attention-encoder family-E TeX tables from credit_primary_test.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "tables" / "credit_primary_test.json"
OUT_ASSIST = REPO / "paper" / "tables" / "table_attn_assist.tex"
OUT_JUNYI = REPO / "paper" / "tables" / "table_attn_junyi.tex"

CONTRASTS = (
    r"Package vs.\ no-time",
    r"Aligned gap vs.\ T-zero",
    r"Aligned gap vs.\ T-misaligned",
)


def _cells(by: dict, contrasts: tuple[str, ...]) -> tuple[list[str], list[str], float]:
    body = []
    outcomes = []
    p_tau_max = 0.0
    missing = [c for c in contrasts for f in (0, 1, 2) if (c, f) not in by]
    if missing:
        raise SystemExit(f"missing attn rows in {SRC}: {missing[:3]}...")
    for contrast in contrasts:
        cells = []
        for f in (0, 1, 2):
            r = by[(contrast, f)]
            outcomes.append(r["outcome"])
            p_tau_max = max(p_tau_max, float(r["p_tau_adj"]))
            cells.append(
                f"${1e3 * r['mean']:+.2f}$~$[{1e3 * r['lcb95_two_sided']:+.2f}]$ "
                f"({r['seed_k']}/10)"
            )
        body.append(f"{contrast} &\n" + " &\n".join(cells) + r" \\")
    return body, outcomes, p_tau_max


def _outcome_note(outcomes: list[str]) -> str:
    uniq = sorted(set(outcomes))
    if len(uniq) == 1:
        return "every row is " + uniq[0]
    return "row outcomes are " + ", ".join(uniq)


def main() -> int:
    all_rows = json.loads(SRC.read_text(encoding="utf-8"))["rows"]
    assist = [
        r for r in all_rows if r["setting"] == "ASSIST2012 sec attn" and r["family"] == "E"
    ]
    junyi = [
        r
        for r in all_rows
        if r["setting"] == "Junyi attn. history-only" and r["family"] == "E"
    ]
    assist_body, assist_out, assist_p = _cells(
        {(r["contrast"], r["fold"]): r for r in assist}, CONTRASTS
    )
    junyi_contrasts = (
        r"History-only package",
        r"History-only gap vs.\ T-zero",
    )
    junyi_body, junyi_out, junyi_p = _cells(
        {(r["contrast"], r["fold"]): r for r in junyi}, junyi_contrasts
    )

    assist_text = rf"""% Auto: scripts/65_attn_assist_table.py (from credit_primary_test.json)
\begin{{table}}[tbp]
\caption{{Exploratory (family~E) attention-encoder time contrasts on
ASSISTments~2012 one-second export, reference recipe, ten paired seeds.
Differences are mean paired pooled held-out evaluation AUC ($10^{{-3}}$);
brackets are lower ends of descriptive two-sided 95\% intervals; $k/10$ is
the seed-stability descriptor. One-sided Benjamini--Hochberg $p_\tau$
values are taken from the full family~E in
\texttt{{credit\_primary\_test.json}} (all $\ge {assist_p:.3f}$ here):
{_outcome_note(assist_out)}. These rows are not in family~C.}}
\label{{tab:attn-assist}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
Contrast & Fold~0 & Fold~1 & Fold~2 \\
\midrule
""" + "\n".join(assist_body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    junyi_text = rf"""% Auto: scripts/65_attn_assist_table.py (from credit_primary_test.json)
\begin{{table}}[tbp]
\caption{{Exploratory (family~E) attention-encoder history-only contrasts
on all Junyi learners, reference recipe, ten paired seeds. The onset
rebuild failed the availability audit, so these arms feed the completion
gap to the recurrent input only. Differences are mean paired pooled
held-out evaluation AUC ($10^{{-3}}$); brackets are lower ends of
descriptive two-sided 95\% intervals; $k/10$ is the seed-stability
descriptor. One-sided Benjamini--Hochberg $p_\tau$ values are taken from
the full family~E in \texttt{{credit\_primary\_test.json}}
(all $\ge {junyi_p:.3f}$ here): {_outcome_note(junyi_out)}.
These rows are not in family~C.}}
\label{{tab:attn-junyi}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
Contrast & Fold~0 & Fold~1 & Fold~2 \\
\midrule
""" + "\n".join(junyi_body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    OUT_ASSIST.write_text(assist_text, encoding="utf-8")
    OUT_JUNYI.write_text(junyi_text, encoding="utf-8")
    print(
        f"wrote {OUT_ASSIST}; outcomes={sorted(set(assist_out))}; "
        f"max p_tau={assist_p:.4f}"
    )
    print(
        f"wrote {OUT_JUNYI}; outcomes={sorted(set(junyi_out))}; "
        f"max p_tau={junyi_p:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
