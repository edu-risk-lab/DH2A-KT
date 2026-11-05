#!/usr/bin/env python3
"""Rewrite paper/tables/table_attn_assist.tex from credit_primary_test.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "tables" / "credit_primary_test.json"
OUT = REPO / "paper" / "tables" / "table_attn_assist.tex"

CONTRASTS = (
    r"Package vs.\ no-time",
    r"Aligned gap vs.\ T-zero",
    r"Aligned gap vs.\ T-misaligned",
)


def main() -> int:
    rows = [
        r
        for r in json.loads(SRC.read_text(encoding="utf-8"))["rows"]
        if r["setting"] == "ASSIST2012 sec attn" and r["family"] == "E"
    ]
    by = {(r["contrast"], r["fold"]): r for r in rows}
    missing = [c for c in CONTRASTS for f in (0, 1, 2) if (c, f) not in by]
    if missing:
        raise SystemExit(f"missing attn rows in {SRC}: {missing[:3]}...")

    body = []
    outcomes = []
    p_tau_max = 0.0
    for contrast in CONTRASTS:
        cells = []
        for f in (0, 1, 2):
            r = by[(contrast, f)]
            outcomes.append(r["outcome"])
            p_tau_max = max(p_tau_max, float(r["p_tau_adj"]))
            cells.append(
                f"$+{1e3 * r['mean']:.2f}$~$[+{1e3 * r['lcb95_two_sided']:.2f}]$ "
                f"({r['seed_k']}/10)"
            )
        body.append(f"{contrast} &\n" + " &\n".join(cells) + r" \\")

    uniq = sorted(set(outcomes))
    outcome_note = (
        "every row is " + uniq[0]
        if len(uniq) == 1
        else "row outcomes are " + ", ".join(uniq)
    )

    text = rf"""% Auto: scripts/65_attn_assist_table.py (from credit_primary_test.json)
\begin{{table}}[tbp]
\caption{{Exploratory (family~E) attention-encoder time contrasts on
ASSISTments~2012 one-second export, reference recipe, ten paired seeds.
Differences are mean paired pooled held-out evaluation AUC ($10^{{-3}}$);
brackets are lower ends of descriptive two-sided 95\% intervals; $k/10$ is
the seed-stability descriptor. One-sided Benjamini--Hochberg $p_\tau$
values are taken from the full family~E in
\texttt{{credit\_primary\_test.json}} (all $\ge {p_tau_max:.3f}$ here):
{outcome_note}. These rows are not in family~C.}}
\label{{tab:attn-assist}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
Contrast & Fold~0 & Fold~1 & Fold~2 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}; outcomes={uniq}; max p_tau={p_tau_max:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
