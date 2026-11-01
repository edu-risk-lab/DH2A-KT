#!/usr/bin/env python3
"""Rewrite paper/tables/table_attn_assist.tex from ladder CSVs."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "tables"
OUT = REPO / "paper" / "tables" / "table_attn_assist.tex"
T975 = 2.2621571627409915
TAU = 0.002


def arm(group: str, name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for p in ROOT.glob(f"dh2_kt_ladder_{group}_{name}_s*_vs_p0.csv"):
        rest = p.name[len(f"dh2_kt_ladder_{group}_{name}_s") : -len("_vs_p0.csv")]
        seed = int(rest)
        row = next(r for r in csv.DictReader(p.open(newline="", encoding="utf-8")) if r["fold"] != "mean")
        out[seed] = float(row["dh2_kt_auc"])
    return out


def cell(on: dict[int, float], off: dict[int, float]) -> str:
    seeds = sorted(set(on) & set(off))
    d = [on[s] - off[s] for s in seeds]
    n = len(d)
    m = statistics.mean(d)
    lo = m - T975 * statistics.stdev(d) / math.sqrt(n)
    k = sum(x >= TAU for x in d)
    return f"$+{1e3 * m:.2f}$~$[+{1e3 * lo:.2f}]$ (${k}/10$)"


def main() -> int:
    rows = []
    for label, a, b in (
        (r"Package vs.\ no-time", "time-package", "no-time"),
        (r"Aligned gap vs.\ T-zero", "time-package", "t-zero"),
        (r"Aligned gap vs.\ T-misaligned", "time-package", "t-misaligned"),
    ):
        cells = [cell(arm(f"attn-assist-fold{f}", a), arm(f"attn-assist-fold{f}", b)) for f in (0, 1, 2)]
        rows.append(f"{label} &\n" + " &\n".join(cells) + r" \\")
    text = r"""% Auto: scripts/65_attn_assist_table.py
\begin{table}[tbp]
\caption{Exploratory (family~E) attention-encoder time contrasts on
ASSISTments~2012 one-second export, reference recipe, ten paired seeds.
Differences are mean paired held-out test AUC in units of $10^{-3}$;
the bracket is the lower end of a descriptive two-sided 95\% interval
(not the one-sided credit test); $k/10$ is the seed-stability
descriptor. Credit decisions use one-sided Benjamini--Hochberg over
family~E: every row is Positive, not credited (none reach adjusted
$p_\tau\le 0.025$). These rows are not in family~C.}
\label{tab:attn-assist}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Contrast & Fold~0 & Fold~1 & Fold~2 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
