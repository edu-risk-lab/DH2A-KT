#!/usr/bin/env python3
"""Write paper/figures/fig_credit_forest.tex from credit_primary_test.json.

All family-C contrasts (29); mean and two-sided 95% LCB/UCB on pooled
evaluation AUC ×10^{-3}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "tables" / "credit_primary_test.json"
OUT = REPO / "paper" / "figures" / "fig_credit_forest.tex"
T975 = 2.2621571627409915  # t_9, 0.975


def ucb(mean: float, sd: float, n: int = 10) -> float:
    return mean + T975 * sd / math.sqrt(n)


def main() -> int:
    rows = [
        r
        for r in json.loads(SRC.read_text(encoding="utf-8"))["rows"]
        if r["family"] == "C"
    ]

    def key(r: dict) -> tuple:
        corp = r["corpus"]
        order = 0
        if corp.startswith("XES3G5M") and "attn" not in corp:
            order = 0
        elif "attn" in corp:
            order = 1
        elif "ref" in corp:
            order = 3
        else:
            order = 2
        ctype = 0 if "incidence" in r["contrast"] else (1 if "Package" in r["contrast"] else 2)
        return (order, corp, ctype, r["contrast"])

    rows = sorted(rows, key=key)
    n = len(rows)
    row_h = 0.34
    y0 = (n - 1) * row_h
    xmin, xmax = -1.5, 4.5

    lines = [
        r"% Auto: scripts/66_credit_forest_plot.py",
        r"\begin{figure}[tbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tikzpicture}[x=1.2cm,y=1cm,font=\scriptsize\sffamily]",
        rf"\draw[gray!60] ({xmin},-0.45) -- ({xmin},{y0 + 0.55});",
        rf"\draw[thick] (0,-0.45) -- (0,{y0 + 0.55});",
        rf"\draw[thick,dashed] (2,-0.45) -- (2,{y0 + 0.55});",
        rf"\foreach \x in {{1,3}} {{\draw[gray!40] (\x,-0.45) -- (\x,{y0 + 0.55});}}",
        rf"\node[below] at (0,-0.45) {{$0$}};",
        rf"\node[below] at (1,-0.45) {{$1$}};",
        rf"\node[below] at (2,-0.45) {{$\tau$}};",
        rf"\node[below] at (3,-0.45) {{$3$}};",
        rf"\node[above] at ({(xmin+xmax)/2},{y0 + 0.7}) {{Paired $\Delta$ pooled eval.\ AUC ($10^{{-3}}$)}};",
    ]

    colors = {
        "Credited": "blue!55",
        "Positive, not credited": "orange!60",
        "Unsupported": "gray!50",
    }
    for i, r in enumerate(rows):
        y = y0 - i * row_h
        m = 1e3 * r["mean"]
        lo = 1e3 * r["lcb95_two_sided"]
        hi = 1e3 * ucb(r["mean"], r["sd"])
        lo = max(lo, xmin + 0.05)
        hi = min(hi, xmax - 0.05)
        col = colors.get(r["outcome"], "black")
        label = f"{r['corpus']}: {r['contrast']}".replace("Q$\\leftarrow$KC", r"Q$\leftarrow$KC")
        label = (
            label.replace("Package, zero incidence", "Pkg zero")
            .replace("Aligned gap vs.\\ T-zero", r"Gap vs T-0")
            .replace("Q$\\leftarrow$KC incidence", r"Q$\leftarrow$KC")
            .replace("ASSIST2012", "ASSIST")
            .replace("XES3G5M", "XES")
        )
        lines.append(rf"\draw[{col}, thick] ({lo},{y}) -- ({hi},{y});")
        lines.append(
            rf"\node[draw, circle, fill={col}, inner sep=1.0pt] at ({m},{y}) {{}};"
        )
        lines.append(rf"\node[anchor=east, align=right] at ({xmin - 0.08},{y}) {{{label}}};")

    lines += [
        rf"\draw[->] ({xmin},-0.45) -- ({xmax + 0.2},-0.45);",
        r"\end{tikzpicture}}",
        rf"\caption{{All ${n}$ family~C contrasts under the primary rule on",
        r"pooled held-out evaluation AUC. Points are paired mean differences",
        r"($10^{-3}$ AUC); horizontal bars are \emph{unadjusted} two-sided",
        r"95\% intervals. The solid line is $0$; the dashed line is",
        r"$\tau{=}{+}2.0\times 10^{-3}$. Colour encodes the Holm-adjusted",
        r"decision (blue Credited, orange Positive not credited, grey",
        r"Unsupported). Credit is decided by the one-sided test of",
        r"$\mu\le\tau$ after Holm, not by whether the interval excludes",
        r"$\tau$.}",
        r"\label{fig:credit-forest}",
        r"\end{figure}",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
