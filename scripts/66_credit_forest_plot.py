#!/usr/bin/env python3
"""Write paper/figures/fig_credit_forest.tex from credit_primary_test.json.

Family-C contrasts only; mean and two-sided 95% LCB/UCB on test AUC ×10^{-3}.
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
    # Keep a readable subset: packages, aligned gaps, incidence on fold 0 / key folds
    keep = []
    for r in rows:
        c = r["contrast"]
        corp = r["corpus"]
        if c == "Q$\\leftarrow$KC incidence" and "fold" in corp:
            continue
        if "Aligned gap" in c and "fold" in corp and "attn" not in corp and "ref" not in corp:
            # keep XES fold gaps
            pass
        keep.append(r)

    # Order: XES LSTM, XES attn, ASSIST, ASSIST ref
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

    rows = sorted(keep, key=key)
    n = len(rows)
    row_h = 0.38
    y0 = (n - 1) * row_h
    xmin, xmax = -1.5, 4.5

    lines = [
        r"% Auto: scripts/66_credit_forest_plot.py",
        r"\begin{figure}[tbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tikzpicture}[x=1.35cm,y=1cm,font=\scriptsize\sffamily]",
        rf"\draw[gray!60] ({xmin},-0.35) -- ({xmin},{y0 + 0.55});",
        rf"\draw[thick] (0,-0.35) -- (0,{y0 + 0.55});",
        rf"\draw[thick,dashed] (2,-0.35) -- (2,{y0 + 0.55});",
        rf"\node[below] at (0,-0.35) {{$0$}};",
        rf"\node[below] at (2,-0.35) {{$\tau$}};",
        rf"\node[above] at ({(xmin+xmax)/2},{y0 + 0.7}) {{Paired $\Delta$ test AUC ($10^{{-3}}$)}};",
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
        # shorten
        label = (
            label.replace("Package, zero incidence", "Pkg zero")
            .replace("Aligned gap vs.\\ T-zero", r"Gap vs T-0")
            .replace("Q$\\leftarrow$KC incidence", r"Q$\leftarrow$KC")
            .replace("ASSIST2012", "ASSIST")
            .replace("XES3G5M", "XES")
        )
        lines.append(rf"\draw[{col}, thick] ({lo},{y}) -- ({hi},{y});")
        lines.append(
            rf"\node[draw, circle, fill={col}, inner sep=1.1pt] at ({m},{y}) {{}};"
        )
        lines.append(rf"\node[anchor=east, align=right] at ({xmin - 0.08},{y}) {{{label}}};")

    lines += [
        rf"\draw[->] ({xmin},-0.35) -- ({xmax + 0.2},-0.35);",
        r"\end{tikzpicture}}",
        r"\caption{Family~C contrasts under the primary rule on held-out test",
        r"AUC. Points are paired mean differences ($10^{-3}$ AUC); horizontal",
        r"bars are two-sided 95\% intervals. The solid line is $0$; the dashed",
        r"line is $\tau{=}{+}2.0\times 10^{-3}$. Colour: blue Credited, orange",
        r"Positive not credited, grey Unsupported (Holm within family~C).",
        r"Credit is decided by the one-sided test of $\mu\le\tau$, not by",
        r"whether the interval excludes $\tau$.}",
        r"\label{fig:credit-forest}",
        r"\end{figure}",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
