#!/usr/bin/env python3
"""Family-E three-arm table for xes-attempt (one row per attempt).

docs/preregistration_2026-11-05.md Section 6. Reads finished CSVs of
scripts/50_credit_ladder_plan.py --group xes-attempt. Does not invent AUC.

    python scripts/67_xes_attempt_table.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
SEEDS = (0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407)
TAU = 0.002
ALPHA = 0.025
T_CRIT9 = 2.2621571627409915
ARMS = ("time-package", "t-boundary", "no-time")
PAIRS = (
    ("Package vs.\\ no time", "time-package", "no-time"),
    ("T-boundary vs.\\ no time", "t-boundary", "no-time"),
    ("Package vs.\\ T-boundary", "time-package", "t-boundary"),
)


def _auc(arm: str, seed: int) -> float | None:
    tag = f"ladder_xes-attempt_{arm}_s{seed}"
    path = TABLES / f"dh2_kt_{tag}_vs_p0.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return float(next(csv.DictReader(fh))["dh2_kt_auc"])


def _t_sf(t: float, df: int) -> float:
    # Survival function of Student-t via regularised incomplete beta if scipy exists.
    try:
        from scipy.stats import t as student_t

        return float(student_t.sf(t, df))
    except Exception:
        # Normal tail fallback.
        return 0.5 * math.erfc(t / math.sqrt(2.0))


def _bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=0):
        k = m - rank
        val = min(prev, pvals[i] * m / k)
        adj[i] = val
        prev = val
    return adj


def main() -> int:
    rows = []
    pending = False
    raw_ptau, raw_p0 = [], []
    for name, on, ctrl in PAIRS:
        deltas = []
        missing = []
        for seed in SEEDS:
            a, b = _auc(on, seed), _auc(ctrl, seed)
            if a is None or b is None:
                missing.append(seed)
                continue
            deltas.append(a - b)
        if len(deltas) < 2:
            pending = True
            rows.append({"name": name, "pending": missing})
            raw_ptau.append(1.0)
            raw_p0.append(1.0)
            continue
        n = len(deltas)
        mean = sum(deltas) / n
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
        lcb = mean - T_CRIT9 * se
        t_tau = (mean - TAU) / se
        t_0 = mean / se
        p_tau = _t_sf(t_tau, n - 1)
        p_0 = _t_sf(t_0, n - 1)
        raw_ptau.append(p_tau)
        raw_p0.append(p_0)
        rows.append(
            {
                "name": name,
                "n": n,
                "missing": missing,
                "mean": mean,
                "lcb": lcb,
                "seed_k": sum(d >= TAU for d in deltas),
                "p_tau": p_tau,
                "p_0": p_0,
            }
        )
    p_tau_bh = _bh(raw_ptau)
    p_0_bh = _bh(raw_p0)
    for r, a, b in zip(rows, p_tau_bh, p_0_bh):
        r["p_tau_bh"] = a
        r["p_0_bh"] = b
        if "pending" in r:
            r["outcome"] = "pending"
        elif a <= ALPHA:
            r["outcome"] = "Credited (exploratory)"
        elif b <= ALPHA:
            r["outcome"] = "Positive, not credited"
        else:
            r["outcome"] = "Unsupported"

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "xes_attempt_threeway.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    tex = [
        r"\begin{table}[tbp]",
        r"\caption{Confirmatory time-versus-boundary comparison on XES3G5M fold~0",
        r"with one row per attempt (docs/preregistration\_2026-11-05.md",
        r"Section~6). LSTM reference recipe, ten seeds, pooled held-out",
        r"evaluation AUC, $\tau{=}{+}2.0\times 10^{-3}$, family~E (BH over these",
        r"three contrasts). The two branched arms share 256 time-map parameters.",
        r"Not added to the 29 family~C rows"
        + (" (pending until the \\texttt{xes-attempt} runs exist)." if pending else ".")
        + "}",
        r"\label{tab:attempt-threeway}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Contrast & Mean [LCB] ($k/10$) & $p_{\tau}^{\mathrm{BH}}$ & $p_{0}^{\mathrm{BH}}$ & Out. \\",
        r"\midrule",
    ]
    short = {
        "Credited (exploratory)": "C$^{\\mathrm{E}}$",
        "Positive, not credited": "P",
        "Unsupported": "U",
        "pending": "pending",
    }
    for r in rows:
        if r["outcome"] == "pending":
            tex.append(f"{r['name']} & pending & pending & pending & pending \\\\")
            continue
        tex.append(
            f"{r['name']} & {1e3 * r['mean']:+.2f} [{1e3 * r['lcb']:+.2f}] "
            f"({r['seed_k']}/{r['n']}) & {r['p_tau_bh']:.3f} & {r['p_0_bh']:.3f} & "
            f"{short[r['outcome']]} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (REPO / "paper" / "tables" / "table_attempt_threeway.tex").write_text("\n".join(tex), encoding="utf-8")
    print("wrote paper/tables/table_attempt_threeway.tex", "pending" if pending else "filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
