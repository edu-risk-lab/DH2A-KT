#!/usr/bin/env python3
"""Export GS Hau B8 Holm–Bonferroni notes from the A2 multi-seed matrix.

Computes paired one-sided t-tests for the two five-seed XES credit tests
(Q←KC vs twin; Linear Δt vs Q←KC) and writes a JSON summary used by the
manuscript credit-gate paragraph.

Usage:
    python scripts/33_b8_holm_credit_tests.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = ["42", "17", "1234", "0", "2024"]


def _vals(by: dict[str, list[dict]], arm: str, key: str = "val_auc") -> list[float]:
    m = {r["seed"]: float(r[key]) for r in by[arm] if r.get(key)}
    return [m[s] for s in SEEDS]


def main() -> int:
    matrix = REPO_ROOT / "results" / "tables" / "a2_multiseed_matrix.csv"
    if not matrix.exists():
        print(f"Missing {matrix}", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(matrix.open(encoding="utf-8")))
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)

    q_on = _vals(by, "hg_qkc_on")
    q_off = _vals(by, "hg_qkc_off")
    dt = _vals(by, "p0_dt_on")
    d_q = [a - b for a, b in zip(q_on, q_off)]
    d_t = [a - b for a, b in zip(dt, q_on)]

    from scipy import stats

    tests = []
    for name, d in (("qkc_vs_twin", d_q), ("dt_vs_qkc", d_t)):
        t, p = stats.ttest_1samp(d, 0.0, alternative="greater")
        tests.append(
            {
                "name": name,
                "n": len(d),
                "mean_delta_val": float(mean(d)),
                "sd_delta_val": float(stdev(d)),
                "t": float(t),
                "p_one_sided": float(p),
            }
        )

    # Holm–Bonferroni among the two five-seed tests (m=2), then also
    # conservatively among eight credit arms (m=8).
    ordered = sorted(tests, key=lambda x: x["p_one_sided"])
    for m in (2, 8):
        for i, row in enumerate(ordered):
            adj = min(1.0, row["p_one_sided"] * (m - i))
            row[f"p_holm_m{m}"] = float(adj)
            row[f"reject_holm_m{m}_alpha05"] = bool(adj < 0.05)

    out = {
        "source": str(matrix.relative_to(REPO_ROOT)).replace("\\", "/"),
        "seeds": SEEDS,
        "alpha": 0.05,
        "tests": ordered,
        "note": (
            "FAIL arms in table_negative_knowledge.tex remain single-seed "
            "and are judged only by the +0.002 gate."
        ),
    }
    out_path = REPO_ROOT / "results" / "tables" / "b8_holm_credit_tests.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for row in ordered:
        print(
            f"{row['name']}: mean_delta={row['mean_delta_val']:.5f} "
            f"t={row['t']:.2f} p={row['p_one_sided']:.3e} "
            f"holm_m2={row['p_holm_m2']:.3e} holm_m8={row['p_holm_m8']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
