#!/usr/bin/env python3
"""Export paired multi-seed credit tests for the KBS manuscript.

Combines the A2 matrix with the capacity-matched Q–KC matrix. The
capacity-mismatched pathway-off arm is intentionally excluded.

Usage:
    python scripts/33_b8_holm_credit_tests.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = ["42", "17", "1234", "0", "2024"]


def _vals(by: dict[str, list[dict]], arm: str, key: str = "val_auc") -> list[float]:
    m = {r["seed"]: float(r[key]) for r in by[arm] if r.get(key)}
    return [m[s] for s in SEEDS]


def main() -> int:
    a2_matrix = REPO_ROOT / "results" / "tables" / "a2_multiseed_matrix.csv"
    qkc_matrix = (
        REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_matrix.csv"
    )
    missing = [path for path in (a2_matrix, qkc_matrix) if not path.exists()]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(a2_matrix.open(encoding="utf-8")))
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)

    q_on = _vals(by, "hg_qkc_on")
    dt = _vals(by, "p0_dt_on")
    d_t_observed = [a - b for a, b in zip(dt, q_on)]

    qkc_rows = list(csv.DictReader(qkc_matrix.open(encoding="utf-8")))
    qkc_by: dict[str, dict[str, float]] = {}
    for row in qkc_rows:
        if row.get("status") == "ok":
            qkc_by.setdefault(row["arm"], {})[row["seed"]] = float(row["val_auc"])
    observed = [qkc_by["observed"][seed] for seed in SEEDS]
    zero = [qkc_by["zero"][seed] for seed in SEEDS]
    zero_time = [qkc_by["zero_time"][seed] for seed in SEEDS]
    d_incidence = [a - b for a, b in zip(observed, zero)]
    d_t_zero = [a - b for a, b in zip(zero_time, zero)]

    from scipy import stats

    tests = []
    for name, d in (
        ("timing_on_observed_incidence", d_t_observed),
        ("incidence_observed_vs_zero", d_incidence),
        ("timing_on_zero_incidence", d_t_zero),
    ):
        t_greater, p_greater = stats.ttest_1samp(d, 0.0, alternative="greater")
        t_two_sided, p_two_sided = stats.ttest_1samp(d, 0.0)
        ci_low, ci_high = stats.t.interval(
            0.95,
            df=len(d) - 1,
            loc=mean(d),
            scale=stats.sem(d),
        )
        tests.append(
            {
                "name": name,
                "n": len(d),
                "mean_delta_val": float(mean(d)),
                "sd_delta_val": float(stdev(d)),
                "ci95_delta_val": [float(ci_low), float(ci_high)],
                "t": float(t_greater),
                "p_one_sided_positive": float(p_greater),
                "p_two_sided": float(p_two_sided),
            }
        )

    # Holm–Bonferroni across the complete three-test, five-seed credit family.
    ordered = sorted(tests, key=lambda x: x["p_one_sided_positive"])
    running_max = 0.0
    for i, row in enumerate(ordered):
        raw_adjusted = min(
            1.0, row["p_one_sided_positive"] * (len(ordered) - i)
        )
        running_max = max(running_max, raw_adjusted)
        row["p_holm_m3"] = float(running_max)
        row["reject_holm_m3_alpha05"] = bool(running_max < 0.05)

    out = {
        "sources": [
            str(a2_matrix.relative_to(REPO_ROOT)).replace("\\", "/"),
            str(qkc_matrix.relative_to(REPO_ROOT)).replace("\\", "/"),
        ],
        "seeds": SEEDS,
        "alpha": 0.05,
        "tests": ordered,
        "note": (
            "FAIL arms in table_negative_knowledge.tex remain single-seed "
            "and are judged only by the +0.002 gate. The historical pathway-off "
            "comparison is excluded because it is capacity-mismatched."
        ),
    }
    out_path = REPO_ROOT / "results" / "tables" / "b8_holm_credit_tests.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for row in ordered:
        print(
            f"{row['name']}: mean_delta={row['mean_delta_val']:.5f} "
            f"t={row['t']:.2f} p={row['p_one_sided_positive']:.3e} "
            f"holm_m3={row['p_holm_m3']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
