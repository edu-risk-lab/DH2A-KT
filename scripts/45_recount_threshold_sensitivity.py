#!/usr/bin/env python3
"""Recount Table 6 (threshold sensitivity) from unrounded per-seed Δval.

Hau Sep-15 A1 + B7: every cell is k/5 from unrounded validation Δ, not from
printed five-decimal cells. Also emits the T-zero vs no-time descriptive
contrast (B2) with the same 95% CI recipe as Table 12.

No training. CPU only.

Usage:
    python scripts/45_recount_threshold_sensitivity.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SEEDS = ["0", "17", "42", "1234", "2024"]
THRESHOLDS = (0.0010, 0.0015, 0.0020, 0.0025, 0.0030)
QKC = REPO / "results" / "tables" / "qkc_capacity_matched_matrix.csv"
A1 = REPO / "results" / "tables" / "a1_timing_input_controls.csv"
A7 = REPO / "results" / "tables" / "a7_query_dt_ablation.csv"
OUT_JSON = REPO / "results" / "tables" / "threshold_sensitivity_recount.json"
OUT_TEX = REPO / "paper" / "tables" / "table_threshold_sensitivity.tex"


def _qkc_by_arm() -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, float]] = {}
    with QKC.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            by.setdefault(row["arm"], {})[row["seed"]] = float(row["val_auc"])
    return by


def _a1_val() -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, float]] = {}
    with A1.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("val_auc") or ""
            if not raw:
                continue
            by.setdefault(row["arm"], {})[row["seed"]] = float(raw)
    return by


def _a7_both() -> dict[str, float]:
    out: dict[str, float] = {}
    with A7.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != "dt_both":
                continue
            out[row["seed"]] = float(row["delta_val"])
    return out


def _pass_count(deltas: list[float], threshold: float) -> int:
    return sum(1 for d in deltas if d >= threshold - 1e-15)


def _fmt_k(k: int) -> str:
    return f"${k}/5$"


def _ci95(deltas: list[float]) -> tuple[float, float, float]:
    lo, hi = stats.t.interval(
        0.95, df=len(deltas) - 1, loc=mean(deltas), scale=stats.sem(deltas)
    )
    return float(mean(deltas)), float(lo), float(hi)


def main() -> int:
    qkc = _qkc_by_arm()
    a1 = _a1_val()
    pkg_obs = _a7_both()

    zero = [qkc["zero"][s] for s in SEEDS]
    zero_time = [qkc["zero_time"][s] for s in SEEDS]
    observed = [qkc["observed"][s] for s in SEEDS]
    t_zero = [a1["t_zero"][s] for s in SEEDS]
    t_mis = [a1["t_misaligned"][s] for s in SEEDS]

    d_pkg_zero = [a - b for a, b in zip(zero_time, zero)]
    d_inc = [a - b for a, b in zip(observed, zero)]
    d_tzero = [a - b for a, b in zip(zero_time, t_zero)]
    d_tmis = [a - b for a, b in zip(zero_time, t_mis)]
    d_pkg_obs = [pkg_obs[s] for s in SEEDS]
    d_tz_vs_notime = [a - b for a, b in zip(t_zero, zero)]

    columns = {
        "package_zero_incidence": d_pkg_zero,
        "package_observed_incidence": d_pkg_obs,
        "aligned_vs_t_zero": d_tzero,
        "aligned_vs_t_misaligned": d_tmis,
        "incidence": d_inc,
    }

    cells: dict[str, dict[str, int]] = {}
    for name, deltas in columns.items():
        cells[name] = {f"{t:.4f}": _pass_count(deltas, t) for t in THRESHOLDS}

    tz_mean, tz_lo, tz_hi = _ci95(d_tz_vs_notime)
    payload = {
        "seeds": [int(s) for s in SEEDS],
        "source_files": [str(p.relative_to(REPO)) for p in (QKC, A1, A7)],
        "unrounded_deltas": {
            name: {s: float(d) for s, d in zip(SEEDS, deltas)}
            for name, deltas in columns.items()
        },
        "t_zero_vs_no_time": {
            "name": "extra-map without aligned gap (T-zero minus no-time)",
            "per_seed": {s: float(d) for s, d in zip(SEEDS, d_tz_vs_notime)},
            "mean": tz_mean,
            "ci95": [tz_lo, tz_hi],
            "sd": float(stdev(d_tz_vs_notime)),
            "note": (
                "Descriptive three-point ladder point, not a causal share "
                "and not a capacity effect. Do not convert to a percentage."
            ),
        },
        "cells_k_of_5": cells,
        "mins": {name: float(min(d)) for name, d in columns.items()},
        "window": {
            "aligned_tzero_min": float(min(d_tzero)),
            "package_obs_min": float(min(d_pkg_obs)),
            "package_zero_min": float(min(d_pkg_zero)),
            "qualitative_map_open": float(min(d_tzero)),
            "qualitative_map_closed_upper": float(min(min(d_pkg_obs), min(d_pkg_zero))),
            "note": (
                "Both package columns are 5/5, aligned-gap columns are not 5/5, "
                "and incidence is 0/5, for any threshold in "
                "(aligned_tzero_min, min(package mins)]."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(REPO)}")
    print("cells", json.dumps(cells, indent=2))
    print("window", json.dumps(payload["window"], indent=2))
    print(
        "T-zero vs no-time mean "
        f"{tz_mean:+.5f} 95% CI [{tz_lo:+.5f}, {tz_hi:+.5f}]"
    )
    assert cells["aligned_vs_t_misaligned"]["0.0015"] == 2, cells
    assert cells["package_zero_incidence"]["0.0020"] == 5
    assert cells["package_observed_incidence"]["0.0020"] == 5
    print(
        "TeX caption is hand-maintained in "
        "paper/tables/table_threshold_sensitivity.tex; JSON is the recount."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
