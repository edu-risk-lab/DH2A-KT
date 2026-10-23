#!/usr/bin/env python3
"""Junyi attempt-onset data check (docs/preregistration_round3.md, Section 8.2).

Reads the raw Junyi problem log and reports whether an attempt onset can be
rebuilt as time_done - duration: which duration field exists, its unit, the
share of rows where it is present and non-negative, and the share of rows
whose rebuilt onset precedes the same learner's previous completion. No
training; the onset runs are launched only if the 99% condition holds.

    python scripts/62_junyi_onset_check.py

Writes results/tables/junyi_onset_check.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "external" / "p0_leakage_audit" / "data" / "raw" / "junyi" / "junyi_ProblemLog_original.csv"
OUT_JSON = REPO / "results" / "tables" / "junyi_onset_check.json"
MIN_VALID_SHARE = 0.99


def _first_attempt_seconds(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.split("&").str[0], errors="coerce")


def _sum_attempt_seconds(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split("&", expand=True)
    return parts.apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def main() -> int:
    header = list(pd.read_csv(RAW, nrows=0).columns)
    fields = [c for c in ("time_taken", "time_taken_attempts") if c in header]
    out: dict = {"raw_file": str(RAW.relative_to(REPO)), "columns": header, "duration_fields": fields}
    if not fields:
        out["decision"] = "no duration field; history-only fallback"
        OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0

    df = pd.read_csv(RAW, usecols=["user_id", "time_done", *fields], low_memory=False)
    df = df.dropna(subset=["user_id", "time_done"])
    done_s = (pd.to_numeric(df["time_done"], errors="coerce") // 1_000_000).astype("Int64")
    candidates = {}
    if "time_taken" in fields:
        candidates["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")
    if "time_taken_attempts" in fields:
        candidates["sum_time_taken_attempts"] = _sum_attempt_seconds(df["time_taken_attempts"])
        candidates["first_time_taken_attempts"] = _first_attempt_seconds(df["time_taken_attempts"])

    out["n_rows"] = int(len(df))
    out["fields"] = {}
    order = np.lexsort((done_s.to_numpy(dtype="int64", na_value=0), df["user_id"].astype(str).to_numpy()))
    users = df["user_id"].astype(str).to_numpy()[order]
    done_sorted = done_s.to_numpy(dtype="float64", na_value=np.nan)[order]
    same_user = np.r_[False, users[1:] == users[:-1]]
    for name, dur in candidates.items():
        d = dur.to_numpy(dtype="float64")
        valid = np.isfinite(d) & (d >= 0)
        onset = done_sorted - d[order]
        prev_done = np.r_[np.nan, done_sorted[:-1]]
        overlap = same_user & valid[order] & np.isfinite(prev_done) & (onset < prev_done)
        out["fields"][name] = {
            "valid_share": float(valid.mean()),
            "median_seconds": float(np.nanmedian(d[valid])) if valid.any() else None,
            "p99_seconds": float(np.nanpercentile(d[valid], 99)) if valid.any() else None,
            "onset_before_previous_completion_share": float(overlap.sum() / max(1, (same_user & valid[order]).sum())),
        }
    if "time_taken" in candidates:
        tt = candidates["time_taken"].to_numpy(dtype="float64")
        st = candidates.get("sum_time_taken_attempts")
        if st is not None:
            both = np.isfinite(tt) & np.isfinite(st.to_numpy(dtype="float64"))
            out["time_taken_equals_attempt_sum_share"] = float(
                (np.abs(tt[both] - st.to_numpy(dtype="float64")[both]) < 1).mean()
            )
    chosen = next(
        (n for n in ("time_taken", "sum_time_taken_attempts") if n in out["fields"]
         and out["fields"][n]["valid_share"] >= MIN_VALID_SHARE),
        None,
    )
    out["chosen_field"] = chosen
    out["decision"] = "onset runs (Section 8.2, first branch)" if chosen else "history-only fallback (Section 8.2)"
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
