#!/usr/bin/env python3
"""Hau Sep-15 A2: split raw Δt=0 rows into three disjoint causes.

CPU only. No training. Needs the processed XES3G5M parquet.

Causes (mutually exclusive, first match wins in this order for reporting
as disjoint counts of rows with raw Δt=0):

1. learner-start: first row of a user (gap defined as 0 by construction)
2. same-attempt KC expansion: continues previous row's (user, item, timestamp)
3. true zero-second gap between distinct attempts of the same learner

Usage:
    python scripts/46_zero_gap_three_way.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids

OUT = REPO / "results" / "tables" / "a2_zero_gap_three_way.json"


def main() -> int:
    _, p0_cfg, _ = load_configs(REPO / "configs" / "xes3g5m.yaml")
    processed = Path(p0_cfg.get("processed_path", ""))
    if not processed.is_absolute():
        processed = REPO / processed
        if not processed.exists():
            processed = REPO / "external" / "p0_leakage_audit" / p0_cfg.get(
                "processed_path", ""
            )
    if not processed.exists():
        OUT.write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": f"processed parquet not found: {processed}",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"SKIP: {processed} missing")
        return 0

    interactions = load_interactions_with_ids(p0_cfg)
    test_df = get_fold_splits(interactions, p0_cfg, 0)["test"]
    df = test_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    n = len(df)
    user = df["user_id"].to_numpy()
    item = df["item_id"].to_numpy()
    ts = df["timestamp"].to_numpy(dtype=np.int64)

    is_start = np.ones(n, dtype=bool)
    is_start[1:] = user[1:] != user[:-1]

    is_repeat = np.zeros(n, dtype=bool)
    if n > 1:
        is_repeat[1:] = (
            (user[1:] == user[:-1])
            & (item[1:] == item[:-1])
            & (ts[1:] == ts[:-1])
        )

    raw_dt = np.zeros(n, dtype=np.int64)
    if n > 1:
        raw_dt[1:] = np.maximum((ts[1:] - ts[:-1]) * (user[1:] == user[:-1]), 0)

    zero = raw_dt == 0
    # Learner-start rows are always raw_dt=0 by construction.
    start_zero = is_start
    repeat_zero = is_repeat & ~is_start
    true_zero = zero & ~is_start & ~is_repeat

    same_user = np.zeros(n, dtype=bool)
    same_user[1:] = user[1:] == user[:-1]
    p0_targets = same_user

    payload = {
        "status": "ok",
        "protocol": {
            "dataset": "xes3g5m",
            "fold": 0,
            "split": "test",
            "sort": "user_id, timestamp, item_id",
            "trained": False,
        },
        "n_test_rows": int(n),
        "n_raw_dt_zero": int(zero.sum()),
        "three_way_all_rows": {
            "learner_start": int(start_zero.sum()),
            "same_attempt_kc_expand": int(repeat_zero.sum()),
            "true_zero_between_attempts": int(true_zero.sum()),
        },
        "three_way_p0_targets": {
            "n_p0_targets": int(p0_targets.sum()),
            "learner_start": int((start_zero & p0_targets).sum()),
            "same_attempt_kc_expand": int((repeat_zero & p0_targets).sum()),
            "true_zero_between_attempts": int((true_zero & p0_targets).sum()),
        },
        "shares_of_p0_targets": {},
        "disjoint_check": {
            "sum_three_way": int(start_zero.sum() + repeat_zero.sum() + true_zero.sum()),
            "n_raw_dt_zero": int(zero.sum()),
            "overlap_start_repeat": int((is_start & is_repeat).sum()),
        },
        "note": (
            "Learner-start rows are never P0 next-step targets (no previous "
            "row). Repeat KC-rows of the same attempt are masked at scoring "
            "but remain in the input sequence, so x=log(1+Δt)=0 still marks "
            "attempt-boundary structure for later scored rows."
        ),
    }
    n_p0 = payload["three_way_p0_targets"]["n_p0_targets"]
    for key in (
        "learner_start",
        "same_attempt_kc_expand",
        "true_zero_between_attempts",
    ):
        payload["shares_of_p0_targets"][key] = (
            payload["three_way_p0_targets"][key] / n_p0 if n_p0 else None
        )
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
