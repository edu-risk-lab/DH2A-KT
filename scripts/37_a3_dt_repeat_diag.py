#!/usr/bin/env python3
"""Cheap A3 diagnosis: Δt at pyKT repeat rows on XES3G5M fold-0 test.

No training. No checkpoint. No attempt-collapse. CPU only.

Repeat rows share user/item/timestamp with the previous row, so wall-clock
Δt is exactly 0 and log1p(Δt) is 0 by construction. This script *measures*
that fact on the live export and counts how many P0-protocol scored
targets are those rows — the same contrast that produced residual +0.0101.

Usage:

    python scripts/37_a3_dt_repeat_diag.py
    python scripts/37_a3_dt_repeat_diag.py --fold 0

Do not start attempt-collapse from this script. If
``repeat_share_of_p0_targets`` is large and ``frac_repeat_log1p_zero`` is
~1, write that into the log and stop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.data.aux_signals import log_time_gaps
from dh2a_kt.hyperedge.p0_inputs import (
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)
from dh2a_kt.train.tier1 import _repeat_flags

OUT = REPO_ROOT / "results" / "tables" / "a3_dt_repeat_dist.json"
LOG = REPO_ROOT / "results" / "tables" / "a3_dt_repeat_diag.log"


def _log(msg: str) -> None:
    print(msg, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(msg.rstrip() + "\n")


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = (0, 25, 50, 75, 90, 99, 100)
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "xes3g5m.yaml")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    _, p0_cfg, _ = load_configs(args.config)
    interactions = load_interactions_with_ids(p0_cfg)
    test_df = get_fold_splits(interactions, p0_cfg, args.fold)["test"]
    sorted_df = test_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    user_ids = sorted_df["user_id"].to_numpy()
    timestamps = sorted_df["timestamp"].to_numpy(dtype=np.int64)
    is_repeat = _repeat_flags(sorted_df)
    log1p = log_time_gaps(timestamps, user_ids)
    raw_dt = np.zeros(len(sorted_df), dtype=np.int64)
    if len(sorted_df) > 1:
        raw_dt[1:] = np.maximum(
            (timestamps[1:] - timestamps[:-1]) * (user_ids[1:] == user_ids[:-1]),
            0,
        )

    n = int(len(sorted_df))
    n_repeat = int(is_repeat.sum())
    repeat_log = log1p[is_repeat]
    fresh_log = log1p[~is_repeat]
    frac_zero = (
        float(np.mean(np.isclose(repeat_log, 0.0))) if n_repeat else float("nan")
    )
    frac_raw_zero = float(np.mean(raw_dt[is_repeat] == 0)) if n_repeat else float("nan")

    # Next-step targets: row i predicts row i (the label at t+1 lives at i).
    # A scored P0 target is every row after a sequence start (i>=1, same user).
    same_user = np.zeros(n, dtype=bool)
    same_user[1:] = user_ids[1:] == user_ids[:-1]
    p0_targets = same_user
    clean_targets = same_user & ~is_repeat
    n_p0 = int(p0_targets.sum())
    n_clean = int(clean_targets.sum())
    n_p0_repeat = int((p0_targets & is_repeat).sum())

    payload = {
        "protocol": {
            "dataset": "xes3g5m",
            "fold": args.fold,
            "split": "test",
            "sort": "user_id, timestamp, item_id",
            "trained": False,
            "attempt_collapse": False,
        },
        "counts": {
            "n_test_rows": n,
            "n_repeat_rows": n_repeat,
            "repeat_share_of_rows": n_repeat / n if n else None,
            "n_p0_targets": n_p0,
            "n_clean_targets": n_clean,
            "n_p0_targets_that_are_repeats": n_p0_repeat,
            "repeat_share_of_p0_targets": n_p0_repeat / n_p0 if n_p0 else None,
            "n_dropped_by_mask_repeats": n_p0 - n_clean,
        },
        "delta_t": {
            "frac_repeat_raw_dt_zero": frac_raw_zero,
            "frac_repeat_log1p_zero": frac_zero,
            "repeat_log1p": _percentiles(repeat_log),
            "fresh_log1p": _percentiles(fresh_log),
            "note": (
                "Repeat rows share timestamp with the previous row, so raw Δt "
                "and log1p(Δt) are 0 by construction. Query-side Δt at a "
                "repeat target is therefore a zero injection into the time branch."
            ),
        },
        "decision": {
            "run_attempt_collapse": False,
            "reason": (
                "Cheap diagnosis only. Headline stays mask-target. "
                "Do not train a collapsed export unless the author authorizes P1."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"n_test_rows={n} n_repeat={n_repeat} frac_repeat_log1p_zero={frac_zero}")
    _log(
        f"P0 targets={n_p0} clean={n_clean} "
        f"repeat_share_of_p0={payload['counts']['repeat_share_of_p0_targets']}"
    )
    _log(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    _log("STOP: do not start attempt-collapse from this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
