#!/usr/bin/env python3
"""Junyi onset input-availability check (docs/preregistration_round3c.md).

The round-3b onset gap is time_done(t) - time_taken(t) minus the same for
row t-1. It is admissible only if that is the real start of item t. This
script computes the Section 2 quantities on learner fold 0, training split,
with the gap functions the runs use, and applies the L1/L2 rule of
Section 3. No training; CPU only. Reads the processed corpus and the raw
log, writes results/tables/junyi_onset_leak_check.json only.

    python scripts/63_junyi_onset_leak_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dh2a_kt.data.aux_signals import (  # noqa: E402
    _stable_int64,
    attach_junyi_gap_timestamps,
    junyi_raw_durations,
)
from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
    p0_path,
)

CONFIG = REPO / "configs" / "junyi.yaml"
OUT_JSON = REPO / "results" / "tables" / "junyi_onset_leak_check.json"
KEY = ["user_id", "item_id", "timestamp"]
L2_MARGIN = 0.10
CHUNK = 1_000_000


def _auc(score: np.ndarray, label: np.ndarray) -> float | None:
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    pos = label == 1
    n_pos = int(pos.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _hash_column(values: pd.Series) -> np.ndarray:
    text = values.astype("string").str.strip()
    lookup = {v: _stable_int64(v) for v in text.dropna().unique()}
    return text.map(lookup).astype("int64").to_numpy()


def _raw_extras(raw_csv: Path) -> pd.DataFrame:
    """count_attempts, raw correct, and first attempt seconds per run key."""
    parts = []
    cols = ["user_id", "exercise", "time_done", "correct", "count_attempts", "time_taken_attempts"]
    for chunk in pd.read_csv(raw_csv, usecols=cols, chunksize=CHUNK, low_memory=False):
        chunk = chunk.dropna(subset=["user_id", "exercise", "time_done"])
        first = pd.to_numeric(
            chunk["time_taken_attempts"].astype("string").str.split("&").str[0], errors="coerce"
        )
        correct = chunk["correct"].astype("string").str.strip().str.lower().map(
            {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
        )
        parts.append(
            pd.DataFrame(
                {
                    "user_id": _hash_column(chunk["user_id"]),
                    "item_id": _hash_column(chunk["exercise"]),
                    "timestamp": pd.to_numeric(chunk["time_done"], errors="coerce").astype("int64").to_numpy(),
                    "raw_correct": correct.astype("float64").to_numpy(),
                    "count_attempts": pd.to_numeric(chunk["count_attempts"], errors="coerce").to_numpy(),
                    "first_sec": first.where(first >= 0).to_numpy(),
                }
            )
        )
    extras = pd.concat(parts, ignore_index=True)
    return extras.groupby(KEY, sort=False).first().reset_index()


def _gap(gap_ts: np.ndarray, same_user: np.ndarray) -> np.ndarray:
    raw = np.zeros(len(gap_ts), dtype=np.float64)
    raw[1:] = np.maximum((gap_ts[1:] - gap_ts[:-1]).astype(np.float64), 0.0)
    raw[~same_user] = 0.0
    return raw


def _share(mask: np.ndarray, within: np.ndarray) -> float | None:
    n = int(within.sum())
    return float((mask & within).sum() / n) if n else None


def _rate(y: np.ndarray, within: np.ndarray) -> float | None:
    n = int(within.sum())
    return float(y[within].mean()) if n else None


def _diag(a: float | None) -> float | None:
    return None if a is None else abs(a - 0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    _, p0_cfg, _ = load_configs(CONFIG)
    raw_csv = p0_path(p0_cfg, "raw_interactions_file")
    splits = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, args.fold)
    train = (
        splits["train"]
        .sort_values(["user_id", "timestamp", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    del splits
    print(f"[fold {args.fold}] train rows={len(train)}", flush=True)

    durations = junyi_raw_durations(raw_csv, "time_taken")
    onset_ts = attach_junyi_gap_timestamps(train, "onset_s", durations)["gap_timestamp"].to_numpy()
    done_ts = attach_junyi_gap_timestamps(train, "done_s")["gap_timestamp"].to_numpy()
    dur = train[KEY].merge(durations, on=KEY, how="left")["duration_sec"].to_numpy(dtype=np.float64)
    del durations

    raw_extras = _raw_extras(raw_csv)
    extras = train[KEY].merge(raw_extras, on=KEY, how="left")
    first_table = raw_extras[KEY].assign(duration_sec=raw_extras["first_sec"]).dropna(subset=["duration_sec"])
    del raw_extras
    first_ts = attach_junyi_gap_timestamps(train, "onset_s", first_table)["gap_timestamp"].to_numpy()
    print("[check] gaps built", flush=True)

    users = train["user_id"].to_numpy()
    items = train["item_id"].to_numpy()
    ts = train["timestamp"].to_numpy(dtype=np.int64)
    y = train["correct"].to_numpy().astype(np.int64)
    same_user = np.zeros(len(train), dtype=bool)
    same_user[1:] = users[1:] == users[:-1]
    repeat = np.zeros(len(train), dtype=bool)
    repeat[1:] = same_user[1:] & (items[1:] == items[:-1]) & (ts[1:] == ts[:-1])
    scored = same_user & ~repeat

    g_done = _gap(done_ts, same_user)
    g_onset = _gap(onset_ts, same_user)
    g_first = _gap(first_ts, same_user)
    prev_dur = np.r_[np.nan, dur[:-1]]
    prev_dur[~same_user] = np.nan
    attempts = extras["count_attempts"].to_numpy(dtype=np.float64)
    retried = attempts > 1
    single = attempts == 1
    raw_correct = extras["raw_correct"].to_numpy(dtype=np.float64)

    valid = np.isfinite(dur)
    prev_done = np.r_[done_ts[0], done_ts[:-1]]
    overlap = scored & valid & (onset_ts < prev_done)
    zero_onset = scored & (g_onset == 0)

    ys = y[scored]
    auc = {
        "completion_gap": _auc(np.log1p(g_done[scored]), ys),
        "onset_gap_time_taken": _auc(np.log1p(g_onset[scored]), ys),
        "onset_gap_first_attempt": _auc(np.log1p(g_first[scored]), ys),
        "time_taken_t_unavailable": _auc(dur[scored], ys),
        "retried_t_unavailable": _auc(retried[scored].astype(np.float64), ys),
        "time_taken_t_minus_1_available": _auc(prev_dur[scored], ys),
        "onset_gap_time_taken_non_overlap_rows": _auc(
            np.log1p(g_onset[scored & ~overlap]), y[scored & ~overlap]
        ),
    }
    s_retried = scored & retried
    s_single = scored & single
    overlap_retried = _share(overlap, s_retried)
    overlap_single = _share(overlap, s_single)
    both_raw = scored & np.isfinite(raw_correct)

    d_onset, d_done = _diag(auc["onset_gap_time_taken"]), _diag(auc["completion_gap"])
    l1 = bool(d_onset is not None and d_done is not None and d_onset > d_done)
    l2 = bool(
        overlap_retried is not None
        and overlap_single is not None
        and overlap_retried - overlap_single > L2_MARGIN
    )
    out = {
        "prereg": "docs/preregistration_round3c.md",
        "fold": args.fold,
        "split": "train",
        "n_rows": int(len(train)),
        "n_scored": int(scored.sum()),
        "valid_duration_share_scored": float(valid[scored].mean()),
        "count_attempts_matched_share_scored": float(np.isfinite(attempts[scored]).mean()),
        "processed_vs_raw_correct_agreement": (
            float((y[both_raw] == raw_correct[both_raw]).mean()) if both_raw.any() else None
        ),
        "auc": auc,
        "overlap_share": _share(overlap, scored),
        "overlap_share_retried": overlap_retried,
        "overlap_share_single_attempt": overlap_single,
        "correct_rate_overlap": _rate(y, overlap),
        "correct_rate_non_overlap": _rate(y, scored & ~overlap),
        "zero_onset_gap_share_retried": _share(zero_onset, s_retried),
        "zero_onset_gap_share_single_attempt": _share(zero_onset, s_single),
        "retried_share_scored": _share(retried, scored),
        "correct_by_retried": {
            "retried_correct": int((s_retried & (y == 1)).sum()),
            "retried_wrong": int((s_retried & (y == 0)).sum()),
            "single_correct": int((s_single & (y == 1)).sum()),
            "single_wrong": int((s_single & (y == 0)).sum()),
        },
        "rule": {
            "L1": l1,
            "L1_detail": {"abs_onset_minus_half": d_onset, "abs_completion_minus_half": d_done},
            "L2": l2,
            "L2_margin": L2_MARGIN,
        },
        "decision": (
            "onset input unavailable at prediction time: family D (round 3c, Section 4)"
            if (l1 or l2)
            else "onset input admitted: family C unchanged (round 3c, Section 3)"
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
