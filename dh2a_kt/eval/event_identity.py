"""Stable event keys vs the DH2/pyKT occurrence join (Hau A4).

The occurrence key ``user|dense-kc|occurrence-in-scored-set`` is not a native
attempt ID. The same key can pair two different KC-rows when the two pipelines
filter different rows and the occurrence counters drift. Eight joined rows in
``b6_id_join_summary.json`` disagree on the label; those rows are evidence
against the join, not rows to drop.

The pre-chunk key is ``user|timestamp|item|kc``. pyKT's exported ``timestamps``
column is a local index, not this field: keys must be taken from the interaction
frame before export.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator

import pandas as pd

SORT_COLUMNS = ["user_id", "timestamp", "item_id"]


def stable_event_key(user_id: int, timestamp: int, item_id: int, kc_id: int) -> str:
    return f"{int(user_id)}|{int(timestamp)}|{int(item_id)}|{int(kc_id)}"


def parse_occurrence_key(key: str) -> tuple[int, int, int]:
    user, kc, occ = str(key).split("|")
    return int(user), int(kc), int(occ)


def account_join_counts(
    *,
    n_native_dh2: int,
    n_unmapped_kc: int,
    n_pykt: int,
    n_only_dh2: int,
    n_only_pykt: int,
    n_joined: int,
) -> dict[str, Any]:
    """Close native 1,093,755 against the occurrence join."""
    n_join_eligible = int(n_native_dh2) - int(n_unmapped_kc)
    ok = (
        n_join_eligible == int(n_native_dh2) - int(n_unmapped_kc)
        and n_join_eligible - int(n_only_dh2) == int(n_joined)
        and int(n_pykt) - int(n_only_pykt) == int(n_joined)
    )
    return {
        "n_native_dh2": int(n_native_dh2),
        "n_unmapped_kc": int(n_unmapped_kc),
        "n_join_eligible_dh2": n_join_eligible,
        "n_pykt": int(n_pykt),
        "n_only_dh2": int(n_only_dh2),
        "n_only_pykt": int(n_only_pykt),
        "n_joined": int(n_joined),
        "n_net_dh2_surplus": int(n_only_dh2) - int(n_only_pykt),
        "accounting_ok": bool(ok),
        "one_row_gap_is_unmapped_kc": int(n_unmapped_kc) == 1
        and n_join_eligible == int(n_native_dh2) - 1,
    }


def classify_unmatched_occurrence_keys(keys: list[str]) -> dict[str, Any]:
    """Describe DH2-only occurrence keys without claiming a raw-event trace."""
    parsed = [parse_occurrence_key(k) for k in keys]
    users = {u for u, _, _ in parsed}
    first = [(u, k, o) for u, k, o in parsed if o == 1]
    later = [(u, k, o) for u, k, o in parsed if o > 1]
    per_user: dict[int, int] = defaultdict(int)
    for user, _, _ in parsed:
        per_user[user] += 1
    return {
        "n_keys": len(parsed),
        "n_users": len(users),
        "n_first_occurrence": len(first),
        "n_later_occurrence": len(later),
        "max_rows_per_user": max(per_user.values()) if per_user else 0,
        "users": sorted(users),
        "first_occurrence_pairs": [
            {"user_id": u, "kc_dense": k, "occ": o} for u, k, o in first
        ],
    }


def drop_unmapped_question_kc(
    df: pd.DataFrame,
    q_map: dict[int, int],
    c_map: dict[int, int],
) -> pd.DataFrame:
    """Same filter as ``dh2a_kt.baselines.pykt_clean._build_rows``."""
    mapped = df.copy()
    mapped["qi"] = mapped["item_id"].map(q_map)
    mapped["ci"] = mapped["kc_id"].map(c_map)
    mapped = mapped.dropna(subset=["qi", "ci"])
    if mapped.empty:
        return mapped
    mapped["qi"] = mapped["qi"].astype(int)
    mapped["ci"] = mapped["ci"].astype(int)
    return mapped[(mapped["qi"] >= 0) & (mapped["ci"] >= 0)].copy()


def iter_scored_kc_rows(
    df: pd.DataFrame,
    *,
    max_seq_len: int = 400,
) -> Iterator[dict[str, Any]]:
    """Yield scored KC-rows under chunked L and repeat-target masking.

    Uses the same sort, windowing, and target rule as training. Does not map
    IDs; callers drop unmapped rows first if they need the pyKT export set.
    """
    from dh2a_kt.baselines.pykt_clean import windows
    from dh2a_kt.eval.attempt_safety import scored_target_rows
    from dh2a_kt.train.tier1 import _repeat_flags

    if df.empty:
        return
    sorted_df = df.sort_values(SORT_COLUMNS).reset_index(drop=True)
    flags = _repeat_flags(sorted_df)
    for _, group in sorted_df.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        for start, end in windows(len(idx), max_seq_len, "chunked"):
            local_flags = flags[idx[start:end]]
            length = end - start
            for local_row in scored_target_rows(length, local_flags):
                global_i = int(idx[start + int(local_row)])
                row = sorted_df.iloc[global_i]
                user = int(row["user_id"])
                ts = int(row["timestamp"])
                item = int(row["item_id"])
                kc = int(row["kc_id"])
                yield {
                    "user_id": user,
                    "timestamp": ts,
                    "item_id": item,
                    "kc_id": kc,
                    "label": int(row["correct"]),
                    "stable_key": stable_event_key(user, ts, item, kc),
                }
