"""Collapse pyKT's KC-level rows back into one event per question attempt.

XES3G5M questions can carry several knowledge components: 939 of 7,121 questions
in fold 0's test split map to 2--6 KCs, and 12.6% of attempts are multi-KC. The
processed corpus expresses that by emitting one row per KC, all sharing
``user_id``, ``item_id``, ``timestamp`` and ``correct``.

Two consequences follow, and this module addresses both:

* Those duplicate rows leak the label. A model reading row *k* of an attempt has
  already been shown the answer in row *k-1*, which is why the clean protocol
  masks them (see :func:`dh2a_kt.train.tier1.next_step_mask`). Collapsing removes
  the leak structurally instead of masking it after the fact.
* A multi-KC question is the one genuinely hypergraph-shaped object in this
  corpus: an event that exercises a *set* of KCs at once. Row expansion destroys
  that set, which is why the concept-level hypergraph had nothing to say.

Collapsing keeps the first row of each attempt, so the surviving positions are
the ones the clean protocol already scores and v5 stays comparable to the clean
baseline table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_KEYS = ["user_id", "item_id", "timestamp"]


def attempt_ids(sorted_df: pd.DataFrame) -> np.ndarray:
    """Group id per row: rows of one question attempt share an id.

    Expects the frame already sorted the way the training dataset sorts it
    (``user_id``, ``timestamp``, ``item_id``), so an attempt's rows are adjacent.
    """
    n = len(sorted_df)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    user = sorted_df["user_id"].to_numpy()
    item = sorted_df["item_id"].to_numpy()
    time = sorted_df["timestamp"].to_numpy()
    is_new = np.ones(n, dtype=bool)
    is_new[1:] = (user[1:] != user[:-1]) | (item[1:] != item[:-1]) | (time[1:] != time[:-1])
    return np.cumsum(is_new) - 1


def collapse_to_events(df: pd.DataFrame, *, max_kcs: int | None = None) -> pd.DataFrame:
    """One row per question attempt, with the attempt's KC set in ``kc_ids``.

    ``kc_ids`` holds the distinct ``kc_id`` values in first-seen order.
    ``max_kcs`` truncates longer sets, which keeps the padded tensor width
    bounded; pass ``None`` to keep every KC.
    """
    if len(df) == 0:
        out = df.head(0).copy()
        out["kc_ids"] = pd.Series(dtype=object)
        out["n_kcs"] = pd.Series(dtype=np.int64)
        return out

    ordered = df.sort_values(EVENT_KEYS).reset_index(drop=True)
    ordered["_attempt"] = attempt_ids(ordered)

    kc_sets = (
        ordered.groupby("_attempt", sort=True)["kc_id"]
        .apply(lambda s: list(dict.fromkeys(s.tolist())))
        .rename("kc_ids")
    )
    if max_kcs is not None:
        kc_sets = kc_sets.map(lambda kcs: kcs[:max_kcs])

    events = ordered.drop_duplicates("_attempt", keep="first").set_index("_attempt")
    events = events.join(kc_sets)
    events["n_kcs"] = events["kc_ids"].map(len).astype(np.int64)
    # kc_id keeps the first KC so callers that expect the single-KC column, such
    # as the concept-level readout, still work on collapsed frames.
    return events.reset_index(drop=True).drop(columns=["_attempt"], errors="ignore")


def question_kc_sets(df: pd.DataFrame) -> dict[int, list[int]]:
    """Map each ``item_id`` to the union of KCs seen for it, in first-seen order.

    The union is the right choice here: across fold 0's test split only 2 of
    7,121 questions ever present different KC sets on different attempts, so a
    per-question set is well defined in practice.
    """
    if len(df) == 0:
        return {}
    ordered = df.sort_values(EVENT_KEYS)
    grouped = ordered.groupby("item_id", sort=True)["kc_id"].apply(
        lambda s: list(dict.fromkeys(s.tolist()))
    )
    return {int(item): [int(kc) for kc in kcs] for item, kcs in grouped.items()}


def pad_kc_matrix(
    kc_lists: list[list[int]],
    *,
    width: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged KC sets into ``(n, width)`` ids plus a boolean mask.

    The mask, not the pad value, tells the model which slots are real: padding
    with a valid concept id would otherwise silently add concept 0 to every
    single-KC event.
    """
    n = len(kc_lists)
    ids = np.full((n, width), pad_value, dtype=np.int64)
    mask = np.zeros((n, width), dtype=bool)
    for i, kcs in enumerate(kc_lists):
        take = kcs[:width]
        if not take:
            continue
        ids[i, : len(take)] = take
        mask[i, : len(take)] = True
    return ids, mask
