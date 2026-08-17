"""Tests for collapsing KC-level rows back into question attempts (v5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dh2a_kt.data.events import (
    attempt_ids,
    collapse_to_events,
    pad_kc_matrix,
    question_kc_sets,
)


def _multi_kc_logs() -> pd.DataFrame:
    """One learner: a 2-KC question, a 1-KC question, then a 3-KC question.

    Mirrors the corpus layout: the rows of one attempt share user, item,
    timestamp and answer, and differ only in kc_id.
    """
    rows = [
        # attempt 0: item 10 exercises KCs 1 and 2, answered correctly
        (1, 10, 100, 1, 1),
        (1, 10, 100, 2, 1),
        # attempt 1: item 11 exercises KC 3, answered incorrectly
        (1, 11, 200, 3, 0),
        # attempt 2: item 12 exercises KCs 2, 4 and 5, answered correctly
        (1, 12, 300, 2, 1),
        (1, 12, 300, 4, 1),
        (1, 12, 300, 5, 1),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "kc_id", "correct"])


def test_attempt_ids_group_rows_of_the_same_question_attempt():
    df = _multi_kc_logs()
    ids = attempt_ids(df)
    assert ids.tolist() == [0, 0, 1, 2, 2, 2]


def test_attempt_ids_separate_reattempts_of_the_same_question():
    df = pd.DataFrame(
        [
            (1, 10, 100, 1, 1),
            (1, 10, 100, 2, 1),
            (1, 10, 900, 1, 0),  # genuine re-attempt: same item, later time
            (1, 10, 900, 2, 0),
        ],
        columns=["user_id", "item_id", "timestamp", "kc_id", "correct"],
    )
    assert attempt_ids(df).tolist() == [0, 0, 1, 1]


def test_collapse_yields_one_row_per_attempt_carrying_the_kc_set():
    events = collapse_to_events(_multi_kc_logs())
    assert len(events) == 3
    assert events["kc_ids"].tolist() == [[1, 2], [3], [2, 4, 5]]
    assert events["n_kcs"].tolist() == [2, 1, 3]
    # The answer survives once per attempt, not once per KC.
    assert events["correct"].tolist() == [1, 0, 1]
    # kc_id keeps the first KC so single-concept consumers still work.
    assert events["kc_id"].tolist() == [1, 3, 2]


def test_collapse_removes_exactly_the_rows_the_clean_protocol_masks():
    df = _multi_kc_logs()
    n_events = len(collapse_to_events(df))
    # `_repeat_flags` is the clean protocol's definition of a leaking row.
    from dh2a_kt.train.tier1 import _repeat_flags

    sorted_df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    n_kept_by_clean_protocol = int((~_repeat_flags(sorted_df)).sum())
    assert n_events == n_kept_by_clean_protocol


def test_collapse_can_cap_the_kc_set_width():
    events = collapse_to_events(_multi_kc_logs(), max_kcs=2)
    assert events["kc_ids"].tolist() == [[1, 2], [3], [2, 4]]
    assert events["n_kcs"].tolist() == [2, 1, 2]


def test_question_kc_sets_unions_over_attempts():
    df = pd.DataFrame(
        [
            (1, 10, 100, 1, 1),
            (1, 10, 100, 2, 1),
            (2, 10, 400, 1, 0),
            (2, 10, 400, 7, 0),  # same question, extra KC on another learner
        ],
        columns=["user_id", "item_id", "timestamp", "kc_id", "correct"],
    )
    assert question_kc_sets(df) == {10: [1, 2, 7]}


def test_pad_kc_matrix_marks_real_slots_only():
    ids, mask = pad_kc_matrix([[1, 2], [3], []], width=3)
    assert ids.shape == (3, 3)
    np.testing.assert_array_equal(mask, [[True, True, False], [True, False, False], [False] * 3])
    # Padding must not be mistaken for concept 0 being exercised.
    assert ids[1, 1] == 0 and not mask[1, 1]


def test_pad_kc_matrix_truncates_sets_wider_than_the_tensor():
    ids, mask = pad_kc_matrix([[1, 2, 3, 4]], width=2)
    assert ids.tolist() == [[1, 2]]
    assert mask.tolist() == [[True, True]]


def test_collapse_handles_empty_frame():
    empty = _multi_kc_logs().head(0)
    events = collapse_to_events(empty)
    assert len(events) == 0
    assert "kc_ids" in events.columns
