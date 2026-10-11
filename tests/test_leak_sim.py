import numpy as np
import pandas as pd
import pytest

from dh2a_kt.data import leak_sim


def _frame():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "item_id": [10, 10, 11, 10, 10, 10],
            "timestamp": [1, 5, 6, 2, 9, 5],
            "correct": [1, 0, 1, 0, 1, 1],
        }
    )


def test_crossmodal_leak_uses_only_later_rows_and_clean_only_earlier_fit_rows():
    df = _frame()
    parts = leak_sim.crossmodal_parts(df, fit_users={1, 2})
    g = df.loc[df.user_id.isin({1, 2}), "correct"].mean()
    k = leak_sim.SHRINK
    # Row 0: item 10 at t=1. Later rows of item 10: t=2 (0), t=5 (0, 1), t=9 (1).
    assert parts.loc[0, "n_leak_support"] == 4
    assert parts.loc[0, "leak_value"] == pytest.approx((2 + k * g) / (4 + k))
    assert parts.loc[0, "n_clean_support"] == 0
    # Row 1: item 10 at t=5; the tie at t=5 (user 3) is excluded from both sides.
    assert parts.loc[1, "n_leak_support"] == 1
    assert parts.loc[1, "n_clean_support"] == 2  # t=1 and t=2, both fit users
    assert parts.loc[1, "clean_value"] == pytest.approx((1 + k * g) / (2 + k))
    # Row 5 (user 3, not fit) sees fit rows before t=5 only.
    assert parts.loc[5, "n_clean_support"] == 2


def test_hint_leak_flags_follow_next_response_and_skip_last_rows():
    rng = np.random.default_rng(0)
    n = 20000
    df = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(200), n // 200),
            "item_id": rng.integers(0, 50, n),
            "timestamp": np.tile(np.arange(n // 200), 200),
            "correct": rng.integers(0, 2, n),
        }
    )
    parts = leak_sim.hint_parts(df, fit_users=set(range(200)))
    leak = parts["leak_value"].to_numpy() > 0.5
    last = np.r_[df.user_id.to_numpy()[1:] != df.user_id.to_numpy()[:-1], True]
    assert not leak[last].any()
    y_next = np.r_[df.correct.to_numpy()[1:], 0]
    rate_wrong = leak[(~last) & (y_next == 0)].mean()
    rate_right = leak[(~last) & (y_next == 1)].mean()
    assert rate_wrong == pytest.approx(0.10, abs=0.01)
    assert rate_right == pytest.approx(0.03, abs=0.01)
    assert parts["clean_value"].mean() == pytest.approx(parts.attrs["fit_rate"], abs=0.01)


def test_graded_statistics_are_zero_clean_and_one_leaky():
    df = _frame()
    rows = np.arange(len(df))
    cm = leak_sim.crossmodal_parts(df, fit_users={1, 2})
    assert leak_sim.graded_crossmodal_share(cm, rows, 0.0, 1) == 0.0
    assert leak_sim.graded_crossmodal_share(cm, rows, 1.0, 1) == 1.0


def test_attach_returns_mode_and_aligned_columns():
    df = _frame()
    train, ev = df.iloc[:4], df.iloc[4:]
    tr, ev2, mode = leak_sim.attach_leak_scalar(train, ev, "crossmodal-leak", {1, 2})
    assert mode == "query"
    assert len(tr) == 4 and len(ev2) == 2
    assert "aux_scalar" in tr.columns and "aux_scalar" in ev2.columns
