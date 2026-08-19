"""Tests for session hyperedge construction (Hint-bearing rows)."""

from __future__ import annotations

import pandas as pd

from dh2a_kt.hyperedge.construction import build_hint_item_hyperedges, build_session_hyperedges


def _toy_interactions() -> pd.DataFrame:
    # Two sessions for user 1 (gap > 1800), one for user 2.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2],
            "item_id": [10, 11, 12, 13, 20],
            "kc_id": [100, 100, 101, 101, 200],
            "timestamp": [0, 100, 5000, 5100, 0],
            "correct": [1, 0, 1, 1, 0],
            "hint_count": [0, 2, 0, 1, 0],
            "hint_used": [0, 1, 0, 1, 0],
        }
    )


def test_session_hyperedges_split_on_gap():
    hes = build_session_hyperedges(_toy_interactions(), fold=0, session_gap_seconds=1800)
    # user1: 2 sessions + user2: 1 session
    assert len(hes) == 3
    assert all(he.kind == "session" for he in hes)
    assert all(he.train_only for he in hes)


def test_session_hyperedges_include_hint_members():
    hes = build_session_hyperedges(_toy_interactions(), fold=0, session_gap_seconds=1800)
    hinted = [he for he in hes if any(t == "hint" for t, _ in he.members)]
    assert len(hinted) == 2
    # First session: hint on item 11
    he0 = next(he for he in hes if he.hyperedge_id.endswith("_u1") and he.provenance["n_interactions"] == 2)
    assert ("hint", 11) in he0.members
    assert ("student", 1) in he0.members
    assert ("exercise", 10) in he0.members
    assert ("concept", 100) in he0.members


def test_session_hyperedges_without_hint_columns():
    df = _toy_interactions().drop(columns=["hint_count", "hint_used"])
    hes = build_session_hyperedges(df, fold=1, session_gap_seconds=1800)
    assert len(hes) == 3
    assert all(not any(t == "hint" for t, _ in he.members) for he in hes)


def test_hint_item_hyperedges_drop_concepts_and_unhinted_sessions():
    hes = build_hint_item_hyperedges(_toy_interactions(), fold=0, session_gap_seconds=1800)
    assert len(hes) == 2
    assert all(he.kind == "session_hint" for he in hes)
    for he in hes:
        types = {t for t, _ in he.members}
        assert types == {"exercise"}
        assert not any(t == "concept" for t, _ in he.members)
        assert he.provenance["n_hint_members"] >= 1
        assert he.provenance["n_items"] >= 2
    item_sets = [{eid for t, eid in he.members} for he in hes]
    assert {10, 11} in item_sets
    assert {12, 13} in item_sets


def test_hint_item_hyperedges_empty_without_hint_columns():
    df = _toy_interactions().drop(columns=["hint_count", "hint_used"])
    hes = build_hint_item_hyperedges(df, fold=0, session_gap_seconds=1800)
    assert hes == []
