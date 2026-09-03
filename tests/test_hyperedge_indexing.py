"""Tests for hyperedge indexing and destruction helpers."""

from __future__ import annotations

from dh2a_kt.hyperedge.construction import Hyperedge
from dh2a_kt.hyperedge.indexing import destroy_hyperedges, hyperedge_index_from_list

pytest = __import__("pytest")
torch = pytest.importorskip("torch")


def _chain_hyperedges() -> list[Hyperedge]:
    return [
        Hyperedge(
            hyperedge_id="h0",
            kind="concept_prerequisite",
            fold=0,
            train_only=True,
            members=[("concept", 0), ("concept", 1), ("concept", 2)],
        ),
        Hyperedge(
            hyperedge_id="h1",
            kind="concept_prerequisite",
            fold=0,
            train_only=True,
            members=[("concept", 2), ("concept", 3)],
        ),
    ]


def test_destroy_hyperedges_reduces_set():
    destroyed = destroy_hyperedges(_chain_hyperedges(), p=0.9, seed=42, operator="node_drop")
    assert len(destroyed) < len(_chain_hyperedges())


def test_degree_preserving_rewire_keeps_degrees_and_sizes():
    from dh2a_kt.hyperedge.rewire import (
        concept_hyperdegrees,
        degree_preserving_rewire_hyperedges,
    )

    hes = [
        Hyperedge(
            hyperedge_id=f"h{i}",
            kind="concept_prerequisite",
            fold=0,
            train_only=True,
            members=[("concept", i), ("concept", i + 1), ("concept", i + 2)],
        )
        for i in range(8)
    ]
    rewired = degree_preserving_rewire_hyperedges(hes, seed=0, n_swaps=400)
    assert concept_hyperdegrees(rewired) == concept_hyperdegrees(hes)
    assert [len(he.members) for he in rewired] == [len(he.members) for he in hes]
    orig = {frozenset(m[1] for m in he.members) for he in hes}
    new = {frozenset(m[1] for m in he.members) for he in rewired}
    assert orig != new


def test_relation_label_permute_keeps_members():
    from dh2a_kt.hyperedge.rewire import permute_relation_labels

    hes = [
        Hyperedge(
            hyperedge_id="a",
            kind="concept_prerequisite",
            fold=0,
            train_only=True,
            members=[("concept", 0), ("concept", 1)],
        ),
        Hyperedge(
            hyperedge_id="b",
            kind="session",
            fold=0,
            train_only=True,
            members=[("concept", 2), ("concept", 3)],
        ),
    ]
    out = permute_relation_labels(hes, seed=1)
    assert [he.members for he in out] == [he.members for he in hes]
    assert {he.kind for he in out} == {he.kind for he in hes}


def test_destroy_hyperedges_structure_ops():
    hes = _chain_hyperedges()
    rewired = destroy_hyperedges(hes, p=0.9, seed=0, operator="degree_preserving_rewire")
    labeled = destroy_hyperedges(hes, p=0.9, seed=0, operator="relation_label_permute")
    assert len(rewired) == len(hes)
    assert len(labeled) == len(hes)
    assert [he.kind for he in labeled] == [he.kind for he in hes]


def test_hyperedge_index_preserves_kinds():
    hes = _chain_hyperedges() + [
        Hyperedge(
            hyperedge_id="s0",
            kind="session",
            fold=0,
            train_only=True,
            members=[("student", 9), ("concept", 0), ("concept", 1), ("hint", 10)],
        )
    ]
    kc_to_idx = {0: 0, 1: 1, 2: 2, 3: 3}
    index = hyperedge_index_from_list(hes, kc_to_idx)
    assert "concept_prerequisite" in index and "session" in index
    assert index["session"].shape[0] == 2
    assert index["session"].numel() > 0


def test_select_session_hyperedges_prefers_hint():
    from dh2a_kt.hyperedge.indexing import select_session_hyperedges_for_training

    hes = [
        Hyperedge(
            hyperedge_id="s0",
            kind="session",
            fold=0,
            train_only=True,
            members=[("concept", 0), ("concept", 1)],
        ),
        Hyperedge(
            hyperedge_id="s1",
            kind="session",
            fold=0,
            train_only=True,
            members=[("concept", 0), ("concept", 2), ("hint", 5)],
        ),
        Hyperedge(
            hyperedge_id="s2",
            kind="session",
            fold=0,
            train_only=True,
            members=[("concept", 0)],  # dropped: <2 concepts
        ),
    ]
    selected = select_session_hyperedges_for_training(hes, max_hyperedges=10)
    assert len(selected) == 2
    assert selected[0].hyperedge_id == "s1"


def test_item_hyperedge_index_uses_exercise_members_only():
    from dh2a_kt.hyperedge.indexing import item_hyperedge_index_from_list

    hes = [
        Hyperedge(
            hyperedge_id="h0",
            kind="session_hint",
            fold=0,
            train_only=True,
            members=[("exercise", 10), ("exercise", 11), ("concept", 100), ("hint", 10)],
        )
    ]
    item_to_idx = {10: 0, 11: 1, 12: 2}
    index = item_hyperedge_index_from_list(hes, item_to_idx)
    assert index.shape == (2, 2)
    assert set(index[0].tolist()) == {0, 1}
    assert set(index[1].tolist()) == {0}


def test_concept_index_ignores_session_hint_exercises():
    hes = [
        Hyperedge(
            hyperedge_id="h0",
            kind="session_hint",
            fold=0,
            train_only=True,
            members=[("exercise", 10), ("exercise", 11)],
        )
    ]
    index = hyperedge_index_from_list(hes, {10: 0, 11: 1})
    assert index["session_hint"].numel() == 0