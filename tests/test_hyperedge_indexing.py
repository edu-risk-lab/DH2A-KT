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