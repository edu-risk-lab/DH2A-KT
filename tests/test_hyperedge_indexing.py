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


def test_hyperedge_index_from_list_shape():
    kc_to_idx = {0: 0, 1: 1, 2: 2, 3: 3}
    index = hyperedge_index_from_list(_chain_hyperedges(), kc_to_idx)
    assert index["concept_prerequisite"].shape[0] == 2
    assert index["concept_prerequisite"].numel() > 0
