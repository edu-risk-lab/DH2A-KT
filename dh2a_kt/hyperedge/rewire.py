"""Structure-preserving hypergraph checks (Hau gold B12).

``degree_preserving_rewire`` keeps node hyper-degree and hyperedge sizes
via Maslov–Sneppen swaps of concept incidences. A drop here is evidence
that the encoder uses *which* concepts co-occur, not merely that it reads
some graph input (node-drop already shows the latter).

``relation_label_permute`` keeps membership and permutes ``kind``. On a
single-kind diagnostic encoder this is expected to be near-null.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dh2a_kt.hyperedge.construction import Hyperedge

STRUCTURE_OPERATORS = frozenset(
    {"degree_preserving_rewire", "relation_label_permute"}
)


def concept_hyperdegrees(hyperedges: list[Hyperedge]) -> dict[int, int]:
    deg: dict[int, int] = {}
    for he in hyperedges:
        for typ, eid in he.members:
            if typ == "concept":
                deg[int(eid)] = deg.get(int(eid), 0) + 1
    return deg


def degree_preserving_rewire_hyperedges(
    hyperedges: list[Hyperedge],
    *,
    seed: int = 42,
    n_swaps: int | None = None,
) -> list[Hyperedge]:
    """Swap concept members across hyperedges; preserve degrees and sizes."""
    if not hyperedges:
        return []
    members = [list(he.members) for he in hyperedges]
    incidences: list[tuple[int, int]] = []
    for i, mem in enumerate(members):
        for j, (typ, _eid) in enumerate(mem):
            if typ == "concept":
                incidences.append((i, j))
    if len(incidences) < 2:
        return [replace(he, members=list(he.members)) for he in hyperedges]

    rng = np.random.default_rng(seed)
    attempts = n_swaps if n_swaps is not None else max(20 * len(incidences), 200)
    for _ in range(attempts):
        a, b = int(rng.integers(0, len(incidences))), int(rng.integers(0, len(incidences)))
        if a == b:
            continue
        hi, si = incidences[a]
        hj, sj = incidences[b]
        if hi == hj:
            continue
        id_i = int(members[hi][si][1])
        id_j = int(members[hj][sj][1])
        if id_i == id_j:
            continue
        concepts_i = {int(eid) for t, eid in members[hi] if t == "concept"}
        concepts_j = {int(eid) for t, eid in members[hj] if t == "concept"}
        if id_j in concepts_i or id_i in concepts_j:
            continue
        members[hi][si] = ("concept", id_j)
        members[hj][sj] = ("concept", id_i)
    return [
        replace(he, members=members[i], hyperedge_id=f"{he.hyperedge_id}_rewire")
        for i, he in enumerate(hyperedges)
    ]


def permute_relation_labels(
    hyperedges: list[Hyperedge],
    *,
    seed: int = 42,
) -> list[Hyperedge]:
    """Keep members; shuffle ``kind`` across hyperedges."""
    if not hyperedges:
        return []
    kinds = [he.kind for he in hyperedges]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(kinds))
    new_kinds = [kinds[int(i)] for i in order]
    return [replace(he, kind=new_kinds[i]) for i, he in enumerate(hyperedges)]
