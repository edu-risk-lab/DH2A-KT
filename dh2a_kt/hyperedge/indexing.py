"""Convert hyperedge lists to PyG incidence tensors and apply P0-style destruction."""

from __future__ import annotations

import pandas as pd

from dh2a_kt.hyperedge.construction import Hyperedge
from dh2a_kt.p0_bridge import apply_edge_drop, apply_node_drop

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def hyperedges_to_pairwise_edges(hyperedges: list[Hyperedge]) -> pd.DataFrame:
    rows = []
    for he in hyperedges:
        ids = [m[1] for m in he.members]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                rows.append({"src_kc": ids[i], "dst_kc": ids[j], "weight": 1.0})
    return pd.DataFrame(rows, columns=["src_kc", "dst_kc", "weight"])


def filter_hyperedges_by_pairwise_survivors(
    hyperedges: list[Hyperedge],
    surviving_edges: pd.DataFrame,
) -> list[Hyperedge]:
    if surviving_edges.empty:
        return []
    kept_ids = set(zip(surviving_edges["src_kc"], surviving_edges["dst_kc"]))
    return [
        he
        for he in hyperedges
        if any(
            (he.members[i][1], he.members[j][1]) in kept_ids
            or (he.members[j][1], he.members[i][1]) in kept_ids
            for i in range(len(he.members))
            for j in range(i + 1, len(he.members))
        )
    ]


def destroy_hyperedges(
    hyperedges: list[Hyperedge],
    *,
    p: float,
    seed: int,
    operator: str = "node_drop",
) -> list[Hyperedge]:
    """Return hyperedges that still have at least one surviving pairwise link."""
    edges = hyperedges_to_pairwise_edges(hyperedges)
    if edges.empty:
        return []
    if operator == "node_drop":
        surviving = apply_node_drop(edges, p, seed)
    elif operator == "edge_drop":
        surviving = apply_edge_drop(edges, p, seed)
    else:
        raise ValueError(f"Unsupported operator {operator!r}; use node_drop or edge_drop")
    return filter_hyperedges_by_pairwise_survivors(hyperedges, surviving)


def hyperedge_index_from_list(
    hyperedges: list[Hyperedge],
    kc_to_idx: dict[int, int],
) -> dict[str, torch.Tensor]:
    if not _TORCH_AVAILABLE:
        raise ImportError("hyperedge_index_from_list requires PyTorch")

    by_kind: dict[str, list[Hyperedge]] = {}
    for he in hyperedges:
        by_kind.setdefault(he.kind, []).append(he)

    out: dict[str, torch.Tensor] = {}
    for kind, hes in by_kind.items():
        node_rows: list[int] = []
        edge_rows: list[int] = []
        for he_id, he in enumerate(hes):
            for member_type, entity_id in he.members:
                if member_type != "concept":
                    continue
                idx = kc_to_idx.get(int(entity_id))
                if idx is None:
                    continue
                node_rows.append(idx)
                edge_rows.append(he_id)
        if not node_rows:
            out[kind] = torch.empty((2, 0), dtype=torch.long)
        else:
            out[kind] = torch.tensor([node_rows, edge_rows], dtype=torch.long)
    if not out:
        return {"concept_prerequisite": torch.empty((2, 0), dtype=torch.long)}
    return out


def select_session_hyperedges_for_training(
    session_hyperedges: list[Hyperedge],
    *,
    min_concepts: int = 2,
    prefer_hint: bool = True,
    max_hyperedges: int | None = 20_000,
    seed: int = 42,
) -> list[Hyperedge]:
    """Keep multi-concept session HEs (concept projection) for Tier-1 training."""
    import numpy as np

    eligible: list[Hyperedge] = []
    for he in session_hyperedges:
        n_concepts = sum(1 for t, _ in he.members if t == "concept")
        if n_concepts >= min_concepts:
            eligible.append(he)
    if prefer_hint:
        eligible.sort(
            key=lambda he: (
                0 if any(t == "hint" for t, _ in he.members) else 1,
                -sum(1 for t, _ in he.members if t == "concept"),
            )
        )
    if max_hyperedges is not None and len(eligible) > max_hyperedges:
        rng = np.random.default_rng(seed)
        hinted = [he for he in eligible if any(t == "hint" for t, _ in he.members)]
        hinted_ids = {he.hyperedge_id for he in hinted}
        plain = [he for he in eligible if he.hyperedge_id not in hinted_ids]
        if len(hinted) >= max_hyperedges:
            idx = rng.choice(len(hinted), size=max_hyperedges, replace=False)
            eligible = [hinted[i] for i in idx]
        else:
            n_plain = max_hyperedges - len(hinted)
            if len(plain) > n_plain:
                idx = rng.choice(len(plain), size=n_plain, replace=False)
                plain = [plain[i] for i in idx]
            eligible = hinted + plain
    return eligible


def destroyed_hyperedge_index(
    hyperedges: list[Hyperedge],
    kc_to_idx: dict[int, int],
    *,
    p: float,
    seed: int,
    device: torch.device,
    operator: str = "node_drop",
) -> dict[str, torch.Tensor]:
    destroyed = destroy_hyperedges(hyperedges, p=p, seed=seed, operator=operator)
    index = hyperedge_index_from_list(destroyed, kc_to_idx)
    return {kind: tensor.to(device) for kind, tensor in index.items()}
