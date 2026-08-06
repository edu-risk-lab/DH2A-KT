"""Build heterogeneous hyperedges on top of P0's leakage-controlled, train-only
pairwise KC graph (E_pre / E_sim).

Plan reference: docs/idea-D-plan.md, section 4 (Pha 1 — "Xay hyperedge tren
nen P0"). Node types: Student, Exercise, Concept, Teacher, Hint, Forum,
Discussion, Video (Idea A/C/D architecture).

Only ``build_concept_prerequisite_hyperedges`` has real, ready-to-run public
benchmark data behind it today (via P0's audited E_pre). The other builders
are interface-complete stubs pending the data sources flagged as an
acknowledged gap in every version of this idea (A/B/C/D): no public KT
benchmark ships Teacher/Forum/Discussion synced to interaction logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Hyperedge:
    """A single hyperedge: a set of typed member entities that co-occur.

    ``members`` is a list of ``(entity_type, entity_id)`` pairs, e.g.
    ``[("student", 42), ("exercise", 7), ("hint", 3), ("video", 11)]``.
    ``provenance`` mirrors P0's edge provenance fields (fold id, train-only
    flag, source tag) so the hyperedge audit (see ``audit.py``) can reuse
    P0's leakage taxonomy plus the 3 new classes from Idea D section 2.
    """

    hyperedge_id: str
    kind: str  # "concept_prerequisite" | "session" | "discussion_thread" | "teacher_intervention"
    members: list[tuple[str, int]]
    fold: int
    train_only: bool
    timestamp_range: tuple[float, float] | None = None
    provenance: dict = field(default_factory=dict)


def build_concept_prerequisite_hyperedges(
    e_pre: pd.DataFrame,
    *,
    fold: int,
    min_chain_len: int = 3,
    max_chain_len: int = 8,
) -> list[Hyperedge]:
    """Group P0's already-audited, acyclic ``E_pre`` edges into hyperedges.

    P0's ``E_pre`` (from ``dh2a_kt.p0_bridge.infer_prerequisites_from_train`` +
    ``audit_dag``) is a *pairwise* prerequisite DAG. We group directed chains
    of length >= ``min_chain_len`` into a single hyperedge per chain, so a
    downstream hypergraph encoder can propagate along an entire prerequisite
    sequence in one hop instead of one pairwise edge at a time.

    This function does not re-run leakage auditing itself — it assumes
    ``e_pre`` has already passed P0's ``audit_dag`` (acyclic, train-only).
    Run ``dh2a_kt.hyperedge.audit.audit_hyperedges`` on the *output* of this
    function before using it downstream (Idea D section 2/6).
    """
    graph = nx.DiGraph()
    graph.add_edges_from(e_pre[["src_kc", "dst_kc"]].itertuples(index=False, name=None))

    hyperedges: list[Hyperedge] = []
    seen_chains: set[tuple[int, ...]] = set()
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    # DFS from each DAG root — equivalent to all_simple_paths on an acyclic
    # E_pre graph but avoids O(|V|^2) NetworkX calls on large benchmarks.
    for root in roots:
        stack: list[tuple[object, list[int]]] = [(root, [int(root)])]
        while stack:
            node, path = stack.pop()
            if len(path) >= min_chain_len:
                key = tuple(path)
                if key not in seen_chains:
                    seen_chains.add(key)
                    hyperedges.append(
                        Hyperedge(
                            hyperedge_id=f"cprereq_f{fold}_{'_'.join(map(str, path))}",
                            kind="concept_prerequisite",
                            members=[("concept", kc) for kc in path],
                            fold=fold,
                            train_only=True,
                            provenance={"source": "p0_e_pre_chain", "chain_len": len(path)},
                        )
                    )
            if len(path) >= max_chain_len:
                continue
            for succ in graph.successors(node):
                stack.append((succ, path + [int(succ)]))
    logger.info("Built %d concept-prerequisite hyperedges (fold=%s)", len(hyperedges), fold)
    return hyperedges


def hyperedges_from_e_pre_pairwise(e_pre: pd.DataFrame, *, fold: int = 0) -> list[Hyperedge]:
    """One 2-node hyperedge per audited ``E_pre`` pair — fast graph for M6 checks."""
    return [
        Hyperedge(
            hyperedge_id=f"prepair_f{fold}_{i}",
            kind="concept_prerequisite",
            members=[("concept", int(row.src_kc)), ("concept", int(row.dst_kc))],
            fold=fold,
            train_only=True,
            provenance={"source": "p0_e_pre_pair"},
        )
        for i, row in enumerate(e_pre.itertuples(index=False))
    ]


def resolve_concept_prerequisite_hyperedges(
    e_pre: pd.DataFrame,
    *,
    fold: int = 0,
    source: str = "chain",
    min_chain_len: int = 3,
    max_chain_len: int = 8,
) -> list[Hyperedge]:
    """Build training/eval hyperedge set from audited ``E_pre``."""
    if source == "pairwise":
        return hyperedges_from_e_pre_pairwise(e_pre, fold=fold)
    if source == "chain":
        return build_concept_prerequisite_hyperedges(
            e_pre,
            fold=fold,
            min_chain_len=min_chain_len,
            max_chain_len=max_chain_len,
        )
    raise ValueError(f"Unsupported hyperedge source {source!r}; use 'chain' or 'pairwise'")


def build_session_hyperedges(
    interactions: pd.DataFrame,
    *,
    fold: int,
    session_gap_seconds: float = 1800.0,
    train_only: bool = True,
) -> list[Hyperedge]:
    """Group {Student, Exercise, Concept, Hint?} co-occurring within one session.

    Sessions are contiguous per-``user_id`` runs where successive
    ``timestamp`` gaps are ≤ ``session_gap_seconds``. When ``hint_count`` or
    ``hint_used`` is present and positive on a row, a ``("hint", item_id)``
    member is included (FoundationalASSIST / similar hint-bearing corpora).
    Optional ``video_id`` columns are included the same way when present.

    Requires columns: ``user_id``, ``item_id``, ``kc_id``, ``timestamp``.
    """
    required = {"user_id", "item_id", "kc_id", "timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")

    df = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    has_hint_count = "hint_count" in df.columns
    has_hint_used = "hint_used" in df.columns
    has_video = "video_id" in df.columns

    user = df["user_id"].to_numpy()
    ts = df["timestamp"].to_numpy(dtype=float)
    new_user = np.empty(len(df), dtype=bool)
    new_user[0] = True
    if len(df) > 1:
        new_user[1:] = user[1:] != user[:-1]
    gap_break = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        gap_break[1:] = (ts[1:] - ts[:-1]) > session_gap_seconds
    session_id = np.cumsum(new_user | gap_break)

    hyperedges: list[Hyperedge] = []
    for sid, rows in df.groupby(session_id, sort=False):
        he = _session_hyperedge_from_rows(
            rows,
            fold=fold,
            session_idx=int(sid),
            train_only=train_only,
            has_hint_count=has_hint_count,
            has_hint_used=has_hint_used,
            has_video=has_video,
        )
        if he is not None:
            hyperedges.append(he)

    logger.info(
        "Built %d session hyperedges (fold=%s gap=%ss)",
        len(hyperedges),
        fold,
        session_gap_seconds,
    )
    return hyperedges


def _session_hyperedge_from_rows(
    rows: pd.DataFrame,
    *,
    fold: int,
    session_idx: int,
    train_only: bool,
    has_hint_count: bool,
    has_hint_used: bool,
    has_video: bool,
) -> Hyperedge | None:
    if rows.empty:
        return None
    user_id = int(rows["user_id"].iloc[0])
    members: list[tuple[str, int]] = [("student", user_id)]
    seen: set[tuple[str, int]] = {("student", user_id)}
    n_hints = 0

    for item_id, kc_id in zip(
        rows["item_id"].astype(int).tolist(),
        rows["kc_id"].astype(int).tolist(),
        strict=True,
    ):
        for key in (("exercise", item_id), ("concept", kc_id)):
            if key not in seen:
                seen.add(key)
                members.append(key)

    if has_hint_count:
        hint_mask = rows["hint_count"].fillna(0).astype(int) > 0
    elif has_hint_used:
        hint_mask = rows["hint_used"].fillna(0).astype(int) > 0
    else:
        hint_mask = None
    if hint_mask is not None and hint_mask.any():
        for item_id in rows.loc[hint_mask, "item_id"].astype(int).tolist():
            hint_key = ("hint", int(item_id))
            if hint_key not in seen:
                seen.add(hint_key)
                members.append(hint_key)
                n_hints += 1

    if has_video:
        for vid in rows["video_id"].dropna().tolist():
            video_key = ("video", int(vid))
            if video_key not in seen:
                seen.add(video_key)
                members.append(video_key)

    if len(members) < 2:
        return None
    t0 = float(rows["timestamp"].min())
    t1 = float(rows["timestamp"].max())
    return Hyperedge(
        hyperedge_id=f"session_f{fold}_{session_idx}_u{user_id}",
        kind="session",
        members=members,
        fold=fold,
        train_only=train_only,
        timestamp_range=(t0, t1),
        provenance={
            "source": "session_gap",
            "n_interactions": int(len(rows)),
            "n_hint_members": n_hints,
        },
    )


def build_discussion_thread_hyperedges(forum_posts: pd.DataFrame, *, fold: int) -> list[Hyperedge]:
    """Group {Student, Forum post, Concept, Teacher} into a discussion-thread
    hyperedge. TODO: no public benchmark provides this; Tier 2 case-study
    scope only (docs/idea-D-plan.md section 5)."""
    raise NotImplementedError("No forum data source wired yet — Tier 2 case-study scope only.")


def build_teacher_intervention_hyperedges(teacher_actions: pd.DataFrame, *, fold: int) -> list[Hyperedge]:
    """Group {Teacher, Student, Exercise/Hint} at an intervention timestamp.
    TODO: no public benchmark provides this; Tier 2 case-study scope only."""
    raise NotImplementedError("No teacher-action data source wired yet — Tier 2 case-study scope only.")
