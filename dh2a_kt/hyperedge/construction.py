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
) -> list[Hyperedge]:
    """Group {Student, Exercise, Hint, Video} co-occurring within one session.

    TODO (Pha 1, docs/idea-D-plan.md): requires interaction logs carrying
    hint_id / video_id columns, which public KT benchmarks studied by P0
    (XES3G5M, ASSISTments2012, Junyi) do not provide natively. Wire this up
    against ES-KT-24 (has video) first; treat Hint as optional per-row column
    until a benchmark with real hint logs is sourced.
    """
    raise NotImplementedError(
        "Session hyperedge construction needs Hint/Video-bearing interaction "
        "logs not present in P0's three core datasets. See docs/idea-D-plan.md "
        "section 5 (Pham vi du lieu) — wire against ES-KT-24 first."
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
