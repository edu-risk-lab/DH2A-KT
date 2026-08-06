"""Manipulation-check-gated destruction test for Tier 1 — reuses P0's DDR
augmentation operators (Section 3.6/4.7) applied to the pairwise projection
of DH2-KT's hyperedges, to verify the model is actually graph-reliant and
not "graph-inert" (P0's term for a backbone whose AUC doesn't move even
under near-total graph destruction — e.g. DGEKT and GKT-on-ASSISTments in
P0's own results). docs/idea-D-plan.md section 11.

If DH2-KT passes this check (AUC drops meaningfully under p=0.90 destruction)
its ablation results are interpretable as evidence of graph benefit, per
P0's own reasoning (Section 4.7): "a backbone whose AUC does not move even
under total destruction is graph-inert, and its DDR-AUC correlation carries
no downstream meaning."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from dh2a_kt.hyperedge.construction import Hyperedge
from dh2a_kt.hyperedge.indexing import destroy_hyperedges, hyperedges_to_pairwise_edges
from dh2a_kt.p0_bridge import apply_node_drop, apply_edge_drop, compute_dag_disruption_rate

logger = logging.getLogger(__name__)

# P0's manipulation-check anchor (Section 4.7): destroy almost the entire graph.
MANIPULATION_CHECK_P = 0.90


@dataclass
class ManipulationCheckResult:
    ddr: float
    auc_clean: float
    auc_destroyed: float
    auc_drop: float
    passes_manipulation_check: bool
    verdict: str


def _hyperedges_to_pairwise_edges(hyperedges: list[Hyperedge]) -> pd.DataFrame:
    return hyperedges_to_pairwise_edges(hyperedges)


def run_manipulation_check(
    hyperedges: list[Hyperedge],
    *,
    eval_auc_fn: Callable[[list[Hyperedge]], float],
    p: float = MANIPULATION_CHECK_P,
    seed: int = 42,
    operator: str = "node_drop",
) -> ManipulationCheckResult:
    """Args:
        hyperedges: DH2-KT's constructed hyperedge set (any kind).
        eval_auc_fn: callback that trains/scores DH2-KT given a hyperedge set
            and returns test AUC. Left as a callback because Tier-1 training
            is Pha 2/3 work (docs/idea-D-plan.md), not something this eval
            module should own.
        p: destruction strength; P0 anchors this check at p=0.90.
        operator: "node_drop" or "edge_drop" (P0's DDR operator families —
            see dh2a_kt.p0_bridge for the other three: attr_mask, subgraph,
            prereq_preserve).
    """
    edges = _hyperedges_to_pairwise_edges(hyperedges)
    auc_clean = eval_auc_fn(hyperedges)

    if operator == "node_drop":
        destroyed_edges = apply_node_drop(edges, p, seed)
    elif operator == "edge_drop":
        destroyed_edges = apply_edge_drop(edges, p, seed)
    else:
        raise ValueError(f"Unsupported operator {operator!r}; use node_drop or edge_drop")

    ddr = compute_dag_disruption_rate(edges, destroyed_edges)

    destroyed_hyperedges = destroy_hyperedges(hyperedges, p=p, seed=seed, operator=operator)
    auc_destroyed = eval_auc_fn(destroyed_hyperedges)
    auc_drop = auc_clean - auc_destroyed

    # P0's own threshold framing (Section 4.7): a null shift at p=0.90 means
    # graph-inert, not "leakage-free" or "good". We don't hardcode a numeric
    # pass/fail bar here (P0 doesn't either — it's read qualitatively against
    # the null-shift benchmarks in P0 Table 7/10, |delta AUC| <= 0.003 there);
    # report auc_drop and let the paper's own ablation table interpret it.
    passes = auc_drop > 0.003
    verdict = (
        f"AUC drop {auc_drop:.4f} at p={p} ({operator}). "
        + ("DH2-KT reads the hypergraph (graph-reliant)." if passes
           else "AUC barely moved — check whether DH2-KT is graph-inert before "
                "interpreting any ablation result as evidence of graph benefit "
                "(see P0 Section 4.7).")
    )
    return ManipulationCheckResult(
        ddr=ddr, auc_clean=auc_clean, auc_destroyed=auc_destroyed, auc_drop=auc_drop,
        passes_manipulation_check=passes, verdict=verdict,
    )
