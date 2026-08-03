"""Hyperedge-level leakage audit — extends P0's 4 pairwise-edge diagnostics
(ECR_flag, ECR_overlap, |rho|, TBMR; P0 Section 3.2 / Algorithm 1) to
heterogeneous hyperedges, and adds 3 leakage classes P0's taxonomy (Table 1)
does not need because it only has one node type (KC).

This module is Idea D's own methodological contribution (docs/idea-D-plan.md
section 2 "Hyperedge-level Leakage Taxonomy") — it is new code, not a
straight re-export from P0 like ``p0_bridge.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from dh2a_kt.hyperedge.construction import Hyperedge
from dh2a_kt.p0_bridge import compute_ecr_flag, compute_rho_edge_outcome, compute_tbvr

logger = logging.getLogger(__name__)


@dataclass
class HyperedgeLeakageReport:
    hyperedge_kind: str
    n_hyperedges: int
    # Reused from P0 (computed on the flattened pairwise projection of the hyperedge set)
    ecr_flag: float
    tbmr: float
    rho: float | None
    # New in Idea D (section 2) — hyperedge-specific leakage classes
    group_membership_leak_rate: float
    cross_modal_leak_rate: float | None
    intervention_timing_leak_rate: float | None
    notes: list[str]


def _flatten_to_pairwise(hyperedges: list[Hyperedge]) -> pd.DataFrame:
    """Project each hyperedge onto all pairwise member combinations so P0's
    pairwise diagnostics (designed for E_pre-style edge lists) can be reused
    without modification. This is a legitimate approximation, not identical
    to a true hyperedge-aware statistic — see notes in the returned report."""
    rows = []
    for he in hyperedges:
        members = he.members
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                rows.append({
                    "src_kc": members[i][1],
                    "dst_kc": members[j][1],
                    "hyperedge_id": he.hyperedge_id,
                    "fold": he.fold,
                })
    return pd.DataFrame(rows, columns=["src_kc", "dst_kc", "hyperedge_id", "fold"])


def compute_group_membership_leakage(
    hyperedges: list[Hyperedge],
    held_out_interaction_ids: set[int],
    member_to_interaction_id: dict[tuple[str, int], set[int]],
) -> float:
    """New leakage class (Idea D section 2): fraction of hyperedges where at
    least one member's supporting interaction(s) belong to the held-out split,
    even though other members are train-only.

    P0 cannot detect this because P0 only ever groups two KC nodes at a time
    (ECR_overlap is defined per pairwise edge); a hyperedge groups >=2
    *heterogeneous* entities, so leakage can enter through any one member
    without showing up in a pairwise check.
    """
    if not hyperedges:
        return 0.0
    tainted = 0
    for he in hyperedges:
        for member in he.members:
            iids = member_to_interaction_id.get(member, set())
            if iids & held_out_interaction_ids:
                tainted += 1
                break
    return tainted / len(hyperedges)


def compute_cross_modal_leakage(
    hyperedges: list[Hyperedge],
    embedding_computed_at: dict[str, float],
    prediction_cutoff_at: dict[str, float],
) -> float | None:
    """New leakage class (Idea D section 2): a Forum/Video embedding computed
    using content timestamped *after* the prediction cutoff for that
    hyperedge's learner (e.g. a thread embedding pooled over future replies).

    ``embedding_computed_at``/``prediction_cutoff_at`` are keyed by
    hyperedge_id; both must be supplied by the embedding pipeline — this
    function only compares timestamps, it does not compute embeddings.
    """
    relevant = [he for he in hyperedges if he.hyperedge_id in embedding_computed_at]
    if not relevant:
        return None
    leaked = sum(
        1 for he in relevant
        if embedding_computed_at[he.hyperedge_id] > prediction_cutoff_at.get(he.hyperedge_id, float("inf"))
    )
    return leaked / len(relevant)


def compute_intervention_timing_leakage(
    teacher_hyperedges: list[Hyperedge],
    intervention_chosen_at: dict[str, float],
    outcome_known_at: dict[str, float],
) -> float | None:
    """New leakage class (Idea D section 2): a Teacher-intervention hyperedge
    whose recorded hint/action content was effectively chosen using knowledge
    of the student's *later* correctness (label leaking into a feature)."""
    relevant = [he for he in teacher_hyperedges if he.hyperedge_id in intervention_chosen_at]
    if not relevant:
        return None
    leaked = sum(
        1 for he in relevant
        if intervention_chosen_at[he.hyperedge_id] >= outcome_known_at.get(he.hyperedge_id, float("inf"))
    )
    return leaked / len(relevant)


def audit_hyperedges(
    hyperedges: list[Hyperedge],
    *,
    splits: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    held_out_interaction_ids: set[int] | None = None,
    member_to_interaction_id: dict | None = None,
) -> HyperedgeLeakageReport:
    """Run the full hyperedge audit: P0's 4 diagnostics (reused via
    ``p0_bridge``, computed on the pairwise projection) + the 3 new classes
    from Idea D section 2, where inputs are available.

    Any check whose required inputs are not supplied returns ``None`` rather
    than a silently wrong 0.0 — callers must treat ``None`` as "not audited",
    not "passed".
    """
    notes: list[str] = []
    if not hyperedges:
        return HyperedgeLeakageReport(
            hyperedge_kind="none", n_hyperedges=0, ecr_flag=0.0, tbmr=0.0, rho=None,
            group_membership_leak_rate=0.0, cross_modal_leak_rate=None,
            intervention_timing_leak_rate=None, notes=["empty hyperedge set"],
        )

    kind = hyperedges[0].kind
    pairwise = _flatten_to_pairwise(hyperedges)
    notes.append(
        "ECR_flag/TBMR/|rho| computed on the pairwise projection of the hyperedge set "
        "(P0's diagnostics are pairwise-native, and compute_rho_edge_outcome expects a "
        "'weight' column that hyperedges do not natively have); this is an approximation "
        "with all projected pairs weighted equally (weight=1.0) — see _flatten_to_pairwise "
        "docstring and Idea D section 2 for why a hyperedge-native statistic is future work."
    )

    ecr_flag = compute_ecr_flag(splits)
    tbmr = compute_tbvr(train_df, pairwise[["src_kc", "dst_kc"]]) if not pairwise.empty else 0.0

    rho = None
    if test_df is not None and not pairwise.empty:
        pairwise_weighted = pairwise.assign(weight=1.0)
        empty_sim = pd.DataFrame(columns=["src_kc", "dst_kc", "weight"])
        rho = compute_rho_edge_outcome(pairwise_weighted, empty_sim, test_df)

    group_leak = 0.0
    if held_out_interaction_ids is not None and member_to_interaction_id is not None:
        group_leak = compute_group_membership_leakage(
            hyperedges, held_out_interaction_ids, member_to_interaction_id
        )
    else:
        notes.append("group_membership_leak_rate not computed: missing held-out id inputs")

    return HyperedgeLeakageReport(
        hyperedge_kind=kind,
        n_hyperedges=len(hyperedges),
        ecr_flag=ecr_flag,
        tbmr=tbmr,
        rho=rho,
        group_membership_leak_rate=group_leak,
        cross_modal_leak_rate=None,
        intervention_timing_leak_rate=None,
        notes=notes,
    )
