"""Tests for Idea D's own contribution: hyperedge construction + the extended
leakage audit (docs/idea-D-plan.md section 2)."""
from __future__ import annotations

import pandas as pd

from dh2a_kt.hyperedge.construction import Hyperedge, build_concept_prerequisite_hyperedges
from dh2a_kt.hyperedge.audit import (
    DEGENERATE_CONSTANT_WEIGHT,
    audit_hyperedges,
    compute_group_membership_leakage,
    pairwise_rho_report,
)


def test_build_concept_prerequisite_hyperedges_groups_chains():
    e_pre = pd.DataFrame({
        "src_kc": [1, 2, 3, 10],
        "dst_kc": [2, 3, 4, 11],
        "weight": [1.0, 1.0, 1.0, 1.0],
    })
    hyperedges = build_concept_prerequisite_hyperedges(e_pre, fold=0, min_chain_len=3, max_chain_len=8)
    assert len(hyperedges) >= 1
    assert all(he.kind == "concept_prerequisite" for he in hyperedges)
    longest = max(hyperedges, key=lambda h: len(h.members))
    assert len(longest.members) >= 3


def test_group_membership_leakage_detects_tainted_hyperedge():
    he_clean = Hyperedge(
        hyperedge_id="a", kind="session", fold=0, train_only=True,
        members=[("student", 1), ("exercise", 2)],
    )
    he_tainted = Hyperedge(
        hyperedge_id="b", kind="session", fold=0, train_only=True,
        members=[("student", 1), ("exercise", 3)],
    )
    member_to_iid = {
        ("student", 1): {100},
        ("exercise", 2): {101},
        ("exercise", 3): {999},  # held-out interaction
    }
    rate = compute_group_membership_leakage(
        [he_clean, he_tainted], held_out_interaction_ids={999}, member_to_interaction_id=member_to_iid
    )
    assert rate == 0.5  # exactly 1 of 2 hyperedges tainted


def test_audit_hyperedges_empty_set_is_safe():
    report = audit_hyperedges([], splits={}, train_df=pd.DataFrame())
    assert report.n_hyperedges == 0
    assert report.ecr_flag == 0.0


def test_constant_weight_rho_is_degenerate_not_zero():
    import numpy as np

    rec = pairwise_rho_report(np.ones(100_000))
    assert rec["rho"] is None
    assert rec["rho_status"] == DEGENERATE_CONSTANT_WEIGHT
    assert rec["weight_std"] == 0.0
    assert rec["n_pairs"] == 100_000


def test_varying_weight_rho_is_marked_defined():
    import numpy as np

    rec = pairwise_rho_report(np.array([1.0, 2.0, 3.0, 4.0]))
    assert rec["rho_status"] == "defined"
    assert rec["weight_std"] > 0.0
