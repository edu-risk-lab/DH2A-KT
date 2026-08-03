"""Smoke tests: P0 bridge must expose exactly the functions Idea D's plan
says it reuses (docs/idea-D-plan.md section 6 table)."""
from __future__ import annotations

import pandas as pd


def test_p0_bridge_imports():
    from dh2a_kt import p0_bridge

    for name in [
        "infer_prerequisites_from_train",
        "compute_ecr_flag",
        "compute_rho_edge_outcome",
        "compute_tbvr",
        "audit_dag",
        "apply_node_drop",
        "apply_edge_drop",
        "compute_dag_disruption_rate",
        "bin_kcs_by_frequency",
        "load_expert_dag",
        "learner_based_folds",
    ]:
        assert hasattr(p0_bridge, name), f"p0_bridge missing {name}"


def test_dag_audit_on_toy_graph():
    from dh2a_kt.p0_bridge import audit_dag

    edges = pd.DataFrame({
        "src_kc": [1, 2, 3],
        "dst_kc": [2, 3, 4],
        "weight": [1.0, 1.0, 1.0],
    })
    report = audit_dag(edges)
    assert report.topo_sort_passed is True
    assert report.n_cycles_after == 0
