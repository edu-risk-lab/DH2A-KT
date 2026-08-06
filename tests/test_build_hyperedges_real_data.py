"""M1 gate: concept-prerequisite hyperedges on real P0 exports (XES3G5M fold 0).

Marked slow — skipped automatically when P0 processed data is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dh2a_kt.hyperedge.audit import audit_hyperedges
from dh2a_kt.hyperedge.construction import build_concept_prerequisite_hyperedges
from dh2a_kt.hyperedge.p0_inputs import (
    REPO_ROOT,
    build_concept_member_interaction_map,
    e_pre_export_path,
    get_fold_splits,
    held_out_interaction_ids,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)

XES3G5M_CONFIG = REPO_ROOT / "configs" / "xes3g5m.yaml"
P0_PARQUET = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed" / "xes3g5m.parquet"


def _real_data_available() -> bool:
    if not XES3G5M_CONFIG.exists() or not P0_PARQUET.exists():
        return False
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    return e_pre_export_path(p0_cfg, 0).exists()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _real_data_available(), reason="P0 XES3G5M processed data not present"),
]


@pytest.mark.slow
def test_build_hyperedges_real_data_xes3g5m_fold0():
    dh2_cfg, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    he_cfg = dh2_cfg["hyperedge"]["concept_prerequisite"]
    fold = 0

    e_pre = load_e_pre(p0_cfg, fold)
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, fold)

    hyperedges = build_concept_prerequisite_hyperedges(
        e_pre,
        fold=fold,
        min_chain_len=int(he_cfg["min_chain_len"]),
        max_chain_len=int(he_cfg["max_chain_len"]),
    )
    report = audit_hyperedges(
        hyperedges,
        splits=splits,
        train_df=splits["train"],
        test_df=splits.get("test"),
        held_out_interaction_ids=held_out_interaction_ids(splits),
        member_to_interaction_id=build_concept_member_interaction_map(splits["train"]),
    )

    assert len(hyperedges) > 0
    assert report.n_hyperedges > 0
    assert report.ecr_flag == 0.0
    assert report.hyperedge_kind == "concept_prerequisite"
