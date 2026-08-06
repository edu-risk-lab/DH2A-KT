"""M4 gate: DH2KT embeddings -> PropensityScoreATE on real P0 train data."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from dh2a_kt.hyperedge.p0_inputs import REPO_ROOT, load_configs, load_e_pre, load_interactions_with_ids, get_fold_splits
from dh2a_kt.models.causal_integration import (
    build_causal_dataset_from_train,
    build_entity_maps,
    estimate_ate_with_dh2kt_confounders,
    hyperedge_index_from_e_pre,
    run_causal_pipeline,
)

pytest.importorskip("torch")
from dh2a_kt.models.dh2_kt import DH2KT, DH2KTConfig

XES3G5M_CONFIG = REPO_ROOT / "configs" / "xes3g5m.yaml"
P0_PARQUET = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed" / "xes3g5m.parquet"


def _real_data_available() -> bool:
    if not XES3G5M_CONFIG.exists() or not P0_PARQUET.exists():
        return False
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    from dh2a_kt.hyperedge.p0_inputs import e_pre_export_path

    return e_pre_export_path(p0_cfg, 0).exists()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _real_data_available(), reason="P0 XES3G5M processed data not present"),
]


@pytest.mark.slow
def test_causal_layer_integration_xes3g5m_fold0(caplog):
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    fold = 0
    e_pre = load_e_pre(p0_cfg, fold)
    interactions = load_interactions_with_ids(p0_cfg)
    train_df = get_fold_splits(interactions, p0_cfg, fold)["train"]
    train_users = train_df["user_id"].unique()[:64]
    train_sample = train_df[train_df["user_id"].isin(train_users)]

    kc_to_idx, item_to_idx = build_entity_maps(train_sample, e_pre)
    config = DH2KTConfig(
        n_concepts=len(kc_to_idx),
        n_exercises=len(item_to_idx),
        hidden_dim=32,
        embed_dim=32,
        n_hypergraph_layers=1,
        dropout=0.0,
        hyperedge_kinds=("concept_prerequisite",),
    )
    model = DH2KT(config)
    hyperedge_index_from_e_pre(e_pre, kc_to_idx)  # smoke: builds without error

    causal_data = build_causal_dataset_from_train(
        train_sample,
        e_pre,
        model,
        max_users=64,
        max_seq_len=15,
        device="cpu",
    )
    assert causal_data.confounders.ndim == 2
    assert len(causal_data.confounders) == len(causal_data.treatment) == len(causal_data.outcome)
    assert len(causal_data.confounders) >= 100
    assert set(np.unique(causal_data.treatment)) == {0, 1}

    with caplog.at_level(logging.INFO):
        ate, diagnostics = estimate_ate_with_dh2kt_confounders(causal_data)

    assert np.isfinite(ate)
    assert diagnostics["n_treated"] > 0
    assert diagnostics["n_control"] > 0
    assert 0.0 <= diagnostics["propensity_min"] <= diagnostics["propensity_max"] <= 1.0
    assert any("propensity=" in record.message for record in caplog.records)


@pytest.mark.slow
def test_run_causal_pipeline_helper():
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    interactions = load_interactions_with_ids(p0_cfg)
    train_df = get_fold_splits(interactions, p0_cfg, 0)["train"]
    e_pre = load_e_pre(p0_cfg, 0)

    ate, diagnostics = run_causal_pipeline(
        train_df,
        e_pre,
        max_users=32,
        max_seq_len=12,
        hidden_dim=32,
        device="cpu",
    )
    assert np.isfinite(ate)
    assert "propensity_min" in diagnostics and "propensity_max" in diagnostics
