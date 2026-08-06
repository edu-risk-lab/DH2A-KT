"""M5 gate: Tier 1 train/eval loop and P0 comparison table."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from dh2a_kt.hyperedge.p0_inputs import REPO_ROOT, load_configs, load_e_pre, load_interactions_with_ids, get_fold_splits
from dh2a_kt.train.tier1 import (
    TrainingBudget,
    next_step_loss,
    resolve_training_budget,
    train_and_evaluate_fold,
    write_comparison_table,
)

XES3G5M_CONFIG = REPO_ROOT / "configs" / "xes3g5m.yaml"
P0_PARQUET = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed" / "xes3g5m.parquet"


def _synthetic_logs(n_users: int = 16, seq_len: int = 12) -> pd.DataFrame:
    rows = []
    i = 0
    for u in range(n_users):
        for t in range(seq_len):
            rows.append(
                {
                    "user_id": u,
                    "item_id": (u + t) % 5,
                    "kc_id": t % 4,
                    "timestamp": u * 1000 + t,
                    "correct": (u + t) % 2,
                }
            )
            i += 1
    return pd.DataFrame(rows)


def test_resolve_training_budget_matches_p0_gkt():
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    budget = resolve_training_budget(p0_cfg, reference_model="gkt")
    assert budget.batch_size == 4
    assert budget.epochs == 10
    assert budget.matched_p0 is True


def test_next_step_loss_ignores_short_sequences():
    torch = pytest.importorskip("torch")
    logits = torch.randn(2, 4, 1)
    targets = torch.randint(0, 2, (2, 4)).float()
    lengths = torch.tensor([4, 1])
    loss = next_step_loss(logits, targets, lengths)
    assert torch.isfinite(loss)


def test_train_and_evaluate_fold_toy():
    torch = pytest.importorskip("torch")
    train_df = _synthetic_logs(n_users=12, seq_len=10)
    eval_df = _synthetic_logs(n_users=8, seq_len=10)
    e_pre = pd.DataFrame({"src_kc": [0, 1], "dst_kc": [1, 2], "weight": [1.0, 1.0]})
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=3,
        lr=1e-2,
        max_seq_len=10,
        matched_p0=True,
    )
    result = train_and_evaluate_fold(
        train_df,
        eval_df,
        e_pre,
        budget,
        device="cpu",
        hidden_dim=16,
    )
    assert np.isfinite(result.auc)
    assert result.n_predictions > 0


def test_write_comparison_table(tmp_path: Path):
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=10,
        lr=0.001,
        max_seq_len=200,
        matched_p0=True,
    )
    out = tmp_path / "dh2_kt_vs_p0.csv"
    df = write_comparison_table(
        [type("R", (), {"fold": 0, "auc": 0.85, "n_predictions": 1000})()],
        dataset="xes3g5m",
        budget=budget,
        comparison_models=["gkt"],
        output_path=out,
    )
    assert out.exists()
    assert "p0_auc" in df.columns
    assert "comparison_type" in df.columns
    assert df["comparison_type"].iloc[0] == "matched"


def _real_data_available() -> bool:
    if not XES3G5M_CONFIG.exists() or not P0_PARQUET.exists():
        return False
    from dh2a_kt.hyperedge.p0_inputs import e_pre_export_path

    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    return e_pre_export_path(p0_cfg, 0).exists()


@pytest.mark.slow
@pytest.mark.skipif(not _real_data_available(), reason="P0 XES3G5M processed data not present")
def test_train_tier1_smoke_xes3g5m_fold0(tmp_path: Path):
    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, 0)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    e_pre = load_e_pre(p0_cfg, 0)
    budget = resolve_training_budget(p0_cfg, reference_model="gkt", max_seq_len=50)
    smoke_budget = replace(budget, epochs=1)

    result = train_and_evaluate_fold(
        splits["train"],
        eval_df,
        e_pre,
        budget=smoke_budget,
        device="cpu",
        hidden_dim=32,
        max_users=32,
    )
    assert np.isfinite(result.auc)

    out = tmp_path / "dh2_kt_vs_p0.csv"
    write_comparison_table(
        [type("R", (), {"fold": 0, "auc": result.auc, "n_predictions": result.n_predictions})()],
        dataset="xes3g5m",
        budget=smoke_budget,
        comparison_models=["gkt"],
        output_path=out,
    )
    assert out.exists()
