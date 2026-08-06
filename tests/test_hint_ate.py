"""Unit tests for FoundationalASSIST-style hint ATE dataset construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dh2a_kt.models.causal_integration import build_hint_causal_dataset, estimate_hint_ate


def _toy() -> pd.DataFrame:
    # User 1: no hint then correct next; User 2: hint then incorrect next
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "item_id": [10, 11, 12, 20, 21, 22],
            "kc_id": [100, 100, 101, 200, 200, 201],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "correct": [1, 1, 0, 0, 0, 1],
            "hint_count": [0, 0, 1, 2, 0, 0],
        }
    )


def test_build_hint_causal_dataset_shapes():
    data = build_hint_causal_dataset(_toy())
    # 2 users × 2 transitions each = 4 rows
    assert len(data.treatment) == 4
    assert data.confounders.shape == (4, 5)
    assert set(np.unique(data.treatment)) == {0, 1}


def test_estimate_hint_ate_finite():
    data = build_hint_causal_dataset(_toy())
    ate, diag = estimate_hint_ate(data)
    assert np.isfinite(ate)
    assert "naive_mean_diff" in diag
    assert diag["treatment"] == "hint_used_t"
    assert diag["n_treated"] >= 1
    assert diag["n_control"] >= 1


def test_max_users_subsamples():
    df = _toy()
    data = build_hint_causal_dataset(df, max_users=1, seed=0)
    assert len(data.treatment) == 2
