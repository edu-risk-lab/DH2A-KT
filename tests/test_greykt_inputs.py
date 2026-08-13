"""Plumbing tests for GreyKT inputs (no torch_geometric required)."""

from __future__ import annotations

import pandas as pd
import pytest

from dh2a_kt.train.greykt_inputs import (
    concept_training_frequency,
    prereq_edge_index_from_e_pre,
    prior_mean_from_train,
)


def test_concept_training_frequency_and_prior():
    train = pd.DataFrame(
        {
            "kc_id": [10, 10, 20, 30],
            "correct": [1, 0, 1, 1],
        }
    )
    kc_to_idx = {10: 0, 20: 1, 30: 2, 99: 3}
    freq = concept_training_frequency(train, kc_to_idx)
    assert list(freq) == [2.0, 1.0, 1.0, 0.0]
    assert prior_mean_from_train(train) == pytest.approx(0.75)


def test_prereq_edge_index_drops_unmapped():
    pytest.importorskip("torch")
    e_pre = pd.DataFrame({"src_kc": [10, 20, 77], "dst_kc": [20, 30, 10]})
    kc_to_idx = {10: 0, 20: 1, 30: 2}
    idx = prereq_edge_index_from_e_pre(e_pre, kc_to_idx)
    assert idx.shape == (2, 2)
    assert idx.tolist() == [[0, 1], [1, 2]]
