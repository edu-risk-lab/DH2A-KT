"""Tests for baseline reuse (docs/idea-D-plan.md section 6, biggest compute saving)."""
from __future__ import annotations

import pytest

from dh2a_kt.baselines.loader import load_p0_baseline_summary, compare_against_p0


def test_load_xes3g5m_simplekt_baseline():
    df = load_p0_baseline_summary("xes3g5m", model="simplekt", graph_construction="train_only")
    assert len(df) == 1
    auc = float(df.iloc[0]["auc"])
    assert 0.8 < auc < 0.95  # sanity range, not an exact-match assertion


def test_compare_against_p0_reports_delta():
    result = compare_against_p0(dh2_kt_auc=0.90, dataset="xes3g5m", model="simplekt")
    assert result["delta_auc"] == pytest.approx(0.90 - result["p0_auc"])


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_p0_baseline_summary("not_a_real_dataset")
