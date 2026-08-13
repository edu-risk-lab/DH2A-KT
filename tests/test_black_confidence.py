"""Numpy-only tests for the C_B diagnostic (no torch required)."""

from __future__ import annotations

import numpy as np
import pytest

from dh2a_kt.diagnostics.black_confidence import (
    VERDICT_SUPPORTED,
    confidence_from_mc_std,
    confidence_from_predictive_prob,
    diagnostic_json_name,
    stratify_by_black_confidence,
    stratify_confidence_vs_error,
)


def test_frequency_saturation_is_not_supported():
    rng = np.random.default_rng(0)
    n = 2000
    labels = rng.integers(0, 2, size=n).astype(float)
    probs = np.clip(labels + rng.normal(0, 0.2, size=n), 0.01, 0.99)
    concept_ids = rng.integers(0, 10, size=n)
    freq = np.linspace(8_000.0, 12_000.0, 10)
    diag = stratify_by_black_confidence(probs, labels, concept_ids, freq, kappa_b=4.0)
    assert diag.signal == "frequency"
    assert diag.confidence_p95 - diag.confidence_p05 < 0.05
    assert any("saturated" in note for note in diag.notes)


def test_mc_std_confidence_tracks_error_when_premise_holds():
    rng = np.random.default_rng(1)
    n = 4000
    std = rng.uniform(0.0, 0.4, size=n)
    labels = rng.integers(0, 2, size=n).astype(float)
    # High MC-std → noisy predictions (the premise we want SUPPORTED).
    noise = rng.normal(0.0, 0.05 + 2.0 * std)
    probs = np.clip(labels + noise, 0.01, 0.99)
    cb = confidence_from_mc_std(std)
    diag = stratify_confidence_vs_error(probs, labels, cb, signal="mc_dropout")
    assert diag.verdict == VERDICT_SUPPORTED
    assert diag.spearman_cb_vs_squared_error <= -0.05


def test_confidence_from_mc_std_is_rank_equivalent_to_neg_std():
    std = np.array([0.0, 0.1, 0.25, 0.5])
    cb = confidence_from_mc_std(std)
    assert cb[0] == pytest.approx(1.0)
    assert cb[-1] == pytest.approx(0.0)
    assert np.all(np.diff(cb) < 0)


def test_diagnostic_json_name_keeps_frequency_artefact_stable():
    assert (
        diagnostic_json_name("xes3g5m", 0, "frequency")
        == "xes3g5m_fold0_black_confidence_diagnostic.json"
    )
    assert (
        diagnostic_json_name("xes3g5m", 0, "mc_dropout")
        == "xes3g5m_fold0_black_confidence_diagnostic_mc_dropout.json"
    )
    assert (
        diagnostic_json_name("xes3g5m", 0, "predictive")
        == "xes3g5m_fold0_black_confidence_diagnostic_predictive.json"
    )


def test_predictive_confidence_is_abs_2p_minus_1():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cb = confidence_from_predictive_prob(p)
    assert list(cb) == pytest.approx([1.0, 0.5, 0.0, 0.5, 1.0])
