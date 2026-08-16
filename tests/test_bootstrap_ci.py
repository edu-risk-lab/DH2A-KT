"""The bootstrap's AUC must agree with sklearn, including on heavy ties."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bootstrap_protocol_ci", REPO_ROOT / "scripts" / "25_bootstrap_protocol_ci.py"
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["bootstrap_protocol_ci"] = _module
_spec.loader.exec_module(_module)

fast_auc = _module.fast_auc
group_positions = _module.group_positions


def test_fast_auc_matches_sklearn_on_continuous_scores():
    roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=5000)
    scores = rng.random(5000)
    assert fast_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_fast_auc_matches_sklearn_when_scores_are_mostly_tied():
    """Resampling learners duplicates score blocks, so ties are the normal case."""
    roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=4000)
    scores = rng.integers(0, 5, size=4000).astype(float)  # only 5 distinct values
    assert fast_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_fast_auc_matches_sklearn_after_duplicating_whole_groups():
    roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, size=500)
    scores = rng.random(500)
    labels = np.concatenate([labels, labels, labels])
    scores = np.concatenate([scores, scores, scores])
    assert fast_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_fast_auc_is_undefined_for_a_single_class():
    assert np.isnan(fast_auc(np.ones(10, dtype=int), np.linspace(0, 1, 10)))


def test_group_positions_indexes_every_row_of_each_learner():
    users = np.array([7, 3, 7, 3, 3, 9])
    groups = group_positions(users)
    assert sorted(groups) == [3, 7, 9]
    assert sorted(groups[3].tolist()) == [1, 3, 4]
    assert sorted(groups[7].tolist()) == [0, 2]
    assert groups[9].tolist() == [5]
    assert sum(g.size for g in groups.values()) == users.size
