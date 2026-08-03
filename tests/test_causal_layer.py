"""Test for dh2a_kt/models/causal_layer.py::PropensityScoreATE.

Per docs/execution-plan.md section 4 ("Viec can lam ngay tuan nay"), this
module was the one already-working piece of code with no test yet -- this
closes that gap using synthetic data (does not require GPU or real KT data,
so it can run before M0/M4 are unblocked).

Ground truth is constructed explicitly: outcome = f(confounder) + true_ate *
treatment + noise, with treatment assignment itself a function of the
confounder (so a naive treated-vs-control mean difference would be
confounded). PropensityScoreATE.estimate_ate() should recover true_ate to
within a wide but non-trivial tolerance, and the raw confounded difference
should not.
"""

from __future__ import annotations

import numpy as np
import pytest

from dh2a_kt.models.causal_layer import PropensityScoreATE

TRUE_ATE = 0.20


def _make_confounded_data(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    # Treatment assignment depends on the confounder (propensity is not 0.5).
    propensity_true = 1.0 / (1.0 + np.exp(-(1.5 * x[:, 0])))
    treatment = rng.binomial(1, propensity_true)
    # Outcome depends on the same confounder plus the true treatment effect.
    noise = rng.normal(scale=0.1, size=n)
    outcome_prob = np.clip(0.5 + 0.3 * x[:, 0] + TRUE_ATE * treatment + noise, 0.0, 1.0)
    outcome = rng.binomial(1, outcome_prob)
    return x, treatment, outcome


def test_estimate_ate_recovers_known_effect_better_than_naive_difference():
    x, treatment, outcome = _make_confounded_data()

    est = PropensityScoreATE().fit(x, treatment)
    ate, diagnostics = est.estimate_ate(x, treatment, outcome)

    naive_diff = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()

    # IPW estimate should land closer to the true ATE than the naive,
    # confounded mean difference -- this is the property the module exists
    # to provide, not just "some number came out".
    assert abs(ate - TRUE_ATE) < abs(naive_diff - TRUE_ATE)
    assert abs(ate - TRUE_ATE) < 0.08


def test_estimate_ate_diagnostics_report_overlap():
    x, treatment, outcome = _make_confounded_data()

    est = PropensityScoreATE()
    _, diagnostics = est.estimate_ate(x, treatment, outcome)

    assert diagnostics["n_treated"] + diagnostics["n_control"] == len(treatment)
    assert 0.0 <= diagnostics["propensity_min"] <= diagnostics["propensity_max"] <= 1.0
    assert "mu1_ipw" in diagnostics and "mu0_ipw" in diagnostics


def test_estimate_ate_warns_on_poor_overlap(caplog):
    # Deterministic treatment assignment (treatment == x > 0) drives
    # propensity scores to the clip bounds -- overlap is poor and the module
    # must warn rather than silently return a point estimate.
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=(n, 1))
    treatment = (x[:, 0] > 0).astype(int)
    outcome = rng.binomial(1, 0.5, size=n)

    est = PropensityScoreATE(clip_propensity=(0.05, 0.95)).fit(x, treatment)
    with caplog.at_level("WARNING"):
        est.estimate_ate(x, treatment, outcome)

    assert any("overlap" in record.message for record in caplog.records)


def test_confounders_must_be_2d():
    est = PropensityScoreATE()
    with pytest.raises(ValueError):
        est.fit(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))
