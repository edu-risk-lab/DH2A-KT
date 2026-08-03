"""Causal-effect layer for Tier 1 (docs/idea-D-plan.md Pha 2: "bat dau bang
propensity-score/2-stage regression truoc khi thu kien truc phuc tap hon").

This is a real, working minimal implementation (inverse-propensity-weighted
ATE estimator), not a stub — deliberately simple per the plan's own
instruction to start simple and justify complexity later. It estimates the
average treatment effect of a binary "intervention" (e.g. hint given /
teacher feedback given) on a binary outcome (e.g. next-step correctness),
adjusting for observed confounders.

Identification assumption (must be stated in the paper, per the risk table in
docs/idea-D-plan.md section 10): no unmeasured confounding given
``confounders`` — i.e. intervention assignment is as-good-as-random once you
condition on the supplied features. This is a strong assumption on
observational KT data and should be reported as a limitation, not proven.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "causal_layer requires scikit-learn. Install with: pip install scikit-learn"
    ) from exc


class PropensityScoreATE:
    """Minimal inverse-propensity-weighted average treatment effect (ATE)
    estimator — the Pha 2 starting point before a learned/graph-integrated
    causal layer is attempted.

    Usage:
        est = PropensityScoreATE().fit(confounders, treatment)
        ate, diagnostics = est.estimate_ate(treatment, outcome)
    """

    def __init__(self, clip_propensity: tuple[float, float] = (0.05, 0.95)):
        self.clip_propensity = clip_propensity
        self._model: LogisticRegression | None = None

    def fit(self, confounders: np.ndarray, treatment: np.ndarray) -> "PropensityScoreATE":
        if confounders.ndim != 2:
            raise ValueError("confounders must be a 2D array (n_samples, n_features)")
        self._model = LogisticRegression(max_iter=1000).fit(confounders, treatment)
        return self

    def propensity_scores(self, confounders: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call .fit() first")
        p = self._model.predict_proba(confounders)[:, 1]
        lo, hi = self.clip_propensity
        return np.clip(p, lo, hi)

    def estimate_ate(
        self, confounders: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
    ) -> tuple[float, dict]:
        """Horvitz-Thompson / IPW ATE estimate.

        Returns (ate, diagnostics) where diagnostics includes propensity score
        summary stats useful for sanity-checking overlap (a near-0 or near-1
        propensity mass indicates poor overlap and an unreliable estimate —
        report this, do not silently trust the point estimate).
        """
        if self._model is None:
            self.fit(confounders, treatment)
        p = self.propensity_scores(confounders)
        treated = treatment == 1
        control = treatment == 0

        weight_treated = 1.0 / p[treated]
        weight_control = 1.0 / (1.0 - p[control])

        mu1 = float(np.sum(outcome[treated] * weight_treated) / np.sum(weight_treated)) if treated.any() else float("nan")
        mu0 = float(np.sum(outcome[control] * weight_control) / np.sum(weight_control)) if control.any() else float("nan")
        ate = mu1 - mu0

        diagnostics = {
            "n_treated": int(treated.sum()),
            "n_control": int(control.sum()),
            "propensity_min": float(p.min()),
            "propensity_max": float(p.max()),
            "propensity_mean": float(p.mean()),
            "mu1_ipw": mu1,
            "mu0_ipw": mu0,
        }
        if p.min() <= self.clip_propensity[0] + 1e-9 or p.max() >= self.clip_propensity[1] - 1e-9:
            logger.warning(
                "Propensity scores near clip bounds (%s) — weak overlap, ATE estimate may be unreliable.",
                self.clip_propensity,
            )
        return ate, diagnostics
