"""Does C_B actually predict black-box error? A precondition test for GreyKT.

GreyKT's reliability gate is g_B = C_B / (C_B + C_W + eps), where
``C_B(c) = N_c^train / (N_c^train + kappa_B)`` is a *proxy* for how much
the black-box branch can be trusted on concept ``c``: the more often the
concept appeared in training, the more reliable its learned representation
is assumed to be. That assumption is the load-bearing premise of the whole
gate. If it is false -- if concepts with low C_B are predicted just as well
as concepts with high C_B -- then the gate is steering fusion using a
signal that carries no information, and no amount of downstream
architecture work (calibration, backoff, hierarchical priors, ...) can
repair it. It has to be redesigned instead.

This module tests that premise directly, and *without* requiring GreyKT:
it needs only a trained black-box model's validation predictions, the
matching labels, the concept id per prediction, and per-concept training
frequencies. So it can (and should) be run *before* committing to a full
GreyKT training run.

Deliberately pure numpy -- no torch, no sklearn, no scipy -- so it runs on
a prediction dump anywhere, and so its own arithmetic is unit-testable in
environments where the deep-learning stack is unavailable.

Method
------
Predictions are stratified into buckets by C_B, and each bucket reports
AUC, NLL, Brier score, ECE, size, and base rate. Two statistics decide
the verdict:

1. ``spearman_cb_vs_squared_error`` -- rank correlation between C_B and
   per-prediction squared error, over *individual predictions* (not
   bucket means). The premise predicts a clearly negative value.
2. The bucket trend -- Brier score in the lowest-C_B bucket versus the
   highest-C_B bucket.

Three methodological cautions are built into the reporting rather than
left to the reader:

- **AUC is not comparable across buckets.** AUC is a within-group ranking
  metric, so it is confounded by each bucket's label balance and intrinsic
  difficulty spread. A low-C_B bucket can show low AUC for reasons having
  nothing to do with reliability. AUC is reported because it is the
  familiar number, but the verdict rests on NLL/Brier (per-prediction
  proper scoring rules, directly comparable across buckets) and on the
  per-prediction rank correlation.
- **Statistical significance is not enough.** Validation sets here have
  10^5--10^6 predictions, at which scale a correlation of 0.01 is
  "significant" and meaningless. The verdict therefore requires
  ``|rho| >= min_effect_size`` (default 0.05) in addition to a small
  p-value. A significant-but-negligible correlation is reported as
  AMBIGUOUS, not SUPPORTED.
- **Base rate is a confound.** Rare concepts may be intrinsically harder,
  not merely under-trained. If per-bucket base rates differ substantially,
  a C_B-error relationship may reflect difficulty rather than model
  reliability, and the diagnostic says so explicitly. Distinguishing the
  two requires a difficulty-matched comparison, which this module flags as
  needed but does not itself perform.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_NOT_SUPPORTED = "NOT_SUPPORTED"
VERDICT_AMBIGUOUS = "AMBIGUOUS"


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x``, ties receiving their average rank (1-based)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auc_np(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U identity, tie-corrected.

    Returns NaN when the input has fewer than one of either class (AUC is
    undefined there) rather than raising -- buckets legitimately can be
    single-class, and the caller reports that as NaN.
    """
    labels = np.asarray(labels).astype(float)
    probs = np.asarray(probs).astype(float)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(probs)
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def negative_log_likelihood_np(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def brier_score_np(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error_np(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Standard equal-width binned ECE."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def spearman_two_sided_pvalue(rho: float, n: int) -> float:
    """Two-sided p-value for a Spearman correlation, normal approximation.

    Uses the t-transform with a normal tail (exact enough at the n >= 10^4
    scale this diagnostic runs at; avoids a scipy dependency). Returns NaN
    for degenerate input.
    """
    if not np.isfinite(rho) or n < 4 or abs(rho) >= 1.0:
        return float("nan") if not np.isfinite(rho) or n < 4 else 0.0
    t = abs(rho) * math.sqrt((n - 2) / (1.0 - rho**2))
    return float(math.erfc(t / math.sqrt(2.0)))


def black_confidence_from_frequency(
    concept_train_freq: np.ndarray, kappa_b: float
) -> np.ndarray:
    """C_B(c) = N_c^train / (N_c^train + kappa_B), elementwise."""
    freq = np.asarray(concept_train_freq, dtype=float)
    return freq / (freq + kappa_b)


def confidence_from_mc_std(std: np.ndarray) -> np.ndarray:
    """Map Bernoulli MC-dropout std in [0, 0.5] to C_B in [0, 1].

    ``C_B = clip(1 - 2*std, 0, 1)``. Rank-equivalent to ``-std``, so
    Spearman(C_B, error) equals Spearman(-std, error). The linear map is
    only for the gate's *scale* versus C_W; the diagnostic verdict does
    not depend on it under quantile bucketing.
    """
    std = np.asarray(std, dtype=float).reshape(-1)
    return np.clip(1.0 - 2.0 * np.maximum(std, 0.0), 0.0, 1.0)


def confidence_from_predictive_prob(probs: np.ndarray) -> np.ndarray:
    """C_B = |2p - 1|: how peaked the *deployed* black-box output is.

    0 at p=0.5 (unconfident), 1 at p in {0, 1}. Rank-equivalent to
    1 - H(Bernoulli(p))/log(2). This is output-confidence, not epistemic
    uncertainty: it will look SUPPORTED whenever errors concentrate near
    p=0.5, and can fail when the model is overconfident (high |2p-1| still
    wrong). It is the cheap next test after frequency and MC-dropout, not
    a claim that C_B measures training reliability.
    """
    p = np.asarray(probs, dtype=float).reshape(-1)
    return np.abs(2.0 * p - 1.0)


def diagnostic_json_name(dataset: str, fold: int, signal: str = "frequency") -> str:
    if signal == "frequency":
        return f"{dataset}_fold{fold}_black_confidence_diagnostic.json"
    return f"{dataset}_fold{fold}_black_confidence_diagnostic_{signal}.json"


@dataclass
class BucketStats:
    label: str
    n: int
    cb_min: float
    cb_max: float
    cb_mean: float
    base_rate: float
    auc: float
    nll: float
    brier: float
    ece: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlackConfidenceDiagnostic:
    """Result of the C_B premise test. ``verdict`` is the headline."""

    verdict: str
    n_predictions: int
    kappa_b: float
    strategy: str
    n_buckets: int
    buckets: list[BucketStats] = field(default_factory=list)
    signal: str = "frequency"
    confidence_p05: float = float("nan")
    confidence_p95: float = float("nan")
    global_auc: float = float("nan")

    spearman_cb_vs_squared_error: float = float("nan")
    spearman_pvalue: float = float("nan")
    spearman_cb_vs_absolute_error: float = float("nan")
    min_effect_size: float = 0.05

    brier_lowest_cb_bucket: float = float("nan")
    brier_highest_cb_bucket: float = float("nan")
    brier_gap_low_minus_high: float = float("nan")

    base_rate_spread: float = float("nan")
    base_rate_confound_warning: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["buckets"] = [b.as_dict() for b in self.buckets]
        return out


def stratify_confidence_vs_error(
    probs: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    n_buckets: int = 5,
    strategy: str = "quantile",
    n_ece_bins: int = 10,
    min_effect_size: float = 0.05,
    base_rate_spread_threshold: float = 0.10,
    signal: str = "frequency",
    kappa_b: float = float("nan"),
) -> BlackConfidenceDiagnostic:
    """Test whether a per-prediction C_B tracks black-box error."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    cb = np.asarray(confidence, dtype=float).reshape(-1)
    if not (probs.size == labels.size == cb.size):
        raise ValueError(
            f"probs/labels/confidence length mismatch: "
            f"{probs.size}/{labels.size}/{cb.size}"
        )
    if probs.size == 0:
        raise ValueError("no predictions supplied")

    squared_error = (probs - labels) ** 2
    absolute_error = np.abs(probs - labels)

    rho = spearman_np(cb, squared_error)
    pvalue = spearman_two_sided_pvalue(rho, cb.size)
    rho_abs = spearman_np(cb, absolute_error)

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_buckets + 1)
        edges = np.unique(np.quantile(cb, qs))
        if edges.size < 2:
            edges = np.array([cb.min(), cb.max() + 1e-9])
    elif strategy == "equal_width":
        edges = np.linspace(cb.min(), cb.max() + 1e-9, n_buckets + 1)
    else:
        raise ValueError(f"unknown strategy {strategy!r} (use 'quantile' or 'equal_width')")

    buckets: list[BucketStats] = []
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == edges.size - 2
        mask = (cb >= lo) & (cb <= hi) if last else (cb >= lo) & (cb < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        buckets.append(
            BucketStats(
                label=f"C_B[{lo:.4f},{hi:.4f}]",
                n=count,
                cb_min=float(cb[mask].min()),
                cb_max=float(cb[mask].max()),
                cb_mean=float(cb[mask].mean()),
                base_rate=float(labels[mask].mean()),
                auc=auc_np(probs[mask], labels[mask]),
                nll=negative_log_likelihood_np(probs[mask], labels[mask]),
                brier=brier_score_np(probs[mask], labels[mask]),
                ece=expected_calibration_error_np(probs[mask], labels[mask], n_ece_bins),
            )
        )

    notes: list[str] = []
    brier_low = buckets[0].brier if buckets else float("nan")
    brier_high = buckets[-1].brier if buckets else float("nan")
    brier_gap = brier_low - brier_high

    base_rates = [b.base_rate for b in buckets]
    base_rate_spread = float(max(base_rates) - min(base_rates)) if base_rates else float("nan")
    base_rate_confound = bool(
        np.isfinite(base_rate_spread) and base_rate_spread > base_rate_spread_threshold
    )

    # --- verdict ---
    # Order matters: magnitude is checked before sign, because a near-zero
    # correlation of either sign means "no usable relationship", which is a
    # different (and more accurately stated) finding than "the relationship
    # runs backwards".
    if not np.isfinite(rho):
        verdict = VERDICT_AMBIGUOUS
        notes.append("Spearman correlation undefined (degenerate C_B or error distribution).")
    elif abs(rho) < min_effect_size:
        verdict = VERDICT_NOT_SUPPORTED
        significance = (
            f"statistically significant (p={pvalue:.2e}) but practically negligible"
            if np.isfinite(pvalue) and pvalue < 0.05
            else f"neither significant nor material (p={pvalue:.2e})"
        )
        notes.append(
            f"No usable relationship between C_B and black-box error: rho={rho:+.4f}, "
            f"below the effect-size threshold of {min_effect_size}. At n={cb.size} "
            f"predictions this is {significance}. C_B as currently defined does not "
            "carry information about where the black-box branch errs, so a gate driven "
            "by it is steering on noise."
        )
    elif rho > 0.0:
        verdict = VERDICT_NOT_SUPPORTED
        notes.append(
            f"Relationship runs BACKWARDS: rho={rho:+.4f} (p={pvalue:.2e}). The premise "
            "requires a negative correlation (lower confidence, higher error), but here "
            "high-C_B predictions are *worse*. A gate using this signal would "
            "systematically trust the wrong branch."
        )
    elif np.isfinite(brier_gap) and brier_gap <= 0.0:
        verdict = VERDICT_AMBIGUOUS
        notes.append(
            f"Per-prediction correlation has the expected sign and magnitude "
            f"(rho={rho:+.4f}), but the bucket trend disagrees: Brier in the lowest-C_B "
            f"bucket ({brier_low:.4f}) is not worse than in the highest-C_B bucket "
            f"({brier_high:.4f}). The relationship is likely non-monotonic -- inspect "
            "the per-bucket table before trusting the gate."
        )
    else:
        verdict = VERDICT_SUPPORTED
        notes.append(
            f"C_B tracks black-box error as assumed: rho={rho:+.4f} (p={pvalue:.2e}), and "
            f"Brier degrades from {brier_high:.4f} in the highest-C_B bucket to "
            f"{brier_low:.4f} in the lowest. The reliability gate has an empirical basis."
        )

    notes.append(
        "AUC is reported per bucket for familiarity but is NOT comparable across buckets "
        "(within-group ranking metric, confounded by each bucket's label balance). The "
        "verdict rests on NLL/Brier and the per-prediction rank correlation."
    )
    cb_p05 = float(np.quantile(cb, 0.05))
    cb_p95 = float(np.quantile(cb, 0.95))
    if cb_p95 - cb_p05 < 0.05:
        notes.append(
            f"C_B is nearly saturated for signal={signal!r}: 5th–95th percentile = "
            f"[{cb_p05:.4f}, {cb_p95:.4f}]. A nearly-constant confidence cannot steer "
            "the gate even if Spearman is well-defined."
        )
    if base_rate_confound:
        notes.append(
            f"CONFOUND: base rate varies by {base_rate_spread:.3f} across buckets "
            f"(> {base_rate_spread_threshold}). Low-C_B items may be intrinsically "
            "harder rather than under-trained, so any C_B-error relationship here is "
            "partly difficulty, not purely model reliability. A difficulty-matched "
            "comparison is needed to separate them."
        )

    return BlackConfidenceDiagnostic(
        verdict=verdict,
        n_predictions=int(cb.size),
        kappa_b=float(kappa_b),
        strategy=strategy,
        n_buckets=len(buckets),
        buckets=buckets,
        signal=signal,
        confidence_p05=cb_p05,
        confidence_p95=cb_p95,
        global_auc=auc_np(probs, labels),
        spearman_cb_vs_squared_error=rho,
        spearman_pvalue=pvalue,
        spearman_cb_vs_absolute_error=rho_abs,
        min_effect_size=min_effect_size,
        brier_lowest_cb_bucket=brier_low,
        brier_highest_cb_bucket=brier_high,
        brier_gap_low_minus_high=brier_gap,
        base_rate_spread=base_rate_spread,
        base_rate_confound_warning=base_rate_confound,
        notes=notes,
    )


def stratify_by_black_confidence(
    probs: np.ndarray,
    labels: np.ndarray,
    concept_ids: np.ndarray,
    concept_train_freq: np.ndarray,
    *,
    kappa_b: float = 4.0,
    n_buckets: int = 5,
    strategy: str = "quantile",
    n_ece_bins: int = 10,
    min_effect_size: float = 0.05,
    base_rate_spread_threshold: float = 0.10,
) -> BlackConfidenceDiagnostic:
    """Frequency-proxy C_B: look up N_c^train / (N_c^train + kappa_b) per prediction."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    concept_ids = np.asarray(concept_ids).reshape(-1).astype(int)
    if not (probs.size == labels.size == concept_ids.size):
        raise ValueError(
            f"probs/labels/concept_ids length mismatch: "
            f"{probs.size}/{labels.size}/{concept_ids.size}"
        )
    cb_table = black_confidence_from_frequency(concept_train_freq, kappa_b)
    if concept_ids.size and concept_ids.max() >= cb_table.size:
        raise ValueError(
            f"concept id {concept_ids.max()} out of range for "
            f"concept_train_freq of length {cb_table.size}"
        )
    cb = cb_table[concept_ids]
    return stratify_confidence_vs_error(
        probs,
        labels,
        cb,
        n_buckets=n_buckets,
        strategy=strategy,
        n_ece_bins=n_ece_bins,
        min_effect_size=min_effect_size,
        base_rate_spread_threshold=base_rate_spread_threshold,
        signal="frequency",
        kappa_b=kappa_b,
    )


def format_report(diag: BlackConfidenceDiagnostic) -> str:
    """Human-readable report, suitable for a log or a note to a co-author."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"C_B PREMISE DIAGNOSTIC -- VERDICT: {diag.verdict}")
    lines.append("=" * 78)
    lines.append(
        f"signal={diag.signal}  predictions={diag.n_predictions}  kappa_B={diag.kappa_b}  "
        f"strategy={diag.strategy}  buckets={diag.n_buckets}  "
        f"C_B[p05,p95]=[{diag.confidence_p05:.4f},{diag.confidence_p95:.4f}]"
    )
    lines.append("")
    header = (
        f"{'bucket':<24}{'n':>9}{'C_B mean':>10}{'base':>8}"
        f"{'AUC':>8}{'NLL':>8}{'Brier':>8}{'ECE':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for b in diag.buckets:
        lines.append(
            f"{b.label:<24}{b.n:>9}{b.cb_mean:>10.4f}{b.base_rate:>8.3f}"
            f"{b.auc:>8.4f}{b.nll:>8.4f}{b.brier:>8.4f}{b.ece:>8.4f}"
        )
    lines.append("")
    lines.append(
        f"Spearman(C_B, squared error) = {diag.spearman_cb_vs_squared_error:+.4f} "
        f"(p={diag.spearman_pvalue:.2e}, threshold |rho|>={diag.min_effect_size})"
    )
    lines.append(
        f"Spearman(C_B, absolute error) = {diag.spearman_cb_vs_absolute_error:+.4f}"
    )
    lines.append(
        f"Brier: lowest-C_B bucket {diag.brier_lowest_cb_bucket:.4f} vs "
        f"highest-C_B bucket {diag.brier_highest_cb_bucket:.4f} "
        f"(gap {diag.brier_gap_low_minus_high:+.4f})"
    )
    lines.append("")
    for note in diag.notes:
        lines.append(f"  - {note}")
    lines.append("")
    if diag.verdict == VERDICT_NOT_SUPPORTED:
        if diag.signal == "predictive":
            next_signal = (
                "stop GreyKT on this dataset (frequency, MC-dropout, and "
                "predictive confidence all failed) — a multi-seed ensemble is "
                "the same epistemic hypothesis as MC-dropout at much higher cost"
            )
        elif diag.signal == "mc_dropout":
            next_signal = (
                "predictive confidence C_B = |2p_B - 1| of the eval-mode "
                "output (one forward, ~40s), not a multi-seed ensemble"
            )
        else:
            next_signal = (
                "MC-dropout variance or predictive |2p-1|, which measure model "
                "uncertainty / output confidence rather than training frequency"
            )
        lines.append(
            "ACTION: do not proceed to a full GreyKT run on this basis. The reliability "
            f"gate needs a different black-box confidence signal (e.g. {next_signal})."
        )
    elif diag.verdict == VERDICT_AMBIGUOUS:
        lines.append(
            "ACTION: inspect the per-bucket table before committing. Consider an "
            "alternative confidence signal, or a difficulty-matched re-run if the base "
            "rate confound is flagged above."
        )
    else:
        lines.append(
            "ACTION: the gate premise holds on this split. Proceeding to a full GreyKT "
            "run is justified; report this diagnostic alongside the results as the "
            "empirical warrant for the gate design."
        )
    lines.append("=" * 78)
    return "\n".join(lines)
