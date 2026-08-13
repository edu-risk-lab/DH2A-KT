"""Verification of dh2a_kt/diagnostics/black_confidence.py against cases
with KNOWN ground truth.

The point of a premise diagnostic is to distinguish "C_B carries signal"
from "C_B carries no signal". A diagnostic that answers SUPPORTED either
way is worse than none at all, so this script constructs both worlds
synthetically and checks the verdict flips correctly, plus verifies the
hand-computable arithmetic (AUC via ranks, Spearman, Brier, ECE).

Pure numpy; runnable anywhere. Not part of the shipped pipeline.
"""

import numpy as np

from dh2a_kt.diagnostics.black_confidence import (
    VERDICT_NOT_SUPPORTED,
    VERDICT_SUPPORTED,
    auc_np,
    black_confidence_from_frequency,
    brier_score_np,
    expected_calibration_error_np,
    format_report,
    negative_log_likelihood_np,
    spearman_np,
    stratify_by_black_confidence,
)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


# ============================================================
# Part 1 -- arithmetic, hand-computed
# ============================================================
print("--- Part 1: arithmetic ---")

# AUC: pairs (pos,neg) -> 0.35>0.1 win, 0.35<0.4 loss, 0.8>0.1 win, 0.8>0.4 win = 3/4
probs = np.array([0.1, 0.4, 0.35, 0.8])
labels = np.array([0.0, 0.0, 1.0, 1.0])
check("AUC matches hand-computed 0.75", abs(auc_np(probs, labels) - 0.75) < 1e-12)

check("AUC ties give 0.5", abs(auc_np(np.array([0.5, 0.5]), np.array([0.0, 1.0])) - 0.5) < 1e-12)
check("AUC single-class returns NaN", not np.isfinite(auc_np(np.array([0.3, 0.7]), np.array([1.0, 1.0]))))
check("AUC perfect separation == 1.0", abs(auc_np(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) - 1.0) < 1e-12)
check("AUC perfectly inverted == 0.0", abs(auc_np(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) - 0.0) < 1e-12)

check("Brier hand-computed", abs(brier_score_np(np.array([0.9, 0.1]), np.array([1.0, 0.0])) - 0.01) < 1e-12)
check("Spearman perfect monotonic == 1.0", abs(spearman_np(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40])) - 1.0) < 1e-12)
check("Spearman perfect anti == -1.0", abs(spearman_np(np.array([1.0, 2, 3, 4]), np.array([40.0, 30, 20, 10])) + 1.0) < 1e-12)
check("Spearman is rank-based, not linear (monotone nonlinear -> 1.0)",
      abs(spearman_np(np.array([1.0, 2, 3, 4]), np.array([1.0, 100, 10000, 1e8])) - 1.0) < 1e-12)

nll_confident_right = negative_log_likelihood_np(np.array([0.99]), np.array([1.0]))
nll_confident_wrong = negative_log_likelihood_np(np.array([0.99]), np.array([0.0]))
check("NLL punishes confident-wrong far more than confident-right", nll_confident_wrong > 10 * nll_confident_right)

# ECE: two groups, each perfectly confident but 10% miscalibrated
ece = expected_calibration_error_np(np.array([0.1, 0.1, 0.9, 0.9]), np.array([0.0, 0.0, 1.0, 1.0]), n_bins=10)
check("ECE hand-computed == 0.1", abs(ece - 0.1) < 1e-9)

check("C_B formula: N=100, kappa=4 -> 100/104", abs(black_confidence_from_frequency(np.array([100.0]), 4.0)[0] - 100.0 / 104.0) < 1e-12)
check("C_B formula: N=0 -> 0.0", black_confidence_from_frequency(np.array([0.0]), 4.0)[0] == 0.0)

# ============================================================
# Part 2 -- WORLD A: the premise is TRUE by construction.
# Rare concepts are predicted badly; frequent concepts well.
# ============================================================
print("\n--- Part 2: WORLD A (C_B genuinely predicts error) ---")
rng = np.random.default_rng(0)
n_concepts = 200
# frequency spans 4 orders of magnitude, like a real KT corpus
freqs = np.exp(rng.uniform(np.log(1.0), np.log(10000.0), size=n_concepts))
cb_table = black_confidence_from_frequency(freqs, 4.0)

n_pred = 40000
cids = rng.integers(0, n_concepts, size=n_pred)
labels_a = rng.integers(0, 2, size=n_pred).astype(float)
# Well-trained concepts (high C_B) -> prediction close to the label.
# Under-trained concepts (low C_B) -> prediction pulled toward 0.5 (uninformative).
skill = cb_table[cids]
probs_a = 0.5 + (labels_a - 0.5) * (0.05 + 0.90 * skill)
probs_a = np.clip(probs_a + rng.normal(0, 0.02, size=n_pred), 0.01, 0.99)

diag_a = stratify_by_black_confidence(probs_a, labels_a, cids, freqs, kappa_b=4.0, n_buckets=5)
print(format_report(diag_a))
check("WORLD A verdict is SUPPORTED", diag_a.verdict == VERDICT_SUPPORTED)
check("WORLD A correlation is negative", diag_a.spearman_cb_vs_squared_error < 0)
check("WORLD A effect size exceeds threshold", abs(diag_a.spearman_cb_vs_squared_error) >= diag_a.min_effect_size)
check("WORLD A Brier worse in lowest-C_B bucket", diag_a.brier_lowest_cb_bucket > diag_a.brier_highest_cb_bucket)

# ============================================================
# Part 3 -- WORLD B: the premise is FALSE by construction.
# Prediction quality is identical regardless of concept frequency.
# This is the case that must NOT return SUPPORTED.
# ============================================================
print("\n--- Part 3: WORLD B (C_B carries NO signal) ---")
labels_b = rng.integers(0, 2, size=n_pred).astype(float)
probs_b = 0.5 + (labels_b - 0.5) * 0.6  # uniformly decent, independent of C_B
probs_b = np.clip(probs_b + rng.normal(0, 0.02, size=n_pred), 0.01, 0.99)

diag_b = stratify_by_black_confidence(probs_b, labels_b, cids, freqs, kappa_b=4.0, n_buckets=5)
print(format_report(diag_b))
check("WORLD B verdict is NOT_SUPPORTED", diag_b.verdict == VERDICT_NOT_SUPPORTED)
check("WORLD B effect size is negligible", abs(diag_b.spearman_cb_vs_squared_error) < 0.05)
check("WORLD B is reported as 'no usable relationship', not 'backwards'",
      any("No usable relationship" in n for n in diag_b.notes))
check("WORLD B note does NOT claim the sign is inverted",
      not any("BACKWARDS" in n for n in diag_b.notes))
# AUC is 1.0 in every bucket of WORLD A by construction (probs are a monotone
# function of the label), while Brier spans ~100x. This is exactly why the
# verdict must not rest on AUC: a purely AUC-based diagnostic would have
# reported "no difference between buckets" and missed the entire effect.
check("WORLD A: AUC is uninformative here while Brier is decisive",
      all(abs(b.auc - 1.0) < 1e-9 for b in diag_a.buckets)
      and diag_a.brier_lowest_cb_bucket > 10 * diag_a.brier_highest_cb_bucket)

# ============================================================
# Part 4 -- WORLD C: sign is inverted (rare concepts predicted BETTER).
# Must be reported NOT_SUPPORTED, not silently accepted.
# ============================================================
print("\n--- Part 4: WORLD C (relationship inverted) ---")
labels_c = rng.integers(0, 2, size=n_pred).astype(float)
inverse_skill = 1.0 - cb_table[cids]
probs_c = 0.5 + (labels_c - 0.5) * (0.05 + 0.90 * inverse_skill)
probs_c = np.clip(probs_c + rng.normal(0, 0.02, size=n_pred), 0.01, 0.99)

diag_c = stratify_by_black_confidence(probs_c, labels_c, cids, freqs, kappa_b=4.0, n_buckets=5)
check("WORLD C verdict is NOT_SUPPORTED", diag_c.verdict == VERDICT_NOT_SUPPORTED)
check("WORLD C correlation is positive (wrong sign)", diag_c.spearman_cb_vs_squared_error > 0)
print(f"  verdict={diag_c.verdict}, rho={diag_c.spearman_cb_vs_squared_error:+.4f}")

# ============================================================
# Part 5 -- base-rate confound detection
# ============================================================
print("\n--- Part 5: base-rate confound detection ---")
# Rare concepts are intrinsically harder: their base rate is much lower.
labels_d = np.where(rng.random(n_pred) < 0.15 + 0.7 * cb_table[cids], 1.0, 0.0)
probs_d = np.clip(0.5 + (labels_d - 0.5) * 0.6 + rng.normal(0, 0.02, n_pred), 0.01, 0.99)
diag_d = stratify_by_black_confidence(probs_d, labels_d, cids, freqs, kappa_b=4.0, n_buckets=5)
check("base-rate confound is flagged", diag_d.base_rate_confound_warning)
check("confound note is present in the report",
      any("CONFOUND" in n for n in diag_d.notes))
print(f"  base_rate_spread={diag_d.base_rate_spread:.3f}, warning={diag_d.base_rate_confound_warning}")

# ============================================================
# Part 6 -- quantile vs equal_width bucketing matters
# ============================================================
print("\n--- Part 6: bucketing strategy ---")
diag_eq = stratify_by_black_confidence(
    probs_a, labels_a, cids, freqs, kappa_b=4.0, n_buckets=5, strategy="equal_width"
)
sizes_quantile = [b.n for b in diag_a.buckets]
sizes_equal = [b.n for b in diag_eq.buckets]
check("quantile bucketing gives roughly balanced buckets",
      max(sizes_quantile) / min(sizes_quantile) < 1.5)
check("equal_width bucketing is heavily imbalanced on saturating C_B (why quantile is the default)",
      max(sizes_equal) / min(sizes_equal) > 3.0)
print(f"  quantile sizes: {sizes_quantile}")
print(f"  equal_width sizes: {sizes_equal}")

print("\nAll black-confidence diagnostic checks passed (numpy).")
print("The diagnostic correctly separates 'C_B has signal' from 'C_B has none'.")
