# Pre-registration, round 3: decision rules, families, time semantics (23 Oct 2026)

This file fixes the rules, hypothesis families, analyses, and runs of
round 3 before any of them is computed or launched. It does not change
`docs/preregistration.md` (commit 99521c6) or
`docs/preregistration_junyi_onset.md` (commit 3f0e089); it adds a new
dated record, as Section 0 of `docs/preregistration.md` requires. Nothing
below may be changed after a result is seen; a change needs a new dated
section.

## 0. Reason and what is already known

Nine advisor comments on the unsubmitted Expert Systems manuscript
(23 Oct 2026; manuscript state of tag `es-submit-2026-10-22b`, which is
not modified) ask for: credit decided on an independent test set;
learner-level and fold-level uncertainty; a justified threshold with a
range and a statistical/practical distinction; one primary rule with a
defined role for the other; error control over the family of credit
decisions with a confirmatory/exploratory split; one time unit and a
per-corpus time-semantics table; removal of inputs not available at
prediction time; wider scope and matched tuning; and a reusable
procedure with a provenance gate.

Already known when this file is written, and therefore not blind:

- validation-AUC and held-out-AUC outcomes of all 68 contrasts of the
  ten-seed map under the interval and seed rules
  (`results/tables/credit_ladder_transfer_10seed.json`, `..._test.json`);
- the two-sided Holm test of mean difference 0 over the 68 contrasts and
  the threshold recount at tau in {1.0, ..., 3.0} x 1e-3;
- the Junyi history-only check (50k learners, fold 0).

Sections 4-6 re-analyse those runs under rules fixed here, after their
outcomes were seen. The manuscript says so in the text that reports
them. Only the runs of Section 8 are blind.

## 1. Prediction time and input availability

- Prediction time: the onset of item t+1, when the learner opens it.
- Available at that time: the responses, items, and onset times of steps
  1..t; completion times of steps 1..t where a corpus records them; the
  identity and onset time of item t+1.
- Not available: the response, duration, and completion time of item t+1,
  and any statistic computed from them or from later rows.
- A contrast whose on arm or control uses an unavailable input is not a
  credit decision. It is reported only as a leakage diagnostic
  (family D, Section 3).

## 2. Time semantics and unit

| Corpus | Raw field | Meaning | Raw unit | Stored | Query-side gap at t+1 | Available at onset |
|---|---|---|---|---|---|---|
| XES3G5M | `timestamps` | interaction start (dataset README) | ms | s (ms // 1000) | start(t+1) - start(t) | yes |
| ASSISTments 2012, one-second export | `start_time` | attempt start | datetime | s (script 54) | start(t+1) - start(t) | yes |
| ASSISTments 2012, 1,000 s export | `start_time` | attempt start | datetime | s // 1000 x 1000 | start(t+1) - start(t) | yes, coarse |
| Junyi | `time_done` | completion | microseconds | microseconds (not converted) | done(t+1) - done(t) | no: includes the duration of item t+1 |

- Unit for every new run: seconds, integer, before `log1p`. Junyi
  completion times are converted as `time_done // 1_000_000`.
- Existing Junyi runs used microseconds. Section 8.1 tests whether that
  changes the outcome; the manuscript no longer argues it qualitatively.
- The table is printed in the manuscript (data section).

## 3. Families

Contrasts are the paired on-arm minus control differences of the ten
training seeds 0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407 on one
learner fold.

**Family C (confirmatory).** The three messages the manuscript makes
claims about, in every attempt-onset setting where they were run:

- package with zero incidence versus no time;
- aligned gap versus T-zero;
- question-KC incidence (capacity-matched), where run.

Settings: XES3G5M LSTM reference recipe folds 0-2; XES3G5M attention
encoder folds 0-2; ASSISTments 2012 one-second export, corpus
configuration folds 0-2; ASSISTments 2012 one-second export, reference
recipe folds 0-2; and, if Section 8.2 runs, Junyi onset 50k corpus
configuration folds 0-2 and Junyi onset all learners reference recipe
folds 0-2. Incidence exists on XES3G5M fold 0, ASSISTments corpus
configuration folds 0-2, and ASSISTments reference fold 0.
Size: 29 contrasts without Junyi onset, 41 with it. The final outcomes
use the full family once every run of Section 8.2 exists; a missing run
is printed as pending and stays in the family count.

**Family E (exploratory).** Every other contrast on attempt-onset inputs:
observed-incidence package; aligned gap versus T-misaligned, T-shuffled,
T-boundary; T-boundary versus T-zero; package versus param-padded
no-time; every 1,000 s ASSISTments row; every history-only row; the
Junyi unit check of Section 8.1.

**Family D (diagnostic, no credit).** Contrasts that use an unavailable
input: every Junyi row with a completion-time query-side gap, and the
synthetic leakage simulation. They are reported as leakage diagnostics,
never in the credit map.

## 4. Primary metric and decision rule

- Metric: AUC on the held-out evaluation learners of the fold (the P0
  validation and test learners, `dh2_kt_auc` of each run), at the
  checkpoint chosen by early stopping on the internal selection split
  carved from training learners (`val_frac = 0.1`). The manuscript calls
  this test AUC. Validation AUC of the selection split is reported as a
  supplementary analysis only.
- tau = +0.002 test AUC (Section 6 for its basis).
- For each contrast with mean difference m, standard deviation s, n = 10:
  - credit test: one-sided paired t-test of H0: mu <= tau,
    p_tau = 1 - F_t9((m - tau) / (s / sqrt(10)));
  - positivity test: one-sided paired t-test of H0: mu <= 0,
    p_0 = 1 - F_t9(m / (s / sqrt(10))).
- Adjustment: family C, Holm over the family for p_tau and, separately,
  for p_0; family E, Benjamini-Hochberg over the family for each.
- Outcomes, alpha = 0.025 one-sided (the level of the earlier interval
  rule, which this rule reduces to when the family has one member):
  - **Credited**: adjusted p_tau <= 0.025;
  - **Positive, not credited**: not credited and adjusted p_0 <= 0.025;
  - **Unsupported**: otherwise.
  Family E outcomes are labelled exploratory.
- The seed rule is no longer a decision rule. Each row prints k/10, the
  number of seeds with difference >= tau, as a seed-stability
  descriptor. A credited row with k < 10 is printed "credited,
  seed-variable". The descriptor is a fraction of ten and is not compared
  across seed counts.
- The earlier outcomes (validation AUC, interval and seed rules, the
  every-seed-positive middle outcome) are kept in an appendix table
  beside the new ones; the text names every row whose outcome changes.

## 5. Learner-level uncertainty and variance components

- Predictions: every run of a family-C contrast is rescored from its
  checkpoint on the held-out learners with
  `scripts/03_train_tier1.py ... --load-checkpoint <ckpt> --dump-predictions <npz>`
  (same command and flags as training; writes learner id, target,
  probability per scored position). A rescored AUC must equal the
  run's stored `dh2_kt_auc` within 1e-6, else the run is reported and
  excluded from Section 5 only.
- Learner bootstrap per contrast: B = 2000 resamples of held-out
  learners with replacement, seed 20261023, the same resample for both
  arms and all ten seeds; statistic = mean over seeds of the AUC
  difference; 95% percentile interval. Reported beside the credit
  outcome; it does not decide credit.
- Variance components, settings with three folds and the same contrast
  on each fold: d_fs = mu + a_f + e_fs, one-way random effects,
  method-of-moments estimates of the between-fold variance (truncated
  at 0) and the within-fold seed variance; the mean learner-bootstrap
  variance of a single seed pair gives the learner component. The pooled
  mean over folds is printed with a t interval on the three fold means
  (df = 2). Three folds give a coarse fold component; the text says so.

## 6. Threshold basis and practical value

tau stays +0.002 (fixed before the ten-seed runs). Reported, not used to
choose a new tau:

- the seed standard deviation of control-arm AUC and of the differences
  across the map, against tau;
- outcome of every family-C and family-E row under the primary rule at
  tau in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0} x 1e-3;
- practical quantities for each family-C contrast from the dumped
  predictions, mean over seeds of on minus control: log loss, Brier
  score, expected calibration error (15 equal-width bins), accuracy at
  0.5, and the share of scored positions whose 0.5 decision flips.

The manuscript separates three statements: a difference is positive
(p_0), it reaches the credit threshold (p_tau), and its size on the
practical quantities. It does not claim educational benefit; no
learner-facing outcome was measured.

## 7. Scripts for Sections 4-6

- `python scripts/60_credit_primary.py` (no training): reads the two
  ten-seed JSONs and `results/tables/junyi_history_check.json`, applies
  Sections 3-4 and the tau curve of Section 6, the seed/fold variance
  components of Section 5 that need no predictions, and writes
  `results/tables/credit_primary_test.json`.
- `python scripts/61_learner_bootstrap.py --stage dump|analyse`
  (GPU for `dump`): Section 5 bootstrap and the practical quantities of
  Section 6.

## 8. Runs (GPU), blind

### 8.1 Junyi unit check

Group `junyi-sec`: Junyi first 50,000 learners, corpus configuration,
fold 0, completion-time gaps converted to seconds, arms `time-package`
and `t-zero` (both-mode), ten seeds: 20 runs. Contrasts: each arm in
seconds minus the same arm in microseconds, same seed; and the two map
contrasts recomputed in seconds. Family E (unit check) and family D
(the completion-time map contrasts). The manuscript reports the mean
seconds-minus-microseconds difference with its 95% interval.

### 8.2 Junyi attempt onset

Data check first, on the raw `junyi_ProblemLog_original.csv`: the
duration field (`time_taken`, else the sum of `time_taken_attempts`),
its unit, the share of rows where it is present and non-negative, and
the share where onset = done - duration precedes the previous row's
completion. The check and its numbers are committed before any run.

- If the duration is present and non-negative on at least 99% of rows:
  onset(t) = time_done(t) // 1e6 - duration(t), rows keep their
  `time_done` order, gaps are onset(t+1) - onset(t) seconds clipped at 0.
  Groups `junyi-onset-partial-fold{0,1,2}` (50k, corpus configuration)
  and `junyi-onset-full-fold{0,1,2}` (all learners, reference recipe),
  arms `time-package` and `t-zero`, ten seeds: 120 runs. The no-time
  arms of `junyi-partial*` and `junyi-full*` are reused (no-time does
  not read timestamps and the row order is unchanged).
- Otherwise: history-only on the remaining Junyi settings, groups
  `junyi-hist-fold{1,2}` (50k) and `junyi-hist-full-fold{0,1,2}`, arms
  `time-history` and `t-zero-history`: 100 runs; these rows join
  family C in place of the onset rows.

Queue: 8.2 data check, then 8.1, then 8.2 runs. A failed run is rerun
once with the same command; a run that fails twice is printed pending.

## 9. Manuscript changes this plan allows

- The credit map, Figure 2, abstract, and conclusion use test AUC and
  the Section 4 rule, family C, whatever they show. Junyi rows with a
  completion-time query gap move to the leakage section (family D).
- New: the time-semantics table, a families table, the learner-bootstrap
  and variance-component table, the practical-quantity table, the tau
  curve, Algorithm 1 with an availability and provenance audit as its
  first, gating step, an execution checklist, and a reproduction
  manifest linking data hash, command, seed, checkpoint, prediction file,
  and outcome for every family-C contrast.
- Wider scope and matched tuning (attention on more corpora, encoder and
  baseline tuning at a common budget, a wider simpleKT grid, a
  graph or prerequisite message at credit level) need a later dated
  addendum before their runs.
