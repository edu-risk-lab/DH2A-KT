# Pre-registration: round-2 experiments (advisor review, 8 Oct 2026)

This file fixes every run, seed, rule, and report format for the round-2
experiments before any of them is launched. It is committed before the
first GPU job. Nothing below may be changed after a result is seen; a
change needs a new dated section that says what changed and why, and the
original plan stays in the record.

Scope comes from four advisor comments on the Expert Systems manuscript
(tag `es-submit-2026-10-08b`) and four decisions taken on 8 Oct 2026:

1. The manuscript is submitted only after these experiments are in it.
2. The whole credit map moves from five to ten training seeds.
3. The leakage simulation also measures how much AUC a leak inflates.
4. AKT and simpleKT get a 12-configuration hyperparameter grid each.

## 1. Seeds

- Existing seeds: `0, 17, 42, 1234, 2024`.
- New seeds, fixed here: `7, 99, 314, 2718, 3407`.
- The ten-seed set is the union. No further seeds are added to any row
  after a result is seen.
- Training is deterministic for a given command and seed (the same
  observed-incidence arm written by scripts 27 and 34 has identical
  validation AUC on all five seeds). Existing per-seed files are reused;
  only the five new seeds are trained for existing rows.

## 2. Decision rules at ten seeds

- Threshold unchanged: tau = +0.002 validation AUC.
- Interval rule (primary): mean(delta) - t_{0.975,9} * SD(delta) / sqrt(10)
  >= tau, with t_{0.975,9} = 2.2621571627409915.
- Seed rule: every one of the ten paired deltas >= tau.
- Outcomes: Credited, Consistent-positive (not credited), Unsupported,
  defined as in Section 4.2 of the manuscript.
- The ten-seed map becomes the main map of the manuscript. The five-seed
  map is kept, unchanged, as an appendix record so the two can be
  compared row by row.
- The seed rule is stricter at ten seeds than at five. If the two rules
  disagree on a row, both outcomes are printed and the text says so.
  The rules are not re-tuned to restore agreement.

## 3. Rows of the credit map and the runs they need

Every row of `paper/tables/table_credit_transfer.tex` and
`paper/tables/table_credit_assist_resolution.tex` is re-scored at ten
seeds. Groups refer to `scripts/50_credit_ladder_plan.py`.

### 3.1 Existing rows, five new seeds each

| Group | Arms | New runs |
|---|---|---|
| `xes-ref` (XES3G5M fold 0, reference recipe) | incidence-on, no-time, time-package, time-on-observed, t-zero, t-misaligned | 30 |
| `package-matched` | no-time-param-padded | 5 |
| `xes-gap-controls` | t-boundary, t-shuffled | 10 |
| `xes-fold1`, `xes-fold2` | no-time, time-package, t-zero | 30 |
| `backbone-attn` | no-time, time-package, t-zero, t-misaligned | 20 |
| `assist-sec`, `assist-sec-ref` | six ASSIST arms | 60 |
| `assist`, `assist-ref` (1,000 s export) | six ASSIST arms | 60 |
| `assist-sec-fold1`, `assist-sec-fold2` | incidence-off, time-on-zero, t-zero | 30 |
| `junyi-partial` | no-time, time-package, t-zero | 15 |
| `junyi-full` | no-time, time-package, t-zero | 15 |

The XES3G5M fold-0 rows were trained by scripts 31, 34, and 38. Their new
seeds are trained through script 50 group `xes-ref` with the same flags,
so scripts 34 and 38 and their summary files are not touched. Group
`xes-ref` also reruns seed 42 for each of its six arms as a
reproducibility check: each must match the frozen seed-42 validation AUC
(`qkc_cm_*`, `a1_*`, `a7_dt_both`) to within 1e-6. If any does not match,
the new XES seeds are not used and the discrepancy is reported before
anything else is run.

Mapping of `xes-ref` arms to the frozen files:

| `xes-ref` arm | Frozen file (seeds 0, 17, 42, 1234, 2024) |
|---|---|
| incidence-on | `qkc_cm_observed_s*.csv` |
| no-time | `qkc_cm_zero_s*.csv` |
| time-package | `qkc_cm_zero_time_s*.csv` |
| time-on-observed | `a7_query_dt_ablation.csv`, mode `both` |
| t-zero | `a1_t_zero_s*.csv` |
| t-misaligned | `a1_t_misaligned_s*.csv` |

### 3.2 New learner folds, ten seeds each

| Group | Corpus, fold, recipe | Arms | Runs |
|---|---|---|---|
| `backbone-attn-fold1`, `-fold2` | XES3G5M, attention encoder, reference recipe | no-time, time-package, t-zero, t-misaligned | 80 |
| `assist-sec-cfg-fold1`, `-fold2` | ASSISTments 2012 one-second export, corpus configuration | six ASSIST arms | 120 |
| `junyi-full-fold1`, `-fold2` | Junyi, all learners, reference recipe | no-time, time-package, t-zero | 60 |
| `junyi-partial-fold1`, `-fold2` | Junyi, first 50,000 learners, corpus configuration | no-time, time-package, t-zero | 60 |

The ten-seed folds 1 and 2 of `assist-sec-ref` keep the three arms they
already have (`assist-sec-fold1`, `assist-sec-fold2`).

### 3.3 Order of the GPU queue

1. `xes-ref` seed-42 reproducibility check (6 runs). Stop if it fails.
2. One timing run each of `junyi-full-fold1` and `backbone-attn-fold1`.
   If a Junyi run takes more than 90 minutes, report before continuing.
3. `junyi-full-fold1`, `junyi-full-fold2`.
4. `backbone-attn-fold1`, `backbone-attn-fold2`.
5. `assist-sec-cfg-fold1`, `assist-sec-cfg-fold2`.
6. `junyi-partial-fold1`, `junyi-partial-fold2`.
7. New seeds of all existing groups (Section 3.1).
8. Baseline grid (Section 5).
9. Leakage simulation AUC runs (Section 6).

A failed run is rerun once with the same command. A run that fails
twice is reported as missing; its row is printed as pending, not
dropped.

## 4. Statistics computed from existing and new files (no training)

- Holm-Bonferroni over every contrast of the credit map, two-sided
  paired t-test of mean delta = 0, alpha = 0.05
  (`scripts/55_holm_all_contrasts.py`). It is reported beside the
  credit outcome and is not a credit rule. Computed at five seeds now
  and at ten seeds when the runs exist.
- Threshold sensitivity on every row of the map, not only XES3G5M fold 0:
  seed-pass counts at tau in {+1.0, +1.5, +2.0, +2.5, +3.0} x 1e-3 and the
  interval-rule outcome at each tau (`scripts/56_threshold_all_corpora.py`).
- Weighted leakage statistic: held-out support share per hyperedge,
  weighted by hyperedge support, reported beside the membership rate in
  the audit table (from the graded control of script 53).

## 5. Baseline grid (AKT and simpleKT)

- Corpus and protocol: XES3G5M fold 0, chunked windows, L = 400,
  repeat-target masking, the protocol of the contextual table.
- Fixed: batch 64, at most 30 epochs, engine patience 3, every other
  hyperparameter inherited from the vendored engine defaults.
- Grid, 12 configurations per model:
  learning rate in {1e-3, 3e-4} x dropout in {0.05, 0.10, 0.20} x
  d_model in {128, 256}. The inherited point (1e-3, 0.05, 256) is one
  of the twelve.
- Selection: highest validation AUC under seed 42. Test AUC is never
  used for selection.
- The selected configuration is then trained on the five seeds
  `0, 17, 42, 1234, 2024`, the seeds of the existing baseline rows.
- Report: the tuned row is added beside the inherited row; the inherited
  row stays. The tracer keeps its single optimiser vector, and the text
  says so.
- Runs: 24 grid runs, plus 8 seed runs (seed 42 is reused).

## 6. Leakage simulation

Both classes are synthetic and are labelled as such. They test whether
the audit registers the leak and how much AUC the leak adds. They are
not evidence about any public corpus.

Backbone for the AUC runs: XES3G5M fold 0, reference recipe,
zero-incidence time package (the credited XES3G5M arm), plus one added
input branch `Linear(1, hidden)` on a precomputed per-row scalar. Each
class has a leaky arm and a clean twin with the same branch, the same
marginal distribution of the scalar where stated, and the same seeds
`0, 17, 42, 1234, 2024`. 4 arms x 5 seeds = 20 runs.

### 6.1 Cross-modal (synthetic content channel)

- Scalar for row r (learner u, item q, start time t): mean correctness of
  item q over interactions of q in the full log, all learners, with
  start time > t, shrunk toward the global train mean with weight 10
  pseudo-counts. This plays the role of a content embedding pooled after
  the prediction cutoff.
- Clean twin: the same statistic over training-learner interactions of q
  with start time < t.
- Injection: next-step query side, as for the query gap.
- Audit statistic: post-cutoff support share, the share of interactions
  supporting the scalar that are timestamped after t or belong to a
  held-out learner. Expected 0 for the clean twin.

### 6.2 Intervention-timing (synthetic hint)

- Scalar for position t: a hint flag h_t chosen with knowledge of the
  learner's response at t + 1, the target scored from position t:
  P(h_t = 1 | y_{t+1} = 0) = 0.10, P(h_t = 1 | y_{t+1} = 1) = 0.03,
  drawn with seed 20261008.
- Clean twin: h_t drawn independently of every label with the marginal
  rate of the leaky flag on the training split.
- Injection: recurrent input at position t only.
- Audit statistic: selection-after-outcome rate, the share of hint rows
  whose selection used an outcome at or after the scored step, which is
  known only because the simulation records it; and a data-only residual
  check, the mean of y_{t+1} - p_{t+1} by hint flag on the training split,
  with p from the no-leak arm and a learner bootstrap 95% interval.

### 6.3 Report

- AUC inflation: leaky minus clean, validation and test-only, five seeds,
  mean and 95% interval.
- Audit detection: each statistic for the leaky arm and its twin.
- Graded contamination for the audit statistics only (no training):
  fractions {0, 1e-4, 1e-3, 1e-2, 0.1, 1} of rows built the leaky way.

## 7. Manuscript changes this plan allows

- Table 7, the 1,000 s table, Figure 2, the abstract, Limitations, and
  the conclusion are rewritten from the ten-seed map, whatever it shows.
- New rows for the new folds; the fold-0-only limitation is removed only
  for the settings that now have three folds.
- Holm column and the all-corpus threshold table are added.
- A tuned-baseline row and one abstract sentence stating that baselines
  are context, not a superiority claim.
- A leakage-simulation subsection with the Section 6 report.
- A new Zenodo release after the rewrite; Data availability and the
  cover letter cite its version DOI.

## 8. Implementation (fixed in the same commit, before the first run)

Training jobs (GPU server):

- Credit map: `python scripts/50_credit_ladder_plan.py --group <group> --launch-all`
  in the order of Section 3.3; the reproducibility check is
  `--group xes-ref --seed 42 --launch-all`. At this commit script 50 lists 601 missing
  credit-map runs; finished runs are skipped by their stamps.
- Leakage simulation: group `leak-sim` of script 50, arms
  `crossmodal-leak`, `crossmodal-clean`, `hint-leak`, `hint-clean`, seeds
  `0, 17, 42, 1234, 2024` (20 runs). Each arm is the `xes-ref`
  time-package command plus `--aux-scalar <arm>`
  (`dh2a_kt/data/leak_sim.py`; v4 config `aux_scalar`). The added
  `Linear(1, hidden)` is created after every other module, so the shared
  parameters start from the same values as the time-package arm.
  "Fit split" in Section 6 means training learners minus the internal
  selection learners (`val_frac = 0.1`, the same split as training).
- Baseline grid: `python scripts/57_baseline_grid.py --stage grid`, then
  `--stage seeds` (scripts 24 `--hp`, `--report-valid`). Validation AUC is
  scored from the saved checkpoint on the validation split only.

No-training jobs:

- `python scripts/53_partial_membership_control.py configs/xes3g5m.yaml --fold 0`
  (adds the support-weighted held-out share; the existing keys and
  values are unchanged).
- `python scripts/58_leak_simulation_audit.py --stage audit` (audit
  statistics, graded contamination; seed for graded draws 20261010).
- After the `leak-sim` runs: `python scripts/58_leak_simulation_audit.py
  --stage auc --device cuda`. Test-only AUC is rescored from each run's
  checkpoint on the test split. The residual check uses each of the five
  `hint-clean` checkpoints as the no-leak model, 1,000 learner bootstrap
  resamples; all five are reported.
- `python scripts/51_transfer_credit_tables.py --n-seeds 10`, then
  `scripts/55_holm_all_contrasts.py --n-seeds 10` and
  `scripts/56_threshold_all_corpora.py --n-seeds 10`.

Families and counts:

- The Holm family is every row of the credit JSON, including the
  1,000 s ASSIST rows: 44 contrasts at five seeds, 68 at ten seeds
  (24 new fold rows: attention 6, ASSIST corpus configuration 10,
  Junyi 8).
- The five-seed Holm and threshold outputs were computed from the frozen
  files when this section was written; they use no new run.
