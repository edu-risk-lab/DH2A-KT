# Pre-registration, round 3b: onset rebuild details, wider scope, matched tuning (23 Oct 2026)

This addendum is the later dated record that Section 9 of
`docs/preregistration_round3.md` requires before runs on more corpora
and before matched tuning. It does not change that file or any earlier
pre-registration. It is committed before any run listed here and before
`scripts/62_junyi_onset_check.py` has been run on the server. Nothing
below may be changed after a result is seen; a change needs a new dated
section.

## 0. Reason and what is already known

Advisor comment 8 (23 Oct 2026) asks for the attention encoder beyond
one corpus, encoders and baselines tuned at a comparable budget, grids
extended where the optimum sits on an edge, and a graph-based model run
under the same protocol. Comment 7 asks for attempt-onset Junyi inputs.

Known when this file is written: everything listed in Section 0 of
round 3, the round-3 reanalysis (`results/tables/credit_primary_test.json`),
and the round-2 baseline grid (`results/tables/baseline_grid_xes3g5m_fold0.csv`;
the simpleKT optimum is at dropout 0.20, the largest value of the grid).
No run below has been launched; the Junyi onset data check has not been
run.

## 1. Junyi onset rebuild (completes round 3, Section 8.2)

- Duration field: the `chosen_field` of
  `results/tables/junyi_onset_check.json` (`time_taken` first, else the
  sum of the `&`-separated `time_taken_attempts`), in seconds.
- Raw rows are matched to processed rows on (learner, exercise,
  `time_done`) with the same hash P0 used for the processed ids. A
  duplicate key takes the median duration.
- A processed row without a valid duration (missing, negative, or not
  matched) takes the completion time of the learner's previous row as its
  onset, and the learner's first row takes its own completion time. That
  gives a gap built only from information available at the onset.
- A run stops with an error if fewer than 99% of its training rows have a
  valid duration after matching (`scripts/03_train_tier1.py`
  `--junyi-time onset_s`). The share is printed in every run log.
- The gap uses a separate `gap_timestamp` column; the row order and the
  repeat flags still come from `time_done`, so the reused no-time arms
  see the same rows in the same order.
- Units: `--junyi-time done_s` (round 3, Section 8.1) is `time_done //
  1e6`; `onset_s` is that minus the rounded duration. Both are integer
  seconds clipped at 0 by the gap transform.

## 2. Attention encoder on ASSISTments and Junyi

- `attn-assist-fold{0,1,2}`: ASSISTments 2012 one-second export
  (`configs/assist2012_sec.yaml`), attention encoder with the reference
  recipe, arms no-time, time-package, t-zero, t-misaligned, ten seeds:
  120 runs.
- `attn-junyi-fold{0,1,2}`: Junyi, all learners, attention encoder with
  the reference recipe, ten seeds. If the onset rebuild runs: arms
  no-time, time-package, t-zero, with onset gaps (90 runs). Otherwise:
  no-time, time-history, t-zero-history (90 runs).
- Timing rule, fixed now: the first Junyi attention run is fold 0, seed
  42, time-package (or time-history). If its wall time times 90 exceeds
  400 GPU-hours, every Junyi attention run uses the first 50,000 learners
  instead (`attn-junyi50k-fold{0,1,2}`, same arms), and the manuscript
  says so. The timing run is not reused across the two sizes.
- Contrasts: time-package versus no-time and aligned gap versus T-zero
  in every setting (aligned gap versus T-misaligned on ASSISTments is
  added beside them).
- Family: E (exploratory). Family C stays as round 3 fixed it (29
  contrasts, 41 with Junyi onset), so its size does not depend on what
  the wider runs show. The manuscript states whether these exploratory
  rows agree in sign and outcome with the family-C rows of the same
  message.

## 3. Matched tuning and the extended grid

These rows go into the contextual model table only and never into the
credit map.

- Tracer (DH2-KT reference, XES3G5M fold 0, time-package arm with the
  reference recipe): learning rate {1e-3, 3e-4} x dropout {0.1, 0.2,
  0.3} x hidden size {128, 256}, 12 configurations on seed 42
  (`tracer-grid`); the configuration with the highest selection-split
  AUC is trained on seeds 0, 17, 1234, 2024 (`tracer-seeds`). Same
  learning-rate and width axes and the same size (12) as the round-2
  AKT and simpleKT grids; dropout values differ because the defaults
  differ.
- simpleKT: dropout {0.30, 0.40} added to the round-2 grid at both
  learning rates and both widths (8 runs, seed 42,
  `scripts/57_baseline_grid.py --stage grid-ext`). Selection over the
  union of the 20 configurations by validation AUC. If the choice moves,
  the new configuration is trained on the five seeds (`--stage
  seeds-ext`) and replaces the round-2 simpleKT row; the old row moves to
  the appendix. If the new optimum is again at dropout 0.40, the
  manuscript says the grid edge was not resolved; no further extension.
- DyGKT (graph-based, 2024): XES3G5M fold 0 through the vendored pyKT
  evaluator, chunked windows, mask-repeats, L = 400, the train-only
  prerequisite and similarity graph of P0, vendored default
  hyperparameters (no grid; the table says so), five seeds
  (`--stage dygkt`).
- sparseKT and stableKT stay blocked: the vendored evaluator has no
  forward for them. The manuscript lists them as not run, with the
  reason.
- Test AUC is written for every tuning run and never read for selection.

## 4. Compute accounting

`scripts/50_credit_ladder_plan.py` now writes the start time and wall
seconds into each new `.ok` stamp. For earlier runs, GPU-hours are taken
from the modification times of the `.log` (end) and the first log line
or, when missing, the previous run's end on the same queue; the
manuscript states that these are wall-clock hours on one GPU.

## 5. Queue

1. `python scripts/62_junyi_onset_check.py`; commit
   `results/tables/junyi_onset_check.json`.
2. `junyi-sec` (20).
3. Junyi onset groups (120) or the history-only fallback (100).
4. `attn-assist-fold{0,1,2}` (120); Junyi attention timing run, then the
   chosen Junyi attention groups (90).
5. simpleKT `grid-ext` (8) and `seeds-ext` (0 or 5).
6. `tracer-grid` (12), then `tracer-seeds` (4).
7. DyGKT (5).

A failed run is rerun once with the same command; a run that fails twice
is printed pending. No other run is added without a new dated section;
in particular no seeds beyond ten (round 2) unless a reviewer asks.
