# Pre-registration, round 3c: Junyi onset input-availability check (27 Oct 2026)

This addendum fixes one data check and its decision rule before the check
is run. It does not change `docs/preregistration_round3.md` or
`docs/preregistration_round3b.md`. It is committed before
`scripts/63_junyi_onset_leak_check.py` has been run anywhere. Nothing
below may be changed after a result is seen; a change needs a new dated
section.

## 0. Reason and what is already known

Section 1 of round 3 admits an input only if it is available at the onset
of item t+1; the duration and completion time of item t+1 are not. The
round-3b onset gap is built from `time_done(t+1) - time_taken(t+1)`, so
it is admissible only if that difference is the real start of item t+1.

Known when this file is written, and therefore not blind:

- `results/tables/junyi_onset_check.json`: `time_taken` equals the sum of
  `time_taken_attempts` on 99.94% of rows; the rebuilt onset precedes the
  same learner's previous completion on 10.85% of rows with `time_taken`
  and on 3.01% with the first attempt time.
- The training outcomes of every onset run pulled so far: the 50k
  learner groups, folds 0-2, ten seeds each, and the all-learner group,
  fold 0, seeds 17, 42, 1234. The package exceeds T-zero by about
  +65 x 1e-3 validation AUC on every 50k fold and by about +73 x 1e-3 on
  the all-learner fold 0, while T-zero equals no-time.

Not computed when this file is written: every quantity of Section 2.
The concern is that `time_taken` adds the time of attempts made after the
first response to item t+1, so the rebuilt onset of a retried item is
pulled early and the gap fed to the query carries the outcome being
scored.

## 1. Data

- Learner fold 0, training split of the processed Junyi corpus, all
  learners, rows sorted by (user, timestamp, item) as in
  `UserSequenceDataset`.
- Gaps are built with the functions the runs use:
  `junyi_raw_durations(raw, "time_taken")` and
  `attach_junyi_gap_timestamps(..., "onset_s" | "done_s", ...)`, then
  `max(gap_timestamp(t) - gap_timestamp(t-1), 0)` seconds within a learner,
  scored as `log1p`.
- Raw `count_attempts`, `correct`, and the first entry of
  `time_taken_attempts` are joined on (hashed learner, hashed exercise,
  `time_done`), the key the runs use for durations.
- Scored rows: non-first rows of a learner that are not repeat KC-rows.
  The target is `correct` of the same row, which is the target the query
  at that row scores.

## 2. Quantities

Single-feature AUC for `correct`, on the scored rows:

- A_done: completion-time gap (`done_s`), the gap of the `junyi-sec` runs;
- A_onset: onset gap with `time_taken`, the gap of the onset runs;
- A_first: onset gap with the first attempt time (descriptive only);
- `time_taken(t)` and `count_attempts(t) > 1`, which are not available at
  the onset (reference for how much the scored item's own record carries);
- `time_taken(t-1)`, which is available (reference).

Overlap: a scored row with a valid duration whose rebuilt onset precedes
the previous row's completion of the same learner. Reported overall,
split by `count_attempts(t) > 1` versus `= 1`, with the correct rate
inside and outside the overlap, A_onset on non-overlap rows only, the
share of zero onset gaps in each `count_attempts` group, the cross-table
of `correct` against `count_attempts(t) > 1`, and the agreement between
processed and raw `correct`.

## 3. Decision rule

The onset input is declared unavailable at prediction time if either holds:

- **L1.** |A_onset - 0.5| > |A_done - 0.5|, both computed in this check.
  The completion-time gap contains the full duration of the scored item;
  a gap that contains only the previous item's duration and idle time
  should not carry more single-feature information about the scored
  response.
- **L2.** The overlap share among rows with `count_attempts(t) > 1`
  exceeds the share among rows with `count_attempts(t) = 1` by more than
  10 percentage points. Attempts after the first response happen after
  the onset, so an onset that moves with them encodes the outcome.

If neither holds, the onset rows stay in family C as round 3 fixed them,
and the manuscript reports the overlap shares beside them.

## 4. Consequences, fixed now

If the input is declared unavailable:

- Every Junyi onset row (50k folds 0-2, all-learner folds 0-2) leaves
  family C and is reported in family D as a leakage diagnostic, with this
  check. Family C returns to the 29 contrasts round 3 lists without Junyi
  onset.
- The queue stops the groups that read onset gaps:
  `junyi-onset-full-fold{0,1,2}` and the onset arms of
  `attn-junyi-fold{0,1,2}` / `attn-junyi50k-fold{0,1,2}`. Finished runs are
  kept and reported in family D.
- The history-only branch of round 3, Section 8.2 replaces them:
  `junyi-hist-fold{1,2}` and `junyi-hist-full-fold{0,1,2}`, and the Junyi
  attention groups use the history-only arms as round 3b, Section 2
  already provides. A later commit changes
  `scripts/50_credit_ladder_plan.py` to block the onset groups and
  unblock these; it adds no other run.
- No onset rebuilt from the first attempt time is run without a new dated
  section, whatever A_first shows.

The check is run on CPU while the GPU queue continues. Stopping a group is
decided only by Section 3, never by the size of a credit difference.

## 5. Script

`python scripts/63_junyi_onset_leak_check.py` (no training, CPU) writes
`results/tables/junyi_onset_leak_check.json` with every quantity of
Section 2, L1, L2, and the decision. It reads the processed corpus and the
raw log and writes nothing else.
