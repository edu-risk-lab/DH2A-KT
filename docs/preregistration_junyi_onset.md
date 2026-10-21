# Pre-registration addendum: Junyi history-only gap check (21 Oct 2026)

This addendum fixes one additional run group before any of its runs. It
does not change `docs/preregistration.md`.

## 1. Reason

Junyi is loaded with `timestamp: time_done`
(`external/p0_leakage_audit/configs/junyi.yaml`), a completion time.
`dh2a_kt.data.aux_signals.log_time_gaps` takes the gap of row t+1 as
`time_done(t+1) - time_done(t)`, which includes the time the learner spent
on item t+1. With `--time-gap-mode both` that gap enters the next-step
query, so the prediction for item t+1 sees how long the learner took on
it. On XES3G5M and ASSISTments 2012 the timestamps are start times and
the query-side gap has elapsed when the item opens; on Junyi it has not.

The recurrent-side gap at step t, `time_done(t) - time_done(t-1)`,
covers only item t, whose response is already an input at step t. The
history-only mode (`--time-gap-mode lstm`) therefore has no look-ahead
on any corpus.

## 2. Runs

Group `junyi-hist` of `scripts/50_credit_ladder_plan.py`: Junyi, first
50,000 learners (`--max-users 50000`), learner fold 0, corpus
configuration (the flags of group `junyi-partial`), the ten seeds
0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407.

| Arm | Flags added to the `junyi-partial` no-time arm |
|---|---|
| `time-history` | `--time-gap --time-gap-mode lstm` |
| `t-zero-history` | `--time-gap --time-gap-mode lstm --time-gap-control zero` |

20 runs. The no-time arm is the existing `junyi-partial_no-time` run of
the same seed.

## 3. Contrasts and rules

1. History-only package: `time-history` minus `junyi-partial_no-time`.
2. History-only aligned gap: `time-history` minus `t-zero-history`.

Same threshold (+0.002), both rules, interval rule primary, validation
AUC for the outcome, test-only AUC reported beside it.

## 4. What the paper says for each result

- If contrast 2 is credited under the interval rule, the paper states
  that the Junyi gap-input credit survives without the query-side gap,
  so it does not rest on the scored item's own duration.
- If contrast 2 is not credited, the paper states that the Junyi
  gap-input credit cannot be separated from that look-ahead, and the
  Junyi rows of the map are reported as completion-time results only.
- Either way, the outcome of contrast 1 is reported, and the five-seed
  XES3G5M history-only result (Table of the injection-site diagnostic)
  is cited as context, not as a paired comparison.

## 5. Scoring

`python scripts/59_junyi_history_check.py` reads the per-seed CSVs and
writes `results/tables/junyi_history_check.json` and
`paper/tables/table_junyi_history.tex`.
