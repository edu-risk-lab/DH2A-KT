# Pre-registration addendum, 5 November 2026

This file locks the analyses and runs that answer six advisor comments
dated 5 November 2026 on the unsubmitted Expert Systems manuscript. It
does not change `docs/preregistration.md`,
`docs/preregistration_primary_rule.md`,
`docs/preregistration_compute_and_grids.md`, or
`docs/preregistration_junyi_onset_availability.md`. A change after a
result is seen needs a new dated file.

The Expert Systems venue is unchanged. The ten-seed family-C map remains
a **reanalysis** under the primary rule of
`docs/preregistration_primary_rule.md`, which was fixed after those
ten-seed outcomes existed. Nothing in this file upgrades that map to a
confirmatory claim.

## 0. Already known (not blind)

- Family-C/E/D outcomes under the primary rule
  (`results/tables/credit_primary_test.json`): three family-C credits,
  all XES3G5M LSTM time-package folds; two ASSISTments reference-recipe
  packages change from interval-rule Credited to primary-rule Positive,
  not credited.
- Learner-clustered bootstrap of all 29 family-C contrasts
  (`results/tables/learner_bootstrap.json`, B = 2000, seed 20261023).
- Five-seed leakage simulation under the earlier interval/seed rules
  (`results/tables/leak_simulation_auc.json`).
- T-boundary versus T-zero and aligned gap versus T-boundary on the
  KC-expanded XES3G5M export (family E of the existing ladder).
- History-only XES3G5M injection-site check on five seeds
  (`paper/tables/table_a7_query_dt.tex`).

Sections 1--3 below re-report those quantities in the manuscript. They
are not confirmatory. Sections 4--6 are locked before their results are
computed.

## 1. Language: reanalysis, not confirmation

The manuscript states, in the abstract, Section 4.2, and the conclusion,
that the primary map re-analyses the collected ten-seed runs under a
rule published after the earlier interval/seed map was known. A
confirmatory credit claim would require locking $\tau$, the controls,
the metric, and the family, then testing on data that were not used to
build that rule. The family-C map does not meet that bar.

The two ASSISTments reference-recipe package rows are printed with the
old interval-rule label and the new primary-rule label side by side,
with mean, two-sided LCB, and $k/10$ on the same pooled evaluation AUC
field. The other 27 family-C rows keep the same primary-rule outcome as
the earlier test-interval rule.

## 2. Learner-clustered intervals (existing dumps)

Report the 29-row learner bootstrap already computed: 95% percentile
interval of the mean-over-seeds AUC difference, resampling held-out
learners, same resample for both arms and all seeds. Print seed SD
(from the ten paired $\Delta_s$) beside the learner-bootstrap SD of
that mean. Practical quantities already in the JSON (log loss, Brier,
accuracy at 0.5, share of 0.5-decision flips) are reported as
descriptive, not as credit.

Primary inference stays the ten-seed $t$-tests of
`docs/preregistration_primary_rule.md` Section 4. The learner interval
does not decide credit.

A credit on the LSTM package and a non-credit on the attention package
is **not** a test that the two effects differ. A direct contrast
$\Delta_{\mathrm{LSTM}}-\Delta_{\mathrm{attn}}$ is Section 5, and is
not claimed until that statistic exists.

## 3. Availability table

The manuscript prints, in the data section, a table of each time-related
variable: when it is created, and which decision it may enter.

| Variable | Created | Usable for |
|---|---|---|
| History gap $\log(1+\Delta t)$ on the recurrent input | After item $t$ opens (start-to-start on XES3G5M / ASSISTments) | Scoring or next-item choice before item $t{+}1$ opens |
| Query-side gap on XES3G5M / ASSISTments | After item $t{+}1$ has opened | Scoring the opened item; **not** choosing the next item |
| Junyi completion gap (`time_done`) | After item $t{+}1$ completes | Leakage diagnostic only (family D) |
| Boundary indicator $1[\Delta t{=}0]$ | Same timestamp as the gap it is built from | Same decision as that gap |

The credited XES3G5M LSTM package uses a query-side gap. The paper does
not claim next-item selection. No new history-only ten-seed XES3G5M
ladder is run in this addendum.

## 4. Alert budget and LSTM-minus-attention (GPU dump, no new training)

Locked before dump. Family E (exploratory). Does not enter family C.

On existing XES3G5M fold-0 checkpoints of the zero-incidence time
package and its no-time control (LSTM and attention, ten seeds):

1. Rescore with `scripts/61_learner_bootstrap.py --stage dump --only XES3G5M`
   (and the attention setting of that script).
2. **Matched-budget warning.** Decision: after the scored item has
   opened, flag the $k\%$ of held-out positions with the lowest predicted
   correctness, $k\in\{1,5,10\}$. Compare time-package versus no-time,
   same $k$, mean over the ten seeds: true alerts (flagged and incorrect),
   false alerts (flagged and correct), number of distinct learners with
   at least one flag. This is a practical-significance panel for Expert
   Systems, not a new credit rule and not a JCAL resubmission.
3. **Difference of effects.** Learner bootstrap (B = 2000, seed
   20261023) of
   $(\mathrm{AUC}_{\mathrm{pkg}}-\mathrm{AUC}_{\mathrm{no\text{-}time}})_{\mathrm{LSTM}}
   -(\mathrm{AUC}_{\mathrm{pkg}}-\mathrm{AUC}_{\mathrm{no\text{-}time}})_{\mathrm{attn}}$
   on the intersection of held-out learners scored by all four arms.
   Report the 95% percentile interval. Do not call this a family-C
   credit.

Script: `scripts/66_nov5_dump_analyses.py`.

## 5. Leakage simulation at ten seeds (family D)

Locked before the five new seeds are trained. Family D; never credited.

Extend `leak-sim` from the five seeds $(0,17,42,1234,2024)$ to the ten
seeds of the credit map. Metric and $\tau$ as in the primary rule;
report the one-sided paired $t$ of $H_0{:}\,\mu\le\tau$ and of
$H_0{:}\,\mu\le 0$ on the ten test-only (or pooled evaluation) paired
differences, **without** Holm/BH, because family D does not assign
credit. The abstract, Table `tab:leak-simulation`, and the conclusion
use the same sentence: the five-seed interval/seed label is the earlier
rule; the ten-seed $t$ is the primary-rule calculation applied as a
diagnostic.

## 6. Time versus record structure on one row per attempt (confirmatory for this question only)

Locked before any collapsed LSTM run is launched. Family E. Not added
to the 29 family-C contrasts.

Setting: XES3G5M learner fold 0, LSTM reference recipe, ten seeds
$\{0,17,42,1234,2024,7,99,314,2718,3407\}$. Sequences are collapsed to
one row per attempt (`--collapse-events`; `dh2a_kt.data.events.collapse_to_events`).
This export was not used to choose $\tau$ or the primary rule.

Three arms, same hidden size 128:

- `no-time`: no time branch (0 extra parameters).
- `t-boundary`: time branch on, input $1[\Delta t{=}0]$
  (`--time-gap --time-gap-control boundary`); Linear(1, 128) with bias
  (256 parameters).
- `time-package`: time branch on, input $\log(1+\Delta t)$ aligned
  (`--time-gap --time-gap-mode both`); same 256 parameters.

Contrasts, primary metric pooled held-out evaluation AUC, $\tau={+}0.002$,
$\alpha{=}0.025$ one-sided, no family-C Holm (family E, BH over these
three contrasts only):

- time-package versus no-time (package rung on collapsed rows);
- t-boundary versus no-time (boundary structure plus the branch);
- time-package versus t-boundary (elapsed-time beyond the indicator,
  matched parameter count).

This is the confirmatory test of whether the KC-expanded ~60%
boundary share survives when each attempt is one row. It does not
re-confirm Table `tab:credit-primary-c`.

Group name in `scripts/50_credit_ladder_plan.py`: `xes-attempt`.
Queue: Section 4 dump, then Section 5 five new leak-sim seeds, then
Section 6 thirty runs (3 arms $\times$ 10 seeds). A failed run is
rerun once; a run that fails twice is printed pending.

## 7. Manuscript changes this file allows

- Reanalysis wording; side-by-side old/new table; learner-bootstrap
  table from existing JSON; availability table; abstract / leak-sim
  caption / conclusion aligned on the five-seed interval/seed
  diagnostic.
- After Section 4--6 results exist: alert-budget table, LSTM-minus-
  attention interval, ten-seed leak-sim block, collapsed three-arm
  table. Pending rows are printed pending, never invented.
