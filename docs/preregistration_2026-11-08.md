# Pre-registration addendum, 8 November 2026

This file locks the scoring of the already-specified Junyi unit check
and the manuscript sentences that report it. It does not change
`docs/preregistration.md`,
`docs/preregistration_primary_rule.md`,
`docs/preregistration_compute_and_grids.md`,
`docs/preregistration_junyi_onset_availability.md`, or
`docs/preregistration_2026-11-05.md`. A change after a result is seen
needs a new dated file.

The Expert Systems venue is unchanged. Family C remains 29 contrasts
and a **reanalysis**. Nothing here upgrades that map to a confirmatory
claim, adds a contrast to family C, or relaunches Junyi onset.

## 0. Already known (not blind)

- Family-C/E/D outcomes under the primary rule, including the
  attention-encoder Junyi history-only rows
  (`results/tables/credit_primary_test.json`).
- The Junyi onset availability audit failed L2; onset groups stay
  blocked (`docs/preregistration_junyi_onset_availability.md`).
- Group `junyi-sec` (20 runs: `time-package` and `t-zero`, ten seeds,
  first 50,000 learners, fold 0, corpus configuration, `--junyi-time
  done_s`) was locked in `docs/preregistration_primary_rule.md`
  Section 8.1. All twenty `.ok` stamps and CSVs are on disk. This
  addendum does **not** retrain them.
- Local launch on 8 November 2026 saw an NVIDIA GeForce MX250 (2 GB)
  and is refused by `scripts/50_credit_ladder_plan.py` (need 20,000 MiB
  free). No new AUC is written from that device.

## 1. Scoring locked in Section 8.1 (no new training)

Script: `python scripts/67_junyi_sec_check.py`, then
`python scripts/60_credit_primary.py`.

Metric, $\tau={+}0.002$, $\alpha=0.025$ one-sided, ten seeds
$\{0,17,42,1234,2024,7,99,314,2718,3407\}$, pooled held-out evaluation
AUC (`dh2_kt_auc`), as in the primary rule.

Microsecond controls are the existing `junyi-partial` arms (same 50k
learners, fold 0, corpus configuration). No-time does not read
timestamps; `junyi-partial_no-time` is reused.

| Contrast | On | Control | Family |
|---|---|---|---|
| Seconds minus microseconds, package | `junyi-sec_time-package` | `junyi-partial_time-package` | E (unit check) |
| Seconds minus microseconds, T-zero | `junyi-sec_t-zero` | `junyi-partial_t-zero` | E (unit check) |
| Package vs.\ no-time, seconds | `junyi-sec_time-package` | `junyi-partial_no-time` | D (completion-time map) |
| Aligned gap vs.\ T-zero, seconds | `junyi-sec_time-package` | `junyi-sec_t-zero` | D (completion-time map) |

Family C is not enlarged. The two unit-check rows join family E as
Section 8.1 specified; Benjamini--Hochberg is recomputed over that
enlarged family. The two seconds map rows are family D and never
credited.

The unit-check *display* in the manuscript is the mean
seconds-minus-microseconds difference with its two-sided 95% interval
(Section 8.1). The family-E tests of $H_0{:}\,\mu\le\tau$ and
$H_0{:}\,\mu\le 0$ are recorded in `credit_primary_test.json` but are
not the invariance claim: invariance is a small interval around zero,
not a lift above $\tau$.

## 2. Not run in this addendum

- Junyi onset groups (blocked by L2).
- `attn-junyi50k`.
- DyGKT, sparseKT, tracer-grid through the ladder.
- Any new family-C contrast.

## 3. Manuscript changes this file allows

After Section 1 numbers exist: a table of the four contrasts; replace
the Limitations sentence that the seconds unit is untested; mention the
unit check in the Junyi results paragraph, protocol family list, Data
Availability family-E size, and cover letter. Do not invent pending
rows. Do not move seconds map contrasts into family C.
