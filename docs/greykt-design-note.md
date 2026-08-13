# GreyKT: Reliability-Aware Hybrid Knowledge Tracing

Status: **v3 implemented. Frequency C_B premise NOT_SUPPORTED on
XES3G5M fold 0** (ρ = −0.032, C_B saturated at ~0.997–1.0; artefact
`results/tables/xes3g5m_fold0_black_confidence_diagnostic.json`). Wrap/train
correctly refused. Next: MC-dropout C_B (`--signal mc_dropout`); see
`docs/greykt-gpu-runbook.md`. Do not put GreyKT numbers in `paper/main.tex`.

## Read this first: the premise has not been tested

Three consecutive design rounds (v1 -> v2 -> v3) refined this model
without a single run on real data. That is the largest risk in this work,
and it is not a risk more architecture can reduce.

The reliability gate rests on one assumption:
`C_B(c) = N_c^train / (N_c^train + kappa_B)` -- how often a concept
appeared in training -- predicts how unreliable the black-box branch is
on that concept. **If that assumption is false, the gate is steering on
noise, and every refinement layered on top of it (v4a...v4e) is wasted
effort.**

`dh2a_kt/diagnostics/black_confidence.py` tests it directly, and needs
only the existing black-box model -- no GreyKT training run:

```
python scripts/21_diagnose_black_confidence.py configs/xes3g5m.yaml --fold 0 --device cuda
```

The verdict is one of SUPPORTED / AMBIGUOUS / NOT_SUPPORTED, decided on
the rank correlation between C_B and per-prediction squared error, plus
the Brier trend across C_B buckets. Three deliberate methodological
choices are documented in that module and worth restating:

- **The verdict does not rest on AUC.** AUC is a within-bucket ranking
  metric, so it is not comparable across buckets. In the synthetic
  validation of the diagnostic, AUC is exactly 1.0 in *every* bucket
  while Brier varies by ~100x -- an AUC-only analysis would have reported
  "no difference" and missed the entire effect.
- **Significance is not enough.** At 10^5+ predictions a correlation of
  0.01 is significant and meaningless, so SUPPORTED additionally requires
  `|rho| >= 0.05` (configurable).
- **Base rate is a confound.** Rare concepts may be intrinsically harder
  rather than under-trained. The diagnostic flags this when per-bucket
  base rates diverge; separating the two needs a difficulty-matched
  comparison it does not itself perform.

If the verdict is NOT_SUPPORTED, the fix is a different black-box
confidence signal -- MC-dropout variance or ensemble disagreement -- not
another refinement of the gate arithmetic. **Frequency C_B is that
NOT_SUPPORTED case on XES3G5M fold 0.** The MC-dropout diagnostic is a
separate JSON (`--signal mc_dropout`) so the frequency artefact stays as
the empirical warrant for abandoning N_c^train.

## v4a / v4b (implemented, OFF by default)

Both are config-gated so v3 stays exactly reproducible and each increment
can be measured on its own. The ablation ladder is v3 / v4a / v4b /
v4a+b, and the intended selection rule is **the smallest version that
shows a clear benefit** -- an increment that does not earn its complexity
on validation data should not ship.

**v4a -- black-box probability calibration** (`GreyKTConfig.temperature`).
Probability-space fusion assumes `p_B` and `p_W` are on a comparable
scale, and by default they are not: neural predictors are typically
overconfident while the Beta-Binomial white-box estimator is conservative
by construction, so `p_B = 0.9` and `p_W = 0.9` do not carry equal
evidential weight and averaging them is not meaningful. Temperature
scaling (`p_B = sigmoid(logits / T)`, fit post-hoc on validation with
`fit_temperature()`) corrects this with one monotone parameter. Because
it is monotone, **AUC is provably unchanged** -- only NLL/Brier/ECE move
-- so it can be applied without re-validating discrimination. Fit on
validation only: a temperature fitted on training data would be far too
small, since the model is most overconfident exactly where it has
memorized.

**v4b -- absolute-reliability backoff**
(`GreyKTConfig.use_absolute_backoff`). The v3 gate is scale-free, so
`C_B = C_W = 0.9` (both branches reliable) and `C_B = C_W = 0.05`
(neither branch informed) both produce `g_B = 0.5`. Those two situations
mean opposite things and v3 cannot distinguish them. v4b adds an
orthogonal absolute weight:

```
R     = C_B + C_W
alpha = R / (R + kappa_r)
p     = alpha * [g_B*p_B + (1-g_B)*p_W] + (1 - alpha) * backoff_prob
```

`g_B` still decides *how* to split between branches; `alpha` decides *how
much* to trust that split at all versus falling back to the base rate.
Both `branch_mixture` (pre-backoff) and `absolute_reliability` (alpha)
are exposed on `GreyKTOutput` so the backoff's contribution is separately
measurable.

**Deliberately not implemented: v4c (hierarchical per-concept prior),
v4d (graph-stability edge weighting), v4e (posterior-variance C_W).**
Each targets a real weakness, but the severity of each is unknown until
GreyKT runs on data even once. Building them now would be a fourth
consecutive design round with zero empirical results. Two of them are
worth flagging as likely-important once data exists:

- v4c is probably *more* urgent than its "medium" ranking suggests: a
  single global `prior_mean` dominates exactly where `N_eff` is small,
  which is the low-`C_B`/high-`C_W` cold-start cell the whole design is
  built to win. A per-concept base rate is a cheap fix, fully consistent
  with the existing "fixed statistic, not learned" convention.
- v4d is grounded in a number already in the paper: Table 7 reports Edge
  F1 = 0.183 between P0's `E_pre` and the Junyi ground-truth structure.
  The white-box branch reads that graph directly, so the reliability of
  the graph itself bounds the reliability of the branch. This deserves a
  higher severity rating than "high" in any weakness triage.

## v3 revision (relative-reliability gate + probability-space fusion)

v2's gate combined a white-box confidence signal with an optional
black-box confidence signal via `min()` -- a floor, not a comparison, and
fusion was done in logit space. Two changes fix both issues:

1. **Relative-reliability gate.** Both confidences are now first-class,
   explicitly computed quantities, and the gate is their ratio:
   `g_B = C_B / (C_B + C_W + eps)`. This is the standard precision-weighted
   (inverse-variance-style) fusion rule from Bayesian model averaging: when
   both confidences are low, the gate lands near 0.5 (genuine shared
   uncertainty) instead of collapsing to whichever branch happens to be
   weaker, which is what `min()` does. `C_B(c) = N_c^train / (N_c^train +
   kappa_b)` is now always computed (not optional) from a per-concept
   training-frequency table; if the table is omitted, `C_B` defaults to 0
   everywhere -- a conservative default that pushes the gate toward the
   white-box branch rather than silently assuming the black-box is
   trustworthy.
2. **Probability-space fusion, not logit-space.** v2 fused
   `gate * black_logit + (1-gate) * white_logit`, a geometric mixture of
   the two branches' probabilities -- at `gate=0.5` this was *not* a
   50/50 blend of probabilities, undercutting the interpretability the
   gate is supposed to provide. Since the gate is deliberately
   non-learned, there is no gradient-flow argument for logit-space
   combination. v3 fuses `p_GreyKT = g_B * p_B + (1-g_B) * p_W`, an
   arithmetic mixture (standard mixture-of-experts form) where the gate
   value *is* the probability weight, directly.
3. **White-box aggregation is now per-prerequisite, not pooled.** v2
   pooled raw sufficient statistics (`S_eff`, `N_eff`) across all
   prerequisites *before* applying Beta-Binomial shrinkage once. v3 computes
   a Beta-Binomial posterior mastery `M_p` and confidence `C_p` for *each*
   prerequisite individually, then combines across prerequisites via a
   `C_p`-and-hop-decay-weighted average:
   `p_W = sum_p(w_pc * C_p * M_p) / sum_p(w_pc * C_p)`. This keeps each
   prerequisite's own contribution inspectable as an intermediate
   quantity, which matters given the whole justification for the
   white-box branch is auditability -- a single pooled estimate cannot be
   attributed back to individual prerequisites as cleanly.
4. **Diagnostics extended**: added `negative_log_likelihood()` and
   `brier_score()` alongside the existing `expected_calibration_error()`,
   and replaced the 1D `stratified_auc_by_confidence()` (single combined
   gate bucket) with `stratified_metrics_2d()`, which strata jointly
   along *both* confidence axes -- `C_B` (concept_train_support) and `C_W`
   (prerequisite_evidence) -- reporting AUC/NLL/Brier per cell. A 1D
   stratification by gate alone cannot distinguish "both confidences low"
   (genuine shared uncertainty) from "one low, one high" (the asymmetric
   cold-start case the design is meant to handle); the 2D grid keeps that
   distinction visible.

Two design decisions filled in gaps not fully pinned down when this
revision was proposed, recorded here for traceability:

- The aggregate `C_W` used inside the gate is defined as the pooled
  hop-weighted effective count run through the same `N/(N+kappa)` shape
  used everywhere else: `C_W = sum_p(w_pc * N_eff_p) / (sum_p(w_pc *
  N_eff_p) + kappa_w)`, rather than (e.g.) an unweighted average of the
  per-prerequisite `C_p` values.
- `kappa_w` (white-box confidence pseudo-count) is decoupled from
  `prior_strength` (`alpha0 + beta0`, the Beta-shrinkage pseudo-count) in
  config, defaulting to `prior_strength` when left unset
  (`GreyKTConfig.resolved_kappa_w()`), but independently tunable.

`GreyKTOutput.logits` is still provided (via a numerically-clamped
inverse-sigmoid of the fused probability) purely for compatibility with
training code written against `BCEWithLogitsLoss`; new training code
should prefer `BCELoss` (or an equivalent) directly on
`GreyKTOutput.probs`.

## v2 revision (fixed weaknesses found in the v1 design review)

v1's white-box branch was deliberately simple (raw empirical mean,
binary coverage, direct-prerequisites-only, flat 0.5 prior) to get an
end-to-end skeleton working quickly. Review surfaced eight weaknesses;
v2 fixed each one (superseded in places by v3 above, noted where
relevant):

1. **Overconfidence from sparse evidence** -> Beta-Binomial shrinkage
   toward a data-derived `prior_mean`. Still true in v3.
2. **Binary coverage jumping straight to "fully trust the black box"
   after one observation** -> continuous confidence. In v3 this became
   the `C_W`/`C_B` pair feeding the relative-reliability gate rather than
   a single confidence value.
3. **Direct-prerequisites-only vs. the black-box branch's bounded-length
   chain hyperedges** -> multi-hop traversal up to `max_hops`, with
   `hop_decay` discounting distant ancestors. Unchanged in v3.
4. **No recency weighting** -> `recency_decay` exponentially discounts
   older running per-concept evidence. Unchanged in v3.
5. **Flat 0.5 prior regardless of the actual data** -> `prior_mean` is a
   config value the caller must set from the training set's real base
   rate before trusting any number. Unchanged in v3.
6. **Gate only captures student-level cold start, not model-level** ->
   v2 added an optional `min()`-combined secondary signal; v3 promotes
   this to the primary gate mechanism (relative-reliability ratio, not a
   floor) -- see v3 point 1 above.
7. **Unvectorized double loop** -> the per-timestep loop is still present
   (the recency recurrence is inherently sequential) but the inner
   per-batch-sample loop is vectorized across the batch dimension.
   Unchanged in v3.
8. **No way to check calibration or test the central hypothesis** ->
   v2 shipped ECE + 1D stratified AUC; v3 adds NLL, Brier, and upgrades
   to 2D stratification -- see v3 point 4 above.

None of these are learned (backprop-trained) parameters; the white-box
branch remains zero-learned-parameter, preserving the auditability
property this design is chosen for.

## Why this name

"Grey-box" is the standard term (control theory, and increasingly ML) for
a model that is neither a pure black box (opaque, e.g. a plain neural
net) nor a pure white box (fully symbolic/rule-based), but an explicit
combination of both, where the combination itself is inspectable. That is
exactly what this module is: a black-box branch (DH2-KT, unchanged) fused
with a white-box branch (a closed-form, zero-parameter prerequisite-mastery
rule read directly off the audited E_pre graph) through a gate that is
itself non-learned and traceable. The "-v3" suffix and "Reliability-Aware"
qualifier were added to make the gate's mechanism (an explicit reliability
comparison between branches, not just "a gate") legible from the name
itself.

Alternatives considered: `NeSy-KT` (ties to the "neuro-symbolic"
terminology used by the closest related paper, Hooshyar et al. 2026,
arXiv:2604.08263, "Responsible-DKT") -- more immediately legible to a
KBS reviewer who knows the term, but "NeSy" is used broadly across many
domains and carries less distinct branding. `DH3-KT` -- keeps continuity
with the existing DH2-KT/DH2A-KT lineage, but the reinterpretation of the
exponent (diagnostic-heterogeneous -> dual-process) needs an explanation
every time it's used. `GreyKT` was chosen as the primary name for its
directness; renaming is a one-file change (`dh2a_kt/models/greykt.py`) if
the authors prefer one of the alternatives after discussion with GS Son.

## Positioning vs. the closest related work

Hooshyar et al. 2026 (Responsible-DKT, arXiv:2604.08263) injects symbolic
mastery/non-mastery rules *into* a sequential neural model's computation
graph and reports both higher AUC (up to 0.90, +13% vs a plain DKT) and
intrinsic interpretability via a "grounded computation graph." GreyKT
differs in mechanism: rather than injecting rules into the black-box's
internal computation, GreyKT keeps the two branches architecturally
separate and fuses their *outputs* through an explicit, inspectable gate.
This is a weaker form of integration than Responsible-DKT's but a
stronger auditability property: every GreyKT prediction can be decomposed
post-hoc into exactly "how much black-box, how much white-box, and why"
(the gate value *is* the why, and in v3 the gate itself decomposes into
`C_B` and `C_W`), which is not true of a model where symbolic rules are
baked into learned weights. The paper should state this trade-off
explicitly rather than imply GreyKT is a strict improvement on
Responsible-DKT's mechanism -- it is a different point in the same design
space, chosen for auditability.

## Architecture

- **Black-box branch**: `dh2a_kt.models.dh2_kt.DH2KT`, reused unmodified
  (composed as `GreyKT.black_box`, not reimplemented) so an existing
  Tier-1 checkpoint loads directly. Its probability `p_B = sigmoid(black_logits)`.
- **White-box branch**: `whitebox_branch()` -- for the current timestep's
  target concept `c`, walks up to `max_hops` ancestors in P0's audited
  E_pre graph (not the chain hyperedges the black-box branch consumes).
  For each ancestor `p`, forms a per-prerequisite Beta-Binomial posterior
  mastery `M_p` and confidence `C_p` from the student's own recency-decayed
  evidence on `p` (strictly before the current timestep -- no leakage).
  Combines across prerequisites via a `C_p`-and-hop-decay-weighted
  average into `p_W`, and separately aggregates a pooled confidence `C_W`.
  Zero learned (backprop-trained) parameters.
- **Black-box confidence**: `black_confidence_table()` -- `C_B(c) =
  N_c^train / (N_c^train + kappa_b)`, a fixed statistic computed once from
  the training corpus's per-concept frequency, not learned.
- **Gate**: `reliability_gate()` -- `g_B = C_B / (C_B + C_W + eps)`, the
  fraction of combined confidence attributable to the black-box branch.
  Non-learned, by design -- see the module docstring for why keeping the
  gate non-learned matters for the auditability story.
- **Fusion**: probability-space convex combination,
  `p_GreyKT = g_B * p_B + (1 - g_B) * p_W`.
- **Diagnostics**: `expected_calibration_error()`, `negative_log_likelihood()`,
  `brier_score()`, and `stratified_metrics_2d()` (joint `C_B` x `C_W`
  grid), for checking fused-branch calibration and testing the cold-start
  hypothesis directly once real data is wired in.

## What is verified vs. not

Verified (`scripts/_greykt_logic_check.py`, torch-free numpy
re-implementation of the v3 algorithm): no-prerequisite/no-train-freq
fallback to prior probability and zero confidences; Beta-Binomial
shrinkage after a single observation (posterior mean strictly between
prior and raw empirical rate); a two-prerequisite case with symmetric
contradictory evidence canceling to exactly the prior, with nonzero
confidence; multi-hop reachability bounded correctly by `max_hops`; the
black-confidence table formula; the reliability gate favoring the more
confident branch, landing at exactly 0.5 when both confidences are equal
(the property a `min()`-based gate lacks), and not producing NaN when
both confidences are zero; and probability-space fusion arithmetic at
gate=0/0.5/1 (confirming the fused value at gate=0.5 is the arithmetic
mean of the two branch probabilities, unlike v2's logit-space fusion).
All 20 checks pass.

**Not verified**: the actual PyTorch module
(`dh2a_kt/models/greykt.py`) and its test suite
(`tests/test_greykt.py`) could not be executed in the authoring sandbox
-- `pip install torch` timed out repeatedly (network-restricted sandbox,
full CUDA wheel is several GB) and `torch_geometric` was never reached.
**Run `pytest tests/test_greykt.py -v` on a machine with torch +
torch_geometric (the RTX 3090 host) before trusting any GreyKT number.**
The test file mirrors `tests/test_dh2_kt.py`'s conventions (shape checks,
an overfit sanity check using `BCELoss` on the fused probability) plus
v3-specific checks: per-prerequisite Beta-Binomial shrinkage, multi-hop
reachability, the black-confidence table, the relative-reliability gate's
symmetry property, the no-evidence fallback, the NLL/Brier/ECE/2D-grid
diagnostics, and that a plain-DH2KT checkpoint's `state_dict()` loads
into `GreyKT.black_box` without error.

## Known implementation limitation

`whitebox_branch()` still loops over timesteps in Python (the
recency-decay recurrence is inherently sequential), with a further inner
loop over batch samples (needed because each sample's target concept has
a different, variable-length ancestor set, which resists a single
vectorized gather). Fine for the existing training budget (`batch=4`)
used elsewhere in this repo; will be a bottleneck if run densely over the
full 6.4M-interaction XES3G5M training set without either (a)
restricting to a subsample per epoch, or (b) a follow-up vectorization
pass (e.g. padding ancestor sets to a fixed max-degree and using masked
batched gathers). Flagging now so it isn't discovered mid-training-run on
the GPU host.

## Next steps

Plumbing for wrap/train is in `scripts/22_run_greykt.py` (see
`docs/greykt-gpu-runbook.md`). `max_hops` inherits
`hyperedge.concept_prerequisite.max_chain_len` (8 on XES3G5M). Do **not**
put GreyKT numbers in `paper/main.tex` until steps 0 and 4 have real
outputs.

0. ~~Frequency C_B diagnostic.~~ **NOT_SUPPORTED** on XES3G5M fold 0
   (ρ = −0.032, C_B ~0.997–1.0). Do not wrap/train on this signal.
0b. **MC-dropout diagnostic** (graph-encoder dropout; eval-mode p_B for
   error). Separate JSON; does not overwrite the frequency artefact:

   ```
   python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode diagnose \
     --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal mc_dropout
   ```

   If this is also NOT_SUPPORTED (or std is saturated), the next signal is
   a multi-seed ensemble, not `--force`.
1. `pytest tests/test_greykt.py tests/test_greykt_inputs.py tests/test_greykt_plumbing.py tests/test_black_confidence.py -v`
2. ~~Wire `prereq_edge_index` / `concept_train_freq` / `prior_mean`.~~ Done.
   Per-prediction `GreyKTBatch.black_confidence` is now the MC-dropout path.
3. ~~Set `max_hops` = chain `ell_max`.~~ Done.
4. Wrap/train **only if** the MC diagnostic is not NOT_SUPPORTED, with
   `--signal mc_dropout`. Test the low-C_B / high-C_W cell.
5. Ablation ladder v3 / v4a / v4b / v4a+b after a working C_B.
6. Only after 4–5 have real results: 3-fold, manipulation check, and
   whether GreyKT belongs in the main paper.
