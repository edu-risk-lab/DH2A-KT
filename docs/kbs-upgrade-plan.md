# KBS upgrade plan — closing the gap to Ni et al. (2026), KBS 344:116071

Context: target journal changed from IEEE TLT to Knowledge-Based Systems
(Elsevier, Q1). `paper/main.tex` and `paper/cover_letter_kbs.md` have been
converted/rewritten already. This document is the experimental roadmap for
the remaining evidence gap relative to Ni et al. (interpretable KT + LLM
explanation).

**Implementation status (2026-08-09):** Items 1–2 harness code is in-repo
(`dh2a_kt/eval/explanation_faithfulness.py`, Diagnostician `GroundedKCs`,
`--ablation ig` / `--skip-critic`, `scripts/18_run_kbs_faithfulness_suite.py`).
Stub harness comparison written under `results/tables/synthetic_fold0_*`.
Full Ollama $n{=}500$ + manipulation folds 1–2: see
`docs/kbs-gpu-runbook.md` (blocked on this checkout: no checkpoint / E_pre /
Ollama). Author confirmation gate: `docs/kbs-author-gate.md`.

**Corrected assumptions (2026-08-09 audit):**
- `diagnostician_explanation` lives on `PilotRecord` / pilot JSONL, **not**
  on `Tier2EvalReport` (`dh2a_kt/tier2/eval.py`).
- XES3G5M has **no** human-readable KC name/synonym table in-repo (Junyi
  does). Faithfulness uses **ID-anchored KC-Jaccard**, not synonym-aware
  matching identical to Ni et al.
- `dh2a_kt/eval/tier1_eval.py` is the **manipulation-check** driver, not an
  AUC ablation matrix.
- Headline XES3G5M 3-fold AUCs are **already** in
  `paper/tables/table_auc_xes3g5m.tex` (0.7501 / 0.7518 / 0.7531). Remaining
  Tier-1 gap is mainly manipulation check on folds 1–2.
- Diagnostician conditions on frozen `P(correct)` + history IDs only (no
  latent concept-state tensor in the prompt). IG = omit `P(correct)` from
  the Diagnostician prompt; Critic still sees the true Tier-1 score.
- LLM model tag is already CLI-parameterized (`--ollama-model` on
  `scripts/04_run_tier2_pilot.py`).
- Pilot JSONL under `results/tables/` is gitignored; restore from the GPU
  machine or re-run before corpus-wide JC.

## 1. Automatic, corpus-wide explanation-faithfulness metric

**Gap.** Tier 2 faithfulness is currently only measured by (a) Critic flag
rate and (b) a 40-sample human dual-rater study. There is no metric over
the full 500-sample pilot.

**What to build (chosen design).** ID-anchored KC-Jaccard:
1. Diagnostician emits a structured trailer `GroundedKCs: [id, ...]`.
2. Support set \(S\) = `{target concept_id}` ∪ KCs in recent history ∪
   1-hop neighbors on audited train-only \(E_{pre}\).
3. \(JC = |G \cap S| / |G \cup S|\) with \(G\) from the trailer (fallback:
   regex `kc \d+` on free text for older JSONL).
4. Paper wording: *ID-anchored KC-Jaccard*, citing Ni 2026 as methodological
   precedent; do **not** claim synonym-aware matching on XES.

**Where it plugs in.**
- `dh2a_kt/agents/diagnostician.py` — structured `GroundedKCs` trailer
- `dh2a_kt/eval/explanation_faithfulness.py` — extract / support set / JC
- `scripts/07_eval_tier2_pilot.py` — report mean JC next to flag rate
- Pilot JSONL: `results/tables/{dataset}_fold{f}_tier2_pilot*.jsonl`

**Effort.** Low–medium (prompt change + post-hoc metric + unit tests).

## 2. "Independent generation" negative control for Tier 2

**Gap.** No experiment isolating whether the Diagnostician's rationale is
grounded in frozen Tier-1 `P(correct)`, versus a fluent LLM guess over
history alone.

**What to build.** Re-run the same 500-sample pilot with Diagnostician
prompt stripped of `predicted_correct_prob` (and unused causal line); keep
history. Critic still receives true Tier-1 `P(correct)`. Compare mean JC
and flag rate: grounded vs IG (`--ablation ig`).

**Where it plugs in.**
- `scripts/04_run_tier2_pilot.py --ablation ig`
- `dh2a_kt/tier2/pilot.py::run_tier2_pilot(..., ablation=...)`
- `dh2a_kt/agents/diagnostician.py::explain(..., omit_tier1_prob=True)`

**Effort.** Medium — one more 500-sample Ollama pilot.

## 3. Paired predictive + faithfulness ablation table

**Gap.** Need AUC (Tier 1) and/or JC + flag rate (Tier 2) side by side per
ablated component, in the spirit of Ni et al. Tables 2–3.

**What to build.** Assemble from real knobs (do **not** invent a
`tier1_eval` ablation matrix):
- w/o \(L_{aux}\): `training.graph_sensitivity_weight: 0`
- w/o Critic: `--skip-critic` on the Tier-2 pilot
- session HE: existing `scripts/16_train_session_ablation.py` (FoundationalASSIST)
- w/o leakage-audited \(E_{pre}\): only if an unaudited graph artifact exists;
  otherwise omit from submission

**Effort.** Low once items 1–2 exist (table assembly + selective re-trains).

## 4. LLM-scale sensitivity sweep for the Critic / Diagnostician

**Gap.** Single model (Qwen2.5-7B) with no scale evidence.

**What to build.** Re-run n≈100 with `--ollama-model qwen2.5:1.5b` /
`qwen2.5:7b` / `qwen2.5:14b` and `--output-tag scale_*`. Report flag rate
+ mean JC vs scale. No new backend code required.

**Effort.** Medium (local Ollama; reduced sample size).

## 5. Multi-dataset breadth for the Tier-1 quantitative core

**Gap.** Credibility vs Ni's 3 datasets × 5-fold × 13 baselines. Do not
match that bar in one cycle; narrow the gap.

**Priority:**
1. **Done for AUC:** all 3 XES3G5M folds already reported in
   `paper/tables/table_auc_xes3g5m.tex`.
2. **Still TODO:** manipulation check on folds 1–2 (fold 0 only today).
3. FoundationalASSIST: keep fold-0 + session ablation; folds 1–2 only if
   budget remains.
4. Fourth public KT benchmark: response-to-reviewers contingency only.

**Effort.** Medium for manipulation folds 1–2; high for ASSIST multi-fold.

## Suggested order of execution

1. Item 1 (ID-anchored KC-Jaccard harness) — unlocks faithfulness tables.
2. Item 2 (IG negative control) — one more pilot run.
3. Item 5.2-style: manipulation check folds 1–2 (Tier 1).
4. Item 3 (paired table assembly).
5. Item 4 (LLM-scale sweep) if time allows.
6. FoundationalASSIST multi-fold / new dataset — last / contingency.

## Author gate (before KBS submission)

- Confirm CRediT roles with all co-authors (currently inferred placeholders
  in `paper/main.tex`).
- Confirm Data availability wording / public repo URL.
- Replace cover-letter "Planned strengthening" once JC + IG numbers exist.
