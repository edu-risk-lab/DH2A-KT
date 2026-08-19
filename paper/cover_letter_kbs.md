# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Construction, Time-Aware Tracing, and Critic-Gated Explanations

**Journal:** Knowledge-Based Systems

**Dear Editors / Reviewers,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is a two-tier *knowledge system* for knowledge tracing (KT): it constructs and audits graph-structured auxiliary knowledge, measures which of that knowledge actually predicts next-step correctness, and then explains frozen scores with a Critic-gated agent layer. KBS has recently published on interpretable KT (Ni et al., “Interpretable knowledge tracing via explicit–implicit alignment,” *Knowledge-Based Systems* 344 (2026) 116071). We take that as evidence of scope fit. DH²A-KT complements that line on **faithfulness of explanations**, not by competing with Ni et al. on AUC.

Three hooks:

1. **Hyperedge leakage audit.** Companion paper P0 audits pairwise prerequisite/similarity edges under train-only construction. We extend the audit to hyperedges (group-membership, cross-modal, and intervention-timing leakage) so higher-order graphs cannot quietly import held-out information.
2. **Which knowledge predicts, under a clean protocol.** A pre-registered validation gate (Δval ≥ +0.002 vs. a matched twin) shows that bipartite question–KC incidence and linear inter-step timing carry accuracy, whereas HypergraphConv, hint-item hyperedges, a full Q-matrix, teacher grouping, an expert DAG subsample, and per-concept exponential forgetting fail. The recommended tracer is a **v4 LSTM** (GIKT-class Q←KC, `--no-graph` on the concept stack) plus Linear log1p Δt — not hypergraph convolution. On XES3G5M fold 0 (mask-repeats, chunked windows, L=400, **test-only**), that tracer reaches **0.8296** versus matched-budget GIKT **0.8258** (minus GIKT: **+0.00378**, 95% CI **[+0.00342, +0.00413]**); Q←KC alone is **0.8270**. Folds 1–2 stay within 0.001 (mean **0.8295±0.0004**). Time is attempt context (query at t+1), not per-concept forgetting. We do **not** lead with graph-only v2 AUC 0.752 vs. GKT 0.834. The FoundationalASSIST session-hyperedge shift 0.720→0.722 is **not** a hypergraph win (hint-item HE Δval +0.00007 FAIL).
3. **Critic-gated explanations over frozen scores.** Diagnostician / Critic / Tutor read Tier 1’s frozen *P*(correct). On the **recommended** tracer, grounded vs. IG flag rates are **0.018 / 0.766** with mean ID-anchored JC **0.161 / 0.152** (n=500). Dual-rater Likert (n=40) is on the **v2** graph-only checkpoint: faithfulness 4.01±1.07, usefulness **3.52**±0.71. Comparison with Ni et al. is a probability gate, not synonym-aware KC overlap or an AUC contest.

**Manuscript history.** Internal verification (integrity check, citation verification, Figure 1 rendering, manipulation-check protocol, then a rewrite of the predictive claim onto the clean L=400 protocol). Substantive points:

1. **Integrity / bibliography.** All bibliography entries checked against primary sources. Ni et al. (2026) is cited in Related Work as the closest KBS precedent on interpretable KT.
2. **Negative knowledge table.** HypergraphConv, hint-item HE, full Q-matrix, teacher grouping, Junyi DAG, and concept-forget are reported as FAIL under the registered gate — we do not loosen the gate to manufacture a hypergraph-AUC story.
3. **Observational ATE (secondary).** IPW ATE of hint use on next-step correctness, with propensity diagnostics and a weak-overlap caveat. Not a primary contribution and not a recommendation against hints.
4. **Tier-2 human rating.** Dual-rater Likert (n=40) on v2: faithfulness 4.01±1.07, usefulness 3.52±0.71; automatic flag/JC in the abstract use the recommended tracer.
5. **Graph-only diagnostic (appendix).** All three XES3G5M folds of the v2 encoder pass frozen-weight node-drop *p*=0.9 (ΔAUC 0.038 / 0.043 / 0.047). This is not the headline system.
6. **Research questions.** RQ1–RQ3 ask which audited knowledge predicts, whether time is forgetting or attempt context, and whether a Critic can explain frozen scores without reopening leakage. Teacher / forum / video remain interface-only.

**Companion paper P0** is cited as an unpublished submission with public code/artefacts; the manuscript is self-contained (Section on the reused P0 protocol). We can attach the P0 PDF as supplementary material on request.

**Dataset access.** FoundationalASSIST is used under Hugging Face Responsible Use Guidelines; we confirm authorized access for this study. P0’s baseline results are vendored verbatim at a pinned commit, not re-derived. GIKT/AKT under the clean L=400 protocol were retrained in this repository.

**Required Elsevier sections** (CRediT authorship contribution statement, Declaration of competing interest, Data availability) are included in the manuscript and have been confirmed by all co-authors. Code: `https://github.com/edu-risk-lab/DH2A-KT`. Corresponding author: Le Hoang Son (`sonlh@vnu.edu.vn`).

Thank you for your consideration.

Sincerely,
The authors
