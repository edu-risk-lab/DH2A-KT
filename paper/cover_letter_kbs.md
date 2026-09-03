# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing

**Journal:** Knowledge-Based Systems

**Dear Editors / Reviewers,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is leakage-audited *knowledge attribution* for knowledge tracing (KT): it constructs and audits graph-structured auxiliary knowledge, then measures which of that knowledge survives a controlled ablation. KBS has recently published on interpretable KT (Ni et al., “Interpretable knowledge tracing via explicit–implicit alignment,” *Knowledge-Based Systems* 344 (2026) 116071). We take that as evidence of scope fit. A Critic-gated agent layer over frozen scores is reported only as an **exploratory survey after the predictive results**; it is not a contribution and is not in the title. An observational hint identification check is appendix-only and is not a contribution.

Two hooks:

1. **Hyperedge leakage audit.** Companion paper P0 audits pairwise prerequisite/similarity edges under train-only construction. We extend the audit to hyperedges (group-membership, cross-modal, and intervention-timing leakage) so higher-order graphs cannot quietly import held-out information.
2. **Which knowledge predicts, under a clean protocol.** A pre-specified validation gate (Δval ≥ +0.002 vs. a matched twin; internal convention, not a public registry) shows that bipartite question–KC incidence and linear inter-step timing carry accuracy, whereas HypergraphConv, hint-item hyperedges, a full Q-matrix, teacher grouping, an expert DAG subsample, and per-concept exponential forgetting fail. The recommended tracer is **QKC-T**, a v4 LSTM (GIKT-class Q←KC, `--no-graph` on the concept stack) plus Linear log1p Δt — not hypergraph convolution. On XES3G5M fold 0 (mask-repeats, chunked windows, L=400, **test-only**), that tracer reaches **0.8296±0.0001** versus **compute-matched** GIKT **0.8258±0.0001** (minus GIKT: **+0.00378**, 95% CI **[+0.00342, +0.00413]**); Q←KC alone is **0.8270±0.0002**. Folds 1–2 stay within 0.001 (mean **0.8295±0.0004**). On this backbone the exponential forgetting form (P4) does not beat Linear Δt; we do not generalise that to every forgetting mechanism. We do **not** lead with graph-only diagnostic AUC 0.752 vs. GKT 0.834. The FoundationalASSIST session-hyperedge shift 0.720→0.722 is **not** a hypergraph win (hint-item HE Δval +0.00007 FAIL).

**Choice on the explanation layer (B10).** We do not have human ratings on the recommended tracer at n≥100 with ≥3 independent raters, and the grounded vs independent-generation flag gap is largely definitional. We therefore **downgrade** that layer: it is removed from the title and from the contribution list, and it appears only as a survey after Tier 1 results.

**Manuscript history.** Internal verification (integrity check, citation verification, Figure 1 rendering, manipulation-check protocol, then a rewrite of the predictive claim onto the clean L=400 protocol). Substantive points:

1. **Integrity / bibliography.** All bibliography entries checked against primary sources. Ni et al. (2026) is cited in Related Work as the closest KBS precedent on interpretable KT.
2. **Negative knowledge table.** HypergraphConv, hint-item HE, full Q-matrix, teacher grouping, Junyi DAG, and exponential concept-forget are reported as FAIL under the pre-specified gate — we do not loosen the gate to manufacture a hypergraph-AUC story.
3. **Observational identification check.** Moved to the appendix. Weak positivity; the negative IPW estimate must not be read as “hints harm.” Not a contribution.
4. **Compute vs tuning (B11).** GIKT shares DH²-KT’s compute envelope (batch 16, epoch cap 30, patience 5). AKT/simpleKT inherit pyKT/P0 defaults with only L raised to 400; we did not retune learning rate or dropout at L=400. Appendix table lists configs tried.
5. **Graph-only diagnostic (appendix).** All three XES3G5M folds of the diagnostic encoder pass frozen-weight node-drop *p*=0.9 (ΔAUC 0.038 / 0.043 / 0.047). This confirms dependence, not correct relation structure; rewiring and label permutation were not run. This is not the headline system.
6. **Research questions.** RQ1–RQ2 ask which audited knowledge predicts and whether Linear Δt on this backbone outperforms the exponential forgetting form we test. Teacher / forum / video remain interface-only.

**Companion paper P0** is cited as an unpublished submission with public code/artefacts; the manuscript is self-contained (Section on the reused P0 protocol). We can attach the P0 PDF as supplementary material on request.

**Dataset access.** FoundationalASSIST is used under Hugging Face Responsible Use Guidelines; we confirm authorized access for this study. P0’s baseline results are vendored verbatim at a pinned commit, not re-derived. GIKT/AKT under the clean L=400 protocol were retrained in this repository.

**Required Elsevier sections** (CRediT authorship contribution statement, Declaration of competing interest, Data availability) are included in the manuscript and have been confirmed by all co-authors. Code: `https://github.com/edu-risk-lab/DH2A-KT`. Corresponding author: Le Hoang Son (`sonlh@vnu.edu.vn`).

Thank you for your consideration.

Sincerely,
The authors
