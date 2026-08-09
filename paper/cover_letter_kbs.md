# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Diagnostic Heterogeneous Hypergraph and Agentic Knowledge Tracing with Leakage-Controlled Graph Construction

**Journal:** Knowledge-Based Systems

**Dear Editors / Reviewers,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript sits at the intersection of graph-augmented knowledge tracing (KT) and LLM-based explainability — a combination KBS has recently published on (e.g., Ni et al., "Interpretable knowledge tracing via explicit–implicit alignment," *Knowledge-Based Systems* 344 (2026) 116071), which we take as evidence of scope fit. DH²A-KT differs from and complements that line of work in three ways: (i) it audits the auxiliary graph itself for train/test leakage before any explanation is generated, at hyperedge granularity, not just pairwise edges; (ii) it adds a causal-effect layer (observational hint ATE via inverse-propensity weighting) rather than only descriptive/explanatory feedback; (iii) its explanation layer (Tier 2, AgentDG-KT) is gated by a mandatory Critic agent that can reject a Diagnostician's rationale outright, rather than only scoring it post hoc.

**Manuscript history.** This draft has been through four internal verification rounds (integrity check, citation verification, Figure 1 rendering fix, manipulation-check protocol clarification) prior to submission; a full record is available on request. Substantive points addressed:

1. **Integrity / bibliography.** All bibliography entries checked against primary sources; no placeholder or unverified citations remain. Ni et al. (2026) is cited in Related Work as the closest KBS precedent.
2. **Heterogeneous session+Hint evidence.** On FoundationalASSIST (distinct from classic ASSISTments 2012), we construct session hyperedges with Hint members, report leakage diagnostics, and a concept-only vs. +session ablation (AUC 0.720→0.722).
3. **Causal layer.** Observational IPW ATE of hint use on next-step correctness, with propensity diagnostics, bootstrap CI, and an explicit weak-overlap caveat.
4. **Tier-2 human rating.** Dual-rater Likert study (n=40): faithfulness 4.01±1.07, usefulness 3.53±0.71.
5. **Manipulation check.** Table reports P0's GKT ΔAUC at the same node-drop p=0.9 anchor as DH²-KT (GKT also passes, with a larger mean drop); the frozen-weight (DH²-KT) vs. retrain (GKT) protocol difference is stated explicitly rather than presented as an apples-to-apples comparison.
6. **Research Questions.** An explicit RQ1–RQ5 decomposition (with one question the paper does not answer) states plainly that Teacher/Forum/Discussion/Video interactions are interface-only in the current schema, not empirically demonstrated.
7. **Explanation faithfulness harness (KBS bar).** We implement ID-anchored KC-Jaccard (structured `GroundedKCs` trailer vs. history ∪ 1-hop $E_{pre}$) and an Independent Generation (IG) negative control that strips frozen $P(\mathrm{correct})$ from the Diagnostician while keeping the Critic conditioned on Tier-1. The evaluation suite and GPU runbook are in-repo (`scripts/18_run_kbs_faithfulness_suite.py`, `docs/kbs-gpu-runbook.md`); StubLLM harness results are reported in Table `tab:kc-jaccard-ig`, with Ollama $n{=}500$ grounded vs. IG cells to be filled from the GPU host before camera-ready if not already present at submission.

**Companion paper P0** is cited as an unpublished submission with public code/artefacts; the manuscript is self-contained (Section on the reused P0 protocol summarizes what is needed to check our results without reading P0). We can attach the P0 PDF as supplementary material on request.

**Dataset access.** FoundationalASSIST is used under Hugging Face Responsible Use Guidelines; we confirm authorized access for this study. P0's baseline results are vendored verbatim at a pinned commit (Section on baselines reused from P0), not re-derived.

**Required Elsevier sections** (CRediT authorship contribution statement, Declaration of competing interest, Data availability) are included in the manuscript and are pending final co-author confirmation (`docs/kbs-author-gate.md`).

Thank you for your consideration.

Sincerely,
The authors
