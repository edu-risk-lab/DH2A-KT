# Cover letter — IEEE TLT Short Paper

**Manuscript:** DH²A-KT: Dynamic Heterogeneous Hypergraph and Agentic Knowledge Tracing with Leakage-Controlled Graph Construction

**Track:** Short Paper (≤8 pages). The compiled PDF is **8 pages**; please review under the Short Paper length limit (not as a truncated Regular Paper).

**Dear Editors / Reviewers,**

We submit a revised Short Paper manuscript addressing the verification review of our prior draft. Summary of substantive updates:

1. **Integrity / bibliography.** All prior `[TODO]` and “verify before submission” bibliography flags are removed; author lists and venues were checked against primary sources.
2. **Heterogeneous session+Hint evidence (C1).** On FoundationalASSIST (distinct from classic ASSISTments 2012; not interchangeable with P0 ASSIST2012 GKT columns), we construct session hyperedges with Hint members, report leakage diagnostics, and a small concept-only vs +session ablation (AUC 0.720→0.722).
3. **Causal layer evidence (C3).** Observational IPW ATE of hint use on next-step correctness is reported with propensity diagnostics and bootstrap CI; we explicitly caution weak overlap.
4. **Tier-2 human rating (C5).** Dual-rater Likert study (n=40): faithfulness 4.01±1.07, usefulness 3.53±0.71, with agreement metrics.
5. **Review-2 follow-ups.** Table of manipulation checks now includes **P0 GKT** ΔAUC at the same node-drop p=0.9 anchor (GKT also passes, with larger mean drop than DH²-KT); simpleKT is marked N/A (no concept graph). Main AUC table now reports **mean±SD** over three folds and paired Δ vs GKT.
6. **Review-3 follow-ups.** After withdrawing the “unique graph-reliance” claim, Discussion now repositions DH²-KT around (i) hyperedge-level leakage audit, (ii) heterogeneous entity interface (+session/Hint on FoundationalASSIST; forum/teacher remain data-blocked), (iii) causal ATE layer, and (iv) Critic-gated Tier-2 explanations — with an explicit frozen-weight vs retrain protocol note for ΔAUC. Intro wording was aligned with the Abstract on blocked types. Fig. 1 TikZ fill/z-order was fixed (`backgrounds` + `\pgfonlayer`).

**Companion paper P0** is cited as an unpublished APIN submission with public code/artefacts; Section III-A provides a self-contained protocol summary. We can attach the P0 PDF as supplementary material upon request.

**Dataset access.** FoundationalASSIST is used under Hugging Face Responsible Use Guidelines; we confirm authorized access for this study.

Thank you for your consideration.

Sincerely,  
The authors
