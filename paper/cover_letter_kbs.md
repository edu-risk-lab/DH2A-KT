# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing

**Journal:** Knowledge-Based Systems

**Dear Editors / Reviewers,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is a leakage-audited *knowledge attribution* pipeline for knowledge tracing: it constructs train-only auxiliary knowledge, audits hyperedge leakage, and credits only messages that survive a matched-twin gate. It is not a new hypergraph predictor. KBS has recently published on interpretable KT (Ni et al., “Interpretable knowledge tracing via explicit–implicit alignment,” *Knowledge-Based Systems* 344 (2026) 116071). A Critic-gated agent layer over frozen scores is an appendix survey only; it is not a contribution and is not in the title.

Two hooks:

1. **Hyperedge leakage audit.** Companion paper P0 (Supplementary Material S1) audits pairwise prerequisite/similarity edges under train-only construction. We extend that audit to hyperedges (group-membership, cross-modal, and intervention-timing leakage) so higher-order graphs cannot quietly import held-out information.

2. **Which knowledge predicts, under a clean protocol.** A pre-specified gate (Δval ≥ +0.002 vs. a matched twin; internal convention) shows that Linear log1p Δt passes on both the observed-incidence and capacity-matched zero-incidence backbones (5/5 each; mean Δval +0.00255 and mean test-only Δ +0.00269 on the zero-incidence twin). Capacity-matched zeroing of the train-only Q–KC *incidence message* fails (0/5; mean Δval −0.00010; 95% CI [−0.00052, +0.00032]). HypergraphConv, hint-item hyperedges, a full Q-matrix, teacher grouping, an expert DAG subsample, and the tested exponential forgetting form also fail. QKC-T is the observed-incidence evaluation arm; Zero+Δt is the minimalist positive control. On XES3G5M fold 0 (mask-repeats, chunked windows, L=400, test-only), those arms reach **0.8296±0.0001** and **0.8295±0.0001** versus compute-matched GIKT **0.8258±0.0001**. History-only timing fails as a credit claim ($2/5$ PASS). Predictive conclusions are scoped to XES3G5M; FoundationalASSIST and Junyi are feasibility/construction diagnostics.

**Choice on the explanation layer.** Blinded Likert ratings on the QKC-T checkpoint are available ($n{=}100$, 3 raters: faithfulness $4.55{\pm}0.63$, usefulness $4.01{\pm}0.74$). We keep that layer as an appendix survey.

**Manuscript history.** Integrity check, citation verification, then a rewrite of the predictive claim onto the clean L=400 protocol and the capacity-matched attribution map.

**Companion paper P0** is cited as an unpublished submission. The manuscript restates the reused train-only protocol; Supplementary Material S1 is the companion PDF. P0 code/artefacts are pinned at commit `5e3d6a0`.

**Dataset access.** XES3G5M is the public release of Liu et al. FoundationalASSIST is used under Hugging Face Responsible Use Guidelines. GIKT/AKT/simpleKT under the clean L=400 protocol were retrained in this repository under a shared compute envelope, not a matched tuning search.

**Required Elsevier sections** (Highlights, CRediT, competing interest, Funding, Ethics, Data availability, generative AI) are included. This research received no specific grant. Code: `https://github.com/edu-risk-lab/DH2A-KT` (default branch, commit `b60e6b8`; public at submission). Corresponding author: Le Hoang Son (`sonlh@vnu.edu.vn`).

Thank you for your consideration.

Sincerely,
The authors
