# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing

**Journal:** Knowledge-Based Systems

**Dear Editors / Reviewers,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is a leakage-audited *knowledge attribution* pipeline for knowledge tracing: it constructs train-only auxiliary knowledge, audits hyperedge leakage, and credits only messages that survive a matched-twin operating rule. It is not a new hypergraph predictor. KBS has recently published on interpretable KT (Ni et al., “Interpretable knowledge tracing via explicit–implicit alignment,” *Knowledge-Based Systems* 344 (2026) 116071). A Critic-gated agent layer over frozen scores is an exploratory pilot only; it is not a contribution and is not in the title.

Two hooks:

1. **Hyperedge leakage audit.** Companion paper P0 (Supplementary Material S1) audits pairwise prerequisite/similarity edges under train-only construction. We extend that audit to hyperedges (group-membership, cross-modal, and intervention-timing leakage) so higher-order graphs cannot quietly import held-out information. The full-log positive control validates the *membership* diagnostic, not pairwise $|\rho|$.

2. **Which knowledge predicts, under a repeat-target-masked KC-expanded protocol.** An operating rule (Δval ≥ +0.002 vs. a matched twin on every one of five paired seeds) is met when a Linear log1p Δt *branch* is added on both the observed-incidence and capacity-matched zero-incidence backbones (5/5 each; mean Δval +0.00255 and mean test-only Δ +0.00269 on the zero-incidence twin). Those twins add maps as well as gap input. Architecture-matched T-zero / T-misaligned controls on the same maps do not meet the val rule (0/5; mean Δval +0.00183 / +0.00144), so aligned Δt is not isolated from extra capacity. Capacity-matched zeroing of the train-only Q–KC *incidence message* does not meet the rule (0/5; mean Δval −0.00010). History-only timing is 2/5 seeds pass and is not credited. QKC-T is the observed-incidence evaluation arm kept for GIKT-family parity; Zero+Δt is the credited-minimal control. On XES3G5M fold 0 (mask-repeats, chunked windows, L=400, test-only), QKC-T and Zero+Δt both reach **0.8295±0.0001** (n=1,093,755; five seeds). Table 5 is a *contextual* benchmark: training-budget-matched—not tuning-matched—GIKT is **0.8258±0.0001** (n=1,093,720). Three learner folds of QKC-T mean **0.8295±0.0004**. The archived combined val+test mean 0.8296±0.0001 (n=1,640,242) is not used as a headline cell. HypergraphConv, hint-item hyperedges, a full Q-matrix, and a Junyi DAG subsample remain single-seed diagnostics; teacher grouping and exponential forgetting are 0/5. Predictive conclusions are scoped to XES3G5M; FoundationalASSIST and Junyi are feasibility/construction diagnostics. This protocol is not claimed to be pyKT’s question-level all-in-one evaluator.

**Choice on the explanation layer.** Blinded ratings on the QKC-T checkpoint are available ($n{=}100$, 3 raters: faithfulness $4.55{\pm}0.63$, usefulness $4.01{\pm}0.74$). We keep that layer as an exploratory pilot.

**Reporting notes (09 Sep 2026).** Holm–Bonferroni on the $m{=}3$ family uses two-sided raw $p$ throughout ($p_{\mathrm{Holm}}$ $8.59{\times}10^{-6}$ / $3.58{\times}10^{-5}$ / $0.53$). The $+0.002$ $5/5$ rule is an operating threshold, not $H_0{:}\,\mu_\Delta=0$. A1 T-zero / T-misaligned: 0/5 (not isolated). B6 occurrence join: $n_{\mathrm{joined}}=1{,}093{,}718$; 36+2 unmatched; 8 label mismatches (kept). The 1,093,755 vs 1,093,754 gap is one hashed KC not in the train-fold map. Table 5 / Appendix C vs pyKT are contextual, not event-paired. B8 Zero+Δt ECE 0.0072. B9 same-clean PASS ($2.0{\times}10^{-9}$). A2: 208 chunk windows start on a global repeat. P0 train-only graph construction remains reused; P0 downstream AUCs scored on every KC-expanded row are not comparable to Table 5. GitHub URL is not yet live.

**Companion paper P0** is cited as an unpublished submission. The manuscript restates the reused train-only protocol; Supplementary Material S1 is the companion PDF. P0 code/artefacts are pinned at commit `5e3d6a0`. DH²A-KT does not wait on P0 acceptance.

**Dataset access.** XES3G5M is the public release of Liu et al. FoundationalASSIST is used under Hugging Face Responsible Use Guidelines. GIKT/AKT/simpleKT under the clean L=400 protocol were retrained in this repository under a shared training envelope, not a matched tuning search.

**Required Elsevier sections** (Highlights, CRediT, competing interest, Funding, Ethics, Data availability, generative AI) are included. This research received no specific grant. Code: provided to reviewers on request; intended public URL `https://github.com/edu-risk-lab/DH2A-KT` is not yet live. Corresponding author: Le Hoang Son (`sonlh@vnu.edu.vn`).

Thank you for your consideration.

Sincerely,
The authors
