# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing

**Journal:** Knowledge-Based Systems

**Dear Editors,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is a leakage-audited *knowledge attribution* pipeline for knowledge tracing: it constructs train-only auxiliary knowledge, audits hyperedge leakage, and credits only messages that survive a matched-twin operating rule. It is not a new hypergraph predictor. KBS has recently published on interpretable KT (Ni et al., *Knowledge-Based Systems* 344 (2026) 116071). A Critic-gated layer over frozen scores is an exploratory pilot in the Supplementary Material; it is not a contribution and is not in the title.

Two hooks:

1. **Hyperedge leakage audit.** Companion paper P0 audits pairwise prerequisite/similarity edges under train-only construction. We extend that audit to hyperedges (group-membership, plus two classes that remain uninstantiated on our corpora). The full-log positive control validates the *membership* diagnostic, not pairwise $|\rho|$ (degenerate under constant projected weights).

2. **Which knowledge predicts.** An operating rule (Δval ≥ +0.002 vs. a matched twin on all five paired seeds) is met when a Linear log1p Δt *branch* is added on both observed-incidence and capacity-matched zero-incidence backbones (5/5). Architecture-matched T-zero / T-misaligned controls on the same maps do not meet the rule (0/5), so aligned Δt is not isolated. Capacity-matched zeroing of the Q–KC incidence *message* does not meet the rule (0/5). On XES3G5M (repeat-target-masked KC-expanded, L=400, test-only), the credited-minimal control and the GIKT-family evaluation arm both reach 0.8295±0.0001. Training-envelope-matched GIKT is a contextual benchmark (0.8258±0.0001), not a tuning-matched superiority claim.

**Companion paper P0** (pairwise leakage protocol) is a separate manuscript prepared for *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier). The two papers share train-only graph construction; they do not share the DH² evaluator or the attribution twins. We attach the P0 PDF as Supplementary Material S1. Code/artefacts for P0 are pinned at commit `5e3d6a0`. Downstream P0 AUCs scored on every KC-expanded row are not comparable to our Table 5. An overlap map is in `docs/p0_overlap_disclosure.md`. DH²A-KT does not wait on P0 acceptance.

**Code.** https://github.com/edu-risk-lab/DH2A-KT (tag `kbs-submit-2026-09-09`). Table-to-script map: `docs/paper-artifacts.md`. XES3G5M is the public Liu et al. release. FoundationalASSIST is used under Hugging Face Responsible Use Guidelines and is not redistributed.

**Required Elsevier sections** (Highlights, CRediT, competing interest, Ethics, Data availability, generative AI) are included. This research received no specific grant. Corresponding author: Le Hoang Son (`sonlh@vnu.edu.vn`).

Thank you for your consideration.

Sincerely,
The authors
