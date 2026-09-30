# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing

**Article type:** Full-length original research article

**Journal:** Knowledge-Based Systems (ISSN 0950-7051)

**Corresponding author:** Le Hoang Son, VNU Information Technology Institute, Vietnam National University, Hanoi; sonlh@vnu.edu.vn; ORCID 0000-0001-6356-0046

Dear Editors,

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is a leakage-audited *knowledge attribution* pipeline for knowledge tracing: it constructs train-only auxiliary knowledge, audits hyperedge *membership* leakage, and credits only messages that survive a laddered ablation with capacity-matched, architecture-matched, and package-level controls. It is not a new hypergraph predictor. KBS has recently published on interpretable KT (Ni et al., *Knowledge-Based Systems* 344 (2026) 116071). A Critic-gated layer over frozen scores is an exploratory pilot in the Supplementary Material; it is not a contribution and is not in the title.

Two hooks:

1. **Hyperedge membership leak diagnostic.** Companion paper P0 audits pairwise prerequisite/similarity edges under train-only construction. We add a membership-leak diagnostic on path-derived concept-group hyperedges, plus a conceptual taxonomy (two of three classes remain uninstantiated). The full-log positive control validates the *membership* diagnostic, not pairwise |ρ| (degenerate under constant projected weights) and not TBMR.

2. **Which knowledge predicts.** An operating rule (Δval ≥ +0.002 vs. a designated control on XES3G5M learner fold 0 across five paired training seeds) is met when a Linear log1p Δt *branch* is added as a package-level ablation on both observed-incidence and capacity-matched zero-incidence backbones (5/5). That package is credited as attempt-context timing and boundary information; the contrast is not capacity-matched. Architecture-matched T-zero / T-misaligned controls on the same maps stay positive but below the rule (0/5; mean Δval +0.00183 / +0.00144). Capacity-matched zeroing of the Q–KC incidence *message* does not meet the rule (0/5). On XES3G5M (repeat-target-masked KC-expanded, L=400, test-only), the credited-minimal control reaches 0.8295±0.0001.

**Companion paper P0 (related unpublished manuscript).** The pairwise leakage protocol is a separate unpublished manuscript, *Bounding graph-mediated leakage in knowledge tracing: a train-only audit protocol with a predictive exposure measure*, intended for *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier). Intended overlap: train-only graph construction, pairwise audit diagnostics, and vendored baseline CSVs. Distinct in this submission: hyperedge membership audit, laddered attribution twins, the credit gate, and the later evaluator. We attach a named P0 PDF as related unpublished manuscript S1 (author names restored because KBS is single-anonymized). Code pin for reused artefacts: `edu-risk-lab/leakage-controlled-kt-audit` commit `5e3d6a0`. Downstream P0 AUCs scored on every KC-expanded row are not comparable to our headline table. DH²A-KT does not wait on P0 acceptance.

**Declarations.** The work has not been published and is not under consideration elsewhere. All authors approve this submission. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The authors have no competing interests to declare. Evaluation freeze: https://github.com/edu-risk-lab/DH2A-KT tag `kbs-submit-2026-09-09` (DOI https://doi.org/10.5281/zenodo.22676635). An earlier manuscript freeze is tag `kbs-submit-2026-09-15` (DOI https://doi.org/10.5281/zenodo.22772808); this PDF supersedes that freeze. XES3G5M is the public Liu et al. release. FoundationalASSIST is used under Hugging Face Responsible Use Guidelines and is not redistributed.

Required Elsevier sections (Highlights, CRediT, competing interest, Ethics, Data availability, generative AI) are included in the manuscript.

Thank you for your consideration.

Sincerely,

Le Hoang Son  
Corresponding author  
sonlh@vnu.edu.vn
