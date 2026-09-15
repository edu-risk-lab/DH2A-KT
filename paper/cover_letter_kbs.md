# Cover letter — Knowledge-Based Systems (Elsevier)

**Manuscript:** DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing

**Journal:** Knowledge-Based Systems

**Dear Editors,**

We submit DH²A-KT for consideration at Knowledge-Based Systems. The manuscript is a leakage-audited *knowledge attribution* pipeline for knowledge tracing: it constructs train-only auxiliary knowledge, audits hyperedge *membership* leakage, and credits only messages that survive a laddered ablation with capacity-matched, architecture-matched, and package-level controls. It is not a new hypergraph predictor. KBS has recently published on interpretable KT (Ni et al., *Knowledge-Based Systems* 344 (2026) 116071). A Critic-gated layer over frozen scores is an exploratory pilot in the Supplementary Material; it is not a contribution and is not in the title.

Two hooks:

1. **Hyperedge membership leak diagnostic.** Companion paper P0 audits pairwise prerequisite/similarity edges under train-only construction. We add a membership-leak diagnostic on path-derived concept-group hyperedges, plus a conceptual taxonomy (two of three classes remain uninstantiated). The full-log positive control validates the *membership* diagnostic, not pairwise |ρ| (degenerate under constant projected weights) and not TBMR.

2. **Which knowledge predicts.** An operating rule (Δval ≥ +0.002 vs. a designated control on XES3G5M learner fold 0 across five paired training seeds) is met when a Linear log1p Δt *branch* is added as a package-level ablation on both observed-incidence and capacity-matched zero-incidence backbones (5/5). That package is credited as attempt-context timing and boundary information; the contrast is not capacity-matched. Architecture-matched T-zero / T-misaligned controls on the same maps stay positive but below the rule (0/5; mean Δval +0.00183 / +0.00144). Capacity-matched zeroing of the Q–KC incidence *message* does not meet the rule (0/5). On XES3G5M (repeat-target-masked KC-expanded, L=400, test-only), the credited-minimal control reaches 0.8295±0.0001.

**Companion paper P0.** The pairwise leakage protocol is a separate manuscript, *Leakage-Controlled Concept Graph Construction for Knowledge Tracing*, prepared for *Engineering Applications of Artificial Intelligence* (EAAI, Elsevier). Intended overlap: train-only graph construction, pairwise audit diagnostics, and vendored baseline CSVs. Distinct in this submission: hyperedge membership audit, laddered attribution twins, the credit gate, and the later evaluator. We attach the P0 PDF as related unpublished manuscript S1 (source: companion repo `edu-risk-lab/leakage-controlled-kt-audit`, pin `5e3d6a0`, file `paper/submission_APIN/main_APIN.tex`). Downstream P0 AUCs scored on every KC-expanded row are not comparable to our headline table; an overlap map is in `docs/p0_overlap_disclosure.md`. DH²A-KT does not wait on P0 acceptance.

**Code.** Evaluation freeze: https://github.com/edu-risk-lab/DH2A-KT tag `kbs-submit-2026-09-09` (DOI https://doi.org/10.5281/zenodo.22676635). This PDF's LaTeX: tag `kbs-submit-2026-09-15` (DOI https://doi.org/10.5281/zenodo.22772808). Table-to-script map: `docs/paper-artifacts.md`. XES3G5M is the public Liu et al. release. FoundationalASSIST is used under Hugging Face Responsible Use Guidelines and is not redistributed.

**Required Elsevier sections** (Highlights, CRediT, competing interest, Ethics, Data availability, generative AI) are included. This research received no specific grant. Corresponding author: Le Hoang Son (`sonlh@vnu.edu.vn`).

Thank you for your consideration.

Sincerely,
The authors
