# Cover letter — Expert Systems (Wiley)

**Manuscript:** Validating Auxiliary Knowledge in Knowledge Tracing: A Credit Protocol with Controls, Availability Checks, and Family-Wise Decisions

**Running title:** Credit with controls and availability.

**Article type:** Original article

**Journal:** Expert Systems (Wiley)

**Corresponding author:** Le Hoang Son, VNU Information Technology Institute, Vietnam National University, Hanoi; sonlh@vnu.edu.vn; ORCID 0000-0001-6356-0046

**Author email addresses:** Tuan Dao Minh, tuanymc@utehy.edu.vn; Khanh-Trinh Nguyen, trinhnk@utehy.edu.vn; Duong Nguyen Tien, duongnt@utehy.edu.vn; Quoc Khanh Ngo, quockhanhngo.official@gmail.com; Van-Hau Nguyen, nvhau66@gmail.com; Le Hoang Son, sonlh@vnu.edu.vn.

Dear Editors,

We submit a knowledge-validation paper for knowledge tracing. Knowledge tracing systems are often built by adding auxiliary knowledge (question–concept incidence, concept graphs, response timing) to a sequence model and are judged by one AUC for the whole system. The manuscript proposes a credit protocol that decides which piece of knowledge earned a gain. It builds auxiliary knowledge from training learners only, audits membership leakage and input availability, compares each message with one designated control (capacity-matched, architecture-matched, or package-level), and assigns one of three outcomes from ten paired training seeds on pooled held-out evaluation AUC (original valid∪test learners; not used for checkpoint selection): credited; positive, not credited; or unsupported. Primary (family C) contrasts use Holm adjustment for family-wise error; exploratory (family E) contrasts use Benjamini–Hochberg for false-discovery rate; inputs unavailable at prediction time are diagnostic only. The earlier validation-AUC interval and seed rules are kept in the appendix. The families, primary metric, and decision rule were fixed in the repository for a retrospective reanalysis of already-collected ten-seed runs; the ten-seed design, baseline grid, and leakage simulation were fixed earlier (commit 99521c6). The paper does not propose a new state-of-the-art tracer.

The paper fits the scope of Expert Systems as a validation method for knowledge-engineered tutoring components: knowledge is acquired from logs or experts, represented as incidence, graphs, or timing features, and injected into a learned model. The protocol decides which injected piece may be credited before deployment, with the caveat that credit labels can differ across corpus, backbone, and timestamp export even when effect sizes are similar. The appendices are supplied as a separate file, as the author guidelines request, and are uploaded as Supporting Information. An exploratory rating of model-generated explanations is in the repository and is not part of the submitted files.

We apply the protocol to one elapsed-time message on three corpora, two encoders, and three learner folds. In the primary analysis family (Holm/FWER), only the time package on XES3G5M with a recurrent tracer is credited, and only as a package: the aligned gap stays positive and is not credited after Holm adjustment on every fold. That package feeds the gap to history and query, so it applies to attempt-onset scoring rather than to selecting the next item before it opens. No primary ASSISTments or attention-encoder contrast is credited; an exploratory attention package on the ASSISTments one-second export stays positive and not credited under BH/FDR. Junyi completion-time query gaps and a rebuilt attempt-onset gap that moves with post-response retries are reported only as leakage diagnostics; history-only Junyi gaps on the 50k subsample and all-learner folds 0–2 are exploratory and not credited. An illustrative simulation shows that a hint chosen with knowledge of the next response raises pooled evaluation AUC by 5.18 × 10⁻³ and can pass the credit tests; it is flagged only by an audit that records how each hint was chosen. The primary credit rule was fixed after earlier interval/seed outcomes had been archived; the manuscript states that history and reports every contrast whose label changes. AKT, simpleKT, and the tracer itself each receive a 12-configuration optimiser grid for context; those grids are not a superiority claim and do not enter the credit map. The manuscript gives a procedure for registering a new message under the same rules.

**Prior submission.** This manuscript was submitted to Knowledge-Based Systems and was rejected. The present version is rewritten as a protocol paper: the credit rule, the three outcomes, and the procedure for registering a new message are stated before the tracer. We do not attach the KBS reviews.

**Related manuscript.** A separate paper, Bounding graph-mediated leakage in knowledge tracing, audits pairwise train-only graphs. It is under review at Engineering Applications of Artificial Intelligence. Shared material: train-only graph construction and the four pairwise membership diagnostics restated in Section~5.3 (contribution C3). New to this manuscript: the credit ladder, three outcomes, primary rule on pooled held-out evaluation AUC, rung controls, availability gate, Junyi onset/timestamp-semantics audit, and the later evaluator. Downstream AUC columns from the pairwise paper are not comparable to the tables here. Credit results do not depend on reading the unpublished pairwise PDF. A preprint deposit of the pairwise paper is prepared for the authors' arXiv account; this submission does not claim an arXiv identifier for it.

**Declarations.** The work has not been published and is not under consideration elsewhere. All authors approve this submission. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The authors have no competing interests to declare. Code and numeric artefacts are at https://github.com/edu-risk-lab/DH2A-KT (tag `es-submit-2026-11-02`). Archival DOIs for earlier manuscript states, and the Zenodo DOI for this tag once deposited, are listed in the Data Availability Statement of the manuscript.

Thank you for your consideration.

Sincerely,

Le Hoang Son  
Corresponding author  
sonlh@vnu.edu.vn
