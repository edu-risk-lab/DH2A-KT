# Cover letter — Expert Systems (Wiley)

**Manuscript:** DH²A-KT: A Credit Protocol for Leakage-Audited Signals in Knowledge Tracing

**Article type:** Research article

**Journal:** Expert Systems (Wiley)

**Corresponding author:** Le Hoang Son, VNU Information Technology Institute, Vietnam National University, Hanoi; sonlh@vnu.edu.vn; ORCID 0000-0001-6356-0046

Dear Editors,

We submit a methodology paper on which leakage-audited signals in knowledge tracing deserve credit for a change in AUC. DH²A-KT is the name of the protocol, not a new predictor. The protocol builds train-only auxiliary knowledge, audits hyperedge membership leakage, and assigns each message one of three outcomes on a three-rung ladder: capacity-matched, architecture-matched, or package-level. A message is credited only when every one of five paired training seeds gains at least +0.002 validation AUC against its designated control. A gain that is positive on every seed and still below that cut is reported and is not credited.

On XES3G5M learner fold 0 the linear log(1+Δt) package meets the rule (5/5) and is credited as attempt-context timing and boundary information. That contrast adds a linear map (256 parameters at hidden size 128). A no-time control that carries the same 256 parameters unused does not close the gap (5/5 at +0.002). Architecture-matched controls that keep the map and zero or misalign the gap stay positive and below the cut (0/5; mean Δval +0.00183 versus T-zero, +0.00144 versus T-misaligned). Capacity-matched removal of the question–KC incidence message does not meet the rule (0/5; mean Δval −0.00010).

The map does not transfer unchanged. On ASSISTments 2012 fold 0 every contrast is positive and no message is credited. On a 50,000-learner Junyi subsample the aligned gap input is credited under both the absolute cut and a cut scaled to seed noise. We report all three maps rather than one headline. The paper does not claim a new state-of-the-art tracer.

**Prior submission.** This manuscript was submitted to Knowledge-Based Systems and was rejected. The present version is rewritten as a protocol paper: the credit rule, the three outcomes, and the procedure for registering a new message are stated before the tracer. We do not attach the KBS reviews.

**Related manuscript.** A separate paper, Bounding graph-mediated leakage in knowledge tracing, audits pairwise train-only graphs. Applied Intelligence rejected that paper. It is now under review at Engineering Applications of Artificial Intelligence. The two papers share train-only graph construction. They do not share the credit ladder, the three outcomes, or the later evaluator. Downstream AUC columns from the pairwise paper are not comparable to the table in this manuscript. The reused pairwise steps are restated here so the credit results do not depend on an unpublished PDF. A preprint deposit of the pairwise paper is prepared for the authors' arXiv account; this submission does not claim an arXiv identifier for it.

**Declarations.** The work has not been published and is not under consideration elsewhere. All authors approve this submission. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The authors have no competing interests to declare. Evaluation code and numeric artefacts for the XES3G5M ladder are frozen at https://github.com/edu-risk-lab/DH2A-KT tag `kbs-submit-2026-09-09` (DOI https://doi.org/10.5281/zenodo.22676635).

Thank you for your consideration.

Sincerely,

Le Hoang Son  
Corresponding author  
sonlh@vnu.edu.vn
