# Cover letter — Expert Systems (Wiley)

**Manuscript:** Validating Auxiliary Knowledge in Knowledge Tracing: A Credit Protocol and Its Dependence on Setting

**Running title:** Credit depends on the setting

**Article type:** Research article

**Journal:** Expert Systems (Wiley)

**Corresponding author:** Le Hoang Son, VNU Information Technology Institute, Vietnam National University, Hanoi; sonlh@vnu.edu.vn; ORCID 0000-0001-6356-0046

Dear Editors,

We submit a knowledge-validation paper for knowledge tracing. Knowledge tracing systems are often built by adding auxiliary knowledge (question–concept incidence, concept graphs, response timing) to a sequence model and are judged by one AUC for the whole system. The manuscript proposes a credit protocol that decides which piece of knowledge earned a gain. It builds auxiliary knowledge from training learners only, audits hyperedge membership leakage, compares each message with one designated control (capacity-matched, architecture-matched, or package-level), and assigns one of three outcomes from five paired training seeds: credited, consistent-positive (not credited), or unsupported. Two decision rules are reported side by side, and they agree in every setting. The paper does not propose a new state-of-the-art tracer. The appendices are supplied as a separate file, as the author guidelines request. An exploratory rating of model-generated explanations is in the repository and is not part of the submitted files.

We apply the protocol to one elapsed-time message on three corpora and two encoders, and the same message receives different verdicts. On XES3G5M with a recurrent tracer the time package is credited only as a package: a map fed only zeros already accounts for about a quarter of the gain, and the gap input itself is not credited. That gap input is credited on Junyi. On ASSISTments 2012 the package is credited only on a tracer whose question–concept incidence is zeroed, and only on some learner folds; the corpus's own configuration credits nothing. An attention encoder on XES3G5M credits no time message. On the 1,000 s ASSISTments export the package is consistent-positive (not credited); on a one-second re-export of the same rows it is credited. Credit is therefore reported per corpus, backbone, recipe, learner fold, and control level, and the manuscript gives a procedure for registering a new message under the same rules.

**Prior submission.** This manuscript was submitted to Knowledge-Based Systems and was rejected. The present version is rewritten as a protocol paper: the credit rule, the three outcomes, and the procedure for registering a new message are stated before the tracer. We do not attach the KBS reviews.

**Related manuscript.** A separate paper, Bounding graph-mediated leakage in knowledge tracing, audits pairwise train-only graphs. It is under review at Engineering Applications of Artificial Intelligence. The two papers share train-only graph construction. They do not share the credit ladder, the three outcomes, or the later evaluator. Downstream AUC columns from the pairwise paper are not comparable to the table in this manuscript. The reused pairwise steps are restated here so the credit results do not depend on an unpublished PDF. A preprint deposit of the pairwise paper is prepared for the authors' arXiv account; this submission does not claim an arXiv identifier for it.

**Declarations.** The work has not been published and is not under consideration elsewhere. All authors approve this submission. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The authors have no competing interests to declare. Code and numeric artefacts are at https://github.com/edu-risk-lab/DH2A-KT. The manuscript state, including the credit-map files, is archived at https://doi.org/10.5281/zenodo.23219471 (tag es-submit-2026-10-08b). An earlier snapshot of the XES3G5M reference ladder alone is https://doi.org/10.5281/zenodo.22676635.

Thank you for your consideration.

Sincerely,

Le Hoang Son  
Corresponding author  
sonlh@vnu.edu.vn
