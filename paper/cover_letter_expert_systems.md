# Cover letter — Expert Systems (Wiley)

**Manuscript:** Validating Auxiliary Knowledge in Knowledge Tracing: A Credit Protocol Shows That Credit Depends on Corpus and Encoder

**Running title:** Credit protocol for knowledge tracing

**Article type:** Research article

**Journal:** Expert Systems (Wiley)

**Corresponding author:** Le Hoang Son, VNU Information Technology Institute, Vietnam National University, Hanoi; sonlh@vnu.edu.vn; ORCID 0000-0001-6356-0046

Dear Editors,

We submit a knowledge-validation paper for knowledge tracing. Knowledge tracing systems are often built by adding auxiliary knowledge (question–concept incidence, concept graphs, response timing) to a sequence model and are judged by one AUC for the whole system. The manuscript states a credit protocol that decides which piece of knowledge earned a gain. It builds auxiliary knowledge from training learners only, audits hyperedge membership leakage, compares each message with one designated control (capacity-matched, architecture-matched, or package-level), and assigns one of three outcomes from five paired training seeds: credited, consistent-positive but not credited, or unsupported. Two decision rules are reported side by side: every seed gains at least +0.002 validation AUC, and the lower 95% bound of the mean paired gain is at least +0.002. The two rules agree in every setting. The paper does not propose a new state-of-the-art tracer. The appendices are supplied as a separate file, as the author guidelines request.

We apply the protocol to one elapsed-time message across two encoders and three corpora, and the same message receives different verdicts. On XES3G5M with a recurrent tracer, the time package is credited (5/5 seeds), including against a no-time control that carries the same 256 parameters unused. With the map held fixed, the aligned gap input is positive on every seed and not credited (mean Δval +0.00183 versus a zeroed gap, +0.00144 versus a misaligned gap). These outcomes repeat on three learner folds. Question–concept incidence is not credited (mean Δval −0.00010). ASSISTments 2012 under the same recipe credits the package on two of three learner folds (5/5 on folds 0 and 2) and leaves it consistent-positive on fold 1 (4/5; one seed +0.00116), and under its own configuration credits nothing. With a causal attention encoder no time message is credited. On Junyi both the package and the aligned gap input are credited. While describing the gaps we found that the ASSISTments export we had inherited stores start times at 1,000 s resolution; on that export the package is not credited, and we report this as a sensitivity result. We therefore report credit per corpus, backbone, recipe, learner fold, and control level rather than as one map, and we give a procedure for registering a new message under the same rules.

**Prior submission.** This manuscript was submitted to Knowledge-Based Systems and was rejected. The present version is rewritten as a protocol paper: the credit rule, the three outcomes, and the procedure for registering a new message are stated before the tracer. We do not attach the KBS reviews.

**Related manuscript.** A separate paper, Bounding graph-mediated leakage in knowledge tracing, audits pairwise train-only graphs. Applied Intelligence rejected that paper. It is now under review at Engineering Applications of Artificial Intelligence. The two papers share train-only graph construction. They do not share the credit ladder, the three outcomes, or the later evaluator. Downstream AUC columns from the pairwise paper are not comparable to the table in this manuscript. The reused pairwise steps are restated here so the credit results do not depend on an unpublished PDF. A preprint deposit of the pairwise paper is prepared for the authors' arXiv account; this submission does not claim an arXiv identifier for it.

**Declarations.** The work has not been published and is not under consideration elsewhere. All authors approve this submission. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The authors have no competing interests to declare. Code and numeric artefacts are at https://github.com/edu-risk-lab/DH2A-KT. The archived snapshot https://doi.org/10.5281/zenodo.22676635 covers the XES3G5M reference ladder; the transfer runs in this version will be archived under a new DOI before publication.

Thank you for your consideration.

Sincerely,

Le Hoang Son  
Corresponding author  
sonlh@vnu.edu.vn
