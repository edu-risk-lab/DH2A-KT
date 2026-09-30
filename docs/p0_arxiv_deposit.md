# Preprint deposit for the pairwise leakage paper (P0)

This note is the deposit checklist. It is not an arXiv identifier. Do not write an arXiv id into `paper/main.tex` until the deposit succeeds.

## What is deposited

The pairwise paper only: train-only prerequisite and similarity graphs, the pairwise leakage diagnostics, and the older evaluator. Title used with Applied Intelligence: *Bounding graph-mediated leakage in knowledge tracing: a train-only audit protocol with a predictive exposure measure*.

## What is not in that deposit

The credit ladder, the three outcomes, the `+0.002` rule, and the repeat-target-masked `L=400` table of DH²A-KT. Those claims belong to `paper/main.tex`.

## History to state on the arXiv comment

Applied Intelligence rejected the pairwise paper. It is under review at Engineering Applications of Artificial Intelligence. The DH²A-KT manuscript was rejected by Knowledge-Based Systems and is being prepared for Expert Systems. The two manuscripts are not the same paper.

## Files

- Source of the pairwise manuscript: `external/p0_leakage_audit/paper/submission_APIN/` (historical APIN package) and the EAAI source if that tree is the version under review.
- Do not upload the double-anonymized EAAI PDF as if it were the public preprint. Deposit the named author version.
- Overlap table: `docs/p0_overlap_disclosure.md`.

## After the deposit

Put the arXiv id in `paper/references.bib` under `daominh2026p0` only after the id is returned by arXiv. Until then the DH²A-KT manuscript restates the reused steps in `sec:p0-selfcontained` and does not cite an arXiv id.
