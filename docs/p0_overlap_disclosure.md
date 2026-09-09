# Companion P0 — what is reused, what is not comparable

P0 (*Leakage-Controlled Concept Graph Construction…*) is a **separate**
manuscript, currently prepared for *Engineering Applications of
Artificial Intelligence* (EAAI, Elsevier). DH²A-KT (this repo, KBS) is
an attribution paper on a later evaluator. Both may be under review at
Elsevier at the same time; they are companions, not duplicate
submissions.

Pinned code: commit `5e3d6a0` in `external/p0_leakage_audit`
([`leakage-controlled-kt-audit`](https://github.com/edu-risk-lab/leakage-controlled-kt-audit)).
Do not edit that tree for DH²A-KT wording.

| P0 claim / artefact | Depends on | Affected by DH² evaluator? | Action in DH²A-KT |
|---|---|---|---|
| Train-only $E_{\mathrm{pre}}$ / $E_{\mathrm{sim}}$ | Learner split | No | Reused; restated in the KBS §3.1 summary |
| Pairwise ECR / TBMR / $\|\rho\|$ on $E_{\mathrm{pre}}$ | Pairwise edges | No | Reused; hyperedge projection is DH² |
| Downstream AUC on every KC-expanded row | Older P0 evaluator | **Yes — not comparable to Table 5** | Do not rank QKC-T / Zero+$\Delta t$ against those columns |
| XES3G5M DKT $0.858$ / DGEKT $0.841$ / GIKT $0.878$ | Older evaluator, $L{=}200$ | Yes as a ranking | Context only; Table 5 is clean $L{=}400$ test-only |
| DGEKT as graph-inert DDR anchor | Destruction sweep | Consistent | Explains Table 5 DGEKT $<$ DKT |
| Cold-start / sparse strata | P0 protocol | Not a DH² claim | Not run as a KBS headline |
| Graph-only vs GKT $\approx$8-point gap | Encoder + $L{=}200$ + protocol | Mixed | Appendix diagnostic only |
| Vendored baseline CSVs (BKT, GKT, SKT) | P0 published tables | Those rows are not in Table 5 | Provenance only |

The KBS cover letter and Data availability attach the P0 PDF as
supplementary material. DH²A-KT does not wait on P0 acceptance.
