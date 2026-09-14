# Citation and prior-art verification log

Date of this pass: 10 September 2026.
Sources: Crossref / OpenAlex snapshot in `tmp/audit_dois.json`; ACM DL, IEEE Xplore, Scopus, arXiv, and Google Scholar for the §2.4 scoped search.

Do **not** change Corbett & Anderson (1995) or Ma et al. Applied Sciences 15(15):8617 (2025). Both were independently confirmed against publisher metadata.

## Scoped “no hyperedge-level protocol” claim

Search date: 10 September 2026.
Sources: ACM Digital Library, IEEE Xplore, Scopus, arXiv, Google Scholar.
Seed terms: “hypergraph knowledge tracing”, “leakage audit”, “graph provenance”, “KT label leakage”, “question-level evaluation”, “controlled attribution”.
Extended terms: “graph self-supervised leakage”, “graph SSL data leakage”, “recommender leakage evaluation”, “hyperedge membership leakage”.

Finding recorded in §2.4: within that scoped search we did not find a protocol that applies P0’s leakage diagnostics at the hyperedge level. This is a search-bounded statement, not a claim that no such protocol exists outside the listed sources.

## Post-2024 bibliography keys

| Key | Year in .bib | DOI | Crossref / OpenAlex title match | HTTP | Notes |
|---|---|---|---|---|---|
| dygkt2024 | 2024 | 10.1145/3637528.3671773 | DyGKT: Dynamic Graph Learning for Knowledge Tracing | 403 (ACM) | Title confirmed; ACM often 403 from scripts |
| dgekt2024 | 2024 | 10.1145/3638350 | DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing | — | TOIS record present |
| psykt2024 | 2024 | 10.3389/fpsyg.2024.1359199 | match | 200 | Keep |
| htkt2025 | 2024/2025 | 10.1109/ISPA63168.2024.00199 | HTKT: Knowledge Tracing Based on Hypergraph Transformer | 202 | Venue year 2024; .bib year 2025 is the printed key, not the venue year |
| thmn2025 | 2025 | 10.1007/978-3-031-99267-4_10 | match | 200 | Keep |
| hgkt2025 | 2025 | 10.3390/app15158617 | match | 403 | Applied Sciences 8617 — confirmed, do not edit |
| esk2024 | 2025 | 10.1007/978-3-031-98284-2_20 | ES-KT-24 … | 0 | LNCS chapter exists in Crossref |
| badran2025leakage | 2025 | 10.48550/arXiv.2508.17092 | match | 200 | arXiv 2508.17092 |
| llmedu2025 | 2025 | 10.18653/v1/2025.findings-emnlp.743 | LLM Agents for Education… | 200 | EMNLP Findings 2025 |
| ni2026ikteia | 2026 | 10.1016/j.knosys.2026.116071 | match | 200 | KBS 344:116071 |
| foundationalassist2026 | 2026 | 10.48550/arXiv.2602.00070 | FoundationalASSIST… | 200 | arXiv 2602.00070 |
| graphaugmentedllm2025 | 2026 | 10.1109/MIS.2025.3642667 | match | 202 | IEEE Intelligent Systems |
| graphrag2025 | 2026 | 10.1609/aaai.v40i17.38479 | GraphRAG-Induced… | 0 | AAAI 2026 record in Crossref |
| agentcat2026 | 2026 | 10.48550/arXiv.2606.21832 | AgentCAT… via OpenAlex | 200 | arXiv fetch timed out in the automated pass; title confirmed via OpenAlex |

## Explicitly left unchanged

- `corbett1995bkt` — 1995, DOI 10.1007/BF01099821, Crossref match.
- `hgkt2025` — Applied Sciences article number 8617.
