# Internal map: KBS reviews to this revision

Not for submission. Source: `KBS_reviewer/KNOSYS-D-26-22180-reviews.pdf`. Target venue of this revision: Expert Systems (Wiley). Applied Intelligence is not the target: that journal rejected the pairwise paper P0.

| Comment | Change | Where |
|---|---|---|
| R2.1 Motivation unclear; reads as an engineering report | Protocol stated before the tracer. Each module has one hypothesis. | `paper/sections/protocol.tex`; `paper/tables/table_module_hypothesis.tex`; Introduction in `paper/main.tex` |
| R2.2 Experiments hard to follow | One decision table for XES and one for transfer. Seed-level and Holm detail stay with the numeric sections. | `paper/tables/table_credit_decisions.tex`; `paper/tables/table_credit_transfer.tex` |
| R2.2 Baselines old | Not done. sparseKT and stableKT have no forward in the vendored pyKT evaluator. Stated in Limitations; no substitute model. | `sec:limitations` |
| R4.W1 / Q1 ASSISTments and Junyi | ASSIST fold 0, one-second re-export (`scripts/54_build_assist_seconds.py`): under the XES recipe (`assist-sec-ref`) the package with zero incidence is credited on folds 0 and 2 (5/5) and consistent-positive on fold 1 (4/5, one seed +0.00116; groups `assist-sec-fold1/2`), and the aligned gap is consistent-positive on all three folds; under the corpus config (`assist-sec`) nothing is credited. The original 1,000 s export (`assist`, `assist-ref`) credits nothing and is kept as a sensitivity table (`tab:credit-assist-resolution`). Junyi all learners, XES recipe: package and aligned gap credited under both cuts. Junyi 50k subsample (corpus config) kept as an earlier, noisier run. | `sec:transfer-ladder`; abstract |
| R4.W1 backbone transfer | simpleKT/AKT path has no gap input. Causal-attention encoder in place of the LSTM, same inputs and readout, XES recipe: package 2/5 at +0.002, no time message credited. | `sec:transfer-ladder`; `sec:limitations` |
| R4.W2 / Q2 threshold and package capacity | Default `+0.002` inside `(+0.00175, +0.00236]`. The relative cut `max(0.002, 10·SD_seed)` is replaced by an interval rule: lower 95% bound of the mean paired Δ (t, df 4) at least +0.002; no fitted constant. Recipe-matched padded control: package 5/5 under the seed rule, lower bound +0.00227 (credited under both rules). Earlier recipe-mismatched padded runs are not used. | `paper/sections/protocol.tex`; `sec:transfer-ladder` |
| R4.W3 Unpublished companion, dense names | Reused pairwise steps restated in `sec:p0-selfcontained`. Deposit checklist: `docs/p0_arxiv_deposit.md`. No arXiv id is invented. | `paper/main.tex`; `docs/p0_arxiv_deposit.md` |
| R4.W4 Narrow positive result | Discussion states why incidence, aligned gap, and the structural arms fail locally, and that the map changes across corpora. | `sec:auc-tradeoff` |
| R4.Q3 Positive but below the threshold | Outcome name: consistent-positive, not credited. ASSIST incidence is an example. | `paper/sections/protocol.tex` |
| R4.Q4 How to add a message | Algorithm in the protocol section and `docs/REGISTER_MESSAGE.md`. | both |
| Venue | KBS rejection disclosed in `paper/cover_letter_expert_systems.md`. P0 rejection by APIN and current EAAI review disclosed there and in `sec:p0-selfcontained`. | cover letter |

Runs: ASSIST in `4e9cf7f`; Junyi partial in `829adab`; recipe-matched padded arm in `bbef2fd`; attention backbone and Junyi full in `673a0e2`; ASSIST under the XES recipe in `2f0131b`; ASSIST one-second ladders in `2e068ef`; ASSIST one-second folds 1–2 in `69a5d62`. Outcomes: `scripts/51_transfer_credit_tables.py` → `results/tables/credit_ladder_transfer.json`.
