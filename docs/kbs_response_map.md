# Internal map: KBS reviews to this revision

Not for submission. Source: `KBS_reviewer/KNOSYS-D-26-22180-reviews.pdf`. Target venue of this revision: Expert Systems (Wiley). Applied Intelligence is not the target: that journal rejected the pairwise paper P0.

| Comment | Change | Where |
|---|---|---|
| R2.1 Motivation unclear; reads as an engineering report | Protocol stated before the tracer. Each module has one hypothesis. | `paper/sections/protocol.tex`; `paper/tables/table_module_hypothesis.tex`; Introduction in `paper/main.tex` |
| R2.2 Experiments hard to follow | One decision table in the protocol section. Seed-level and Holm detail stay with the numeric sections. | `paper/tables/table_credit_decisions.tex` |
| R2.2 Baselines old | Two pyKT 2023–2025 models are a context block, not a leaderboard. Driver: `scripts/50_credit_ladder_plan.py` (`recent-baselines`). Not claimed until the runs exist. | Plan P2; paper still labels Table AUC as contextual |
| R4.W1 / Q1 ASSISTments and Junyi | ASSIST fold-0 ladder is P1. Junyi is a fixed subsample (~50k users), partial replication, P3, not the full corpus. | `scripts/50_credit_ladder_plan.py` |
| R4.W1 backbone transfer | Time package on a second backbone (simpleKT or AKT) is P1. | same driver, group `backbone` |
| R4.W2 / Q2 threshold and package capacity | Default `+0.002` inside `(+0.00175, +0.00236]`. Relative cut `max(0.002, 20·SD_seed)`. Package reports 256 added parameters. Param-padded arm is P2 and is not claimed until run. | `paper/sections/protocol.tex` |
| R4.W3 Unpublished companion, dense names | Reused pairwise steps restated in `sec:p0-selfcontained`. Deposit checklist: `docs/p0_arxiv_deposit.md`. No arXiv id is invented. | `paper/main.tex`; `docs/p0_arxiv_deposit.md` |
| R4.W4 Narrow positive result | Discussion states why incidence, aligned gap, and the structural arms fail locally. | `sec:auc-tradeoff` in `paper/main.tex` |
| R4.Q3 Positive but below the threshold | Outcome name: consistent-positive, not credited. | `paper/sections/protocol.tex` |
| R4.Q4 How to add a message | Algorithm in the protocol section and `docs/REGISTER_MESSAGE.md`. | both |
| Venue | KBS rejection disclosed in `paper/cover_letter_expert_systems.md`. P0 rejection by APIN and current EAAI review disclosed there and in `sec:p0-selfcontained`. | cover letter |

Transfer AUC numbers are not written into the paper until `results/tables/credit_ladder_transfer.json` exists.
