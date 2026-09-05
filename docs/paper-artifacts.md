# Paper artefacts → scripts and result files

Map for the KBS draft (`paper/main.tex`). Checkpoints are gitignored; JSON
under `results/tables/` is allowlisted when it is a reported number.

| Paper table / figure | Script | Result file |
|---|---|---|
| Table AUC XES3G5M (clean L=400) | `scripts/03_train_tier1.py`, pyKT retrains | `paper/tables/table_auc_xes3g5m.tex` |
| Table QKC capacity-matched | `scripts/34_qkc_capacity_matched.py` | `results/tables/qkc_capacity_matched_{matrix.csv,summary.json}` |
| Table A7 history-only Δt | `scripts/03_train_tier1.py --time-gap-mode lstm` | `paper/tables/table_a7_query_dt.tex` |
| Table calibration | `scripts/30_a6_calibration_metrics.py` | `paper/tables/table_calibration_xes3g5m.tex` |
| Fig. reliability | same | `paper/figures/fig_reliability_xes3g5m.pdf` |
| Table negative knowledge | ablation drivers in `scripts/03_train_tier1.py` | `paper/tables/table_negative_knowledge.tex` |
| Table node-drop (v2) | `scripts/05_run_manipulation_check.py` | `results/tables/xes3g5m_fold{0,1,2}_manipulation_check.json` |
| Table B12 rewiring | `scripts/32_b12_structure_checks.py` | `results/tables/xes3g5m_foldN_manipulation_check_degree_preserving_rewire.json` |
| Table B12 label permute | same | `..._relation_label_permute.json` |
| Fig. architecture | TikZ | `paper/figures/fig_architecture.tex` |
| Fig. manipulation sweep | `scripts/09_export_paper_figures.py` | `paper/figures/fig_manipulation_sweep.pdf` |
| Table hparams | recorded configs | `paper/tables/table_hparams.tex` |
| Table hint IPW (appendix) | `scripts/14_run_hint_ate.py` | `results/tables/foundational_assist_fold0_hint_ate.json` |
| Table human Likert n=100 | pack in `results/tables/tier2_human_eval_tracer_n100/` | `paper/tables/table_human_eval.tex` |
| Holm B8 | `scripts/33_b8_holm_credit_tests.py` | `results/tables/b8_holm_credit_tests.json` |

B12 fold~0–2 rewiring is on `main` (ΔAUC 0.0318 / 0.0341 / 0.0351). Capacity-matched
Q←KC (`scripts/34_qkc_capacity_matched.py`) finished: **0/5 PASS**, mean Δval ≈ −0.0001
(do not reuse the historical pathway-off twin for incidence credit).
