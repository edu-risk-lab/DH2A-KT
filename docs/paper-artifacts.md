# Paper artefacts → scripts and result files

Map for the KBS draft (`paper/main.tex`). Checkpoints are gitignored; JSON
under `results/tables/` is allowlisted when it is a reported number.
Do **not** run `scripts/08_export_paper_tables.py` for the headline AUC
table: that exporter still writes the old graph-only $L{=}200$ format.

| Paper table / figure | Script | Result file |
|---|---|---|
| Fig. 1 attribution pipeline | TikZ | `paper/figures/fig_architecture.tex` |
| Table aliases | hand-maintained | `paper/tables/table_aliases.tex` |
| Table AUC XES3G5M (clean L=400) | `scripts/03_train_tier1.py`, `scripts/24_train_baselines_clean.py`, `scripts/27_a2_multiseed_matrix.py` | `results/tables/a2_multiseed_matrix.csv`, `results/tables/xes3g5m_fold0_baselines_protocol.csv` |
| Table attribution summary | `scripts/33_b8_holm_credit_tests.py`, `scripts/34_qkc_capacity_matched.py` | `results/tables/b8_holm_credit_tests.json`, `qkc_capacity_matched_summary.json` |
| Table QKC capacity-matched (app.) | `scripts/34_qkc_capacity_matched.py` | `results/tables/qkc_capacity_matched_{matrix.csv,summary.json}` |
| Table QKC zero-incidence + Δt (app.) | same (`zero_time` arm) | `paper/tables/table_qkc_zero_time.tex` |
| Table A7 history-only Δt | `scripts/31_a7_query_dt_ablation.py` | `results/tables/a7_query_dt_ablation.csv`, `paper/tables/table_a7_query_dt.tex` |
| Table AUC folds 1–2 (QKC-T only) | `scripts/03_train_tier1.py` | `paper/tables/table_auc_p0_dt_folds.tex` |
| Table calibration (DH²) | `scripts/30_a6_calibration_metrics.py` | `results/tables/a6_calibration_xes3g5m_fold0.csv` |
| Table calibration (pyKT) | `scripts/35_pykt_calibration_from_npz.py` | `results/tables/a6_calibration_pykt_xes3g5m_fold0.csv` |
| Bootstrap QKC-T vs GIKT/AKT/simpleKT | `scripts/25_bootstrap_protocol_ci.py` | `results/tables/b10_bootstrap_qkct_vs_{gikt,akt,simplekt}.csv` |
| P4 concept-forget 5-seed | `scripts/03_train_tier1.py` (`--concept-forget`) | `results/tables/p4_forget_5seed.csv` |
| M4 teacher-group 5-seed | `scripts/03_train_tier1.py` (`--group-embed`) | `results/tables/m4_group_5seed.csv` |
| Fig. reliability | same | `paper/figures/fig_reliability_xes3g5m.pdf` |
| Table negative knowledge (app.) | ablation drivers in `scripts/03_train_tier1.py` | `paper/tables/table_negative_knowledge.tex` |
| Table leakage taxonomy / measured | hyperedge audit | `paper/tables/table_leakage_audit.tex` |
| Table FoundationalASSIST | `scripts/03_train_tier1.py` | `results/tables/dh2_kt_foundational_assist.csv` |
| Table Junyi GT overlap | P0 GT overlap | `paper/tables/table_junyi_gt.tex` |
| Table node-drop (v2, app.) | `scripts/05_run_manipulation_check.py` | `results/tables/xes3g5m_fold{0,1,2}_manipulation_check.json` |
| Table B12 rewiring / permute | `scripts/32_b12_structure_checks.py` | `results/tables/xes3g5m_foldN_manipulation_check_*.json` |
| Fig. manipulation sweep | `scripts/09_export_paper_figures.py` | `paper/figures/fig_manipulation_sweep.pdf` |
| Table hparams | recorded configs | `paper/tables/table_hparams.tex` |
| Table hint IPW (app.) | `scripts/14_run_hint_ate.py` | `results/tables/foundational_assist_fold0_hint_ate.json` |
| Table Tier-2 flags (app.) | `scripts/18_run_kbs_faithfulness_suite.py` | `results/tables/xes3g5m_fold0_kbs_faithfulness_comparison.json` |
| Table KC-Jaccard / IG (app.) | same | same JSON |
| Table paired diagnostics (app.) | same | same JSON |
| Table human Likert n=100 (app.) | pack in `results/tables/tier2_human_eval_tracer_n100/` | `paper/tables/table_human_eval.tex` |
| Table graph-only v2 AUC (app.) | `scripts/05_run_manipulation_check.py` | `paper/tables/table_auc_xes3g5m_v2.tex` |
| Holm / paired credit tests | `scripts/33_b8_holm_credit_tests.py` | `results/tables/b8_holm_credit_tests.json` |
| Highlights | hand-maintained | `paper/highlights.txt` |

Capacity-matched Q←KC: incidence **0/5 PASS**, mean Δval ≈ −0.0001;
Linear Δt on zero incidence **5/5 PASS**, mean Δval +0.00255
(test-only Δ +0.00269).
