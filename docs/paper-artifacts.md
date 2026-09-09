# Paper artefacts → scripts and result files

Map for the KBS draft (`paper/main.tex`). Checkpoints are gitignored; JSON
under `results/tables/` is allowlisted when it is a reported number.
Do **not** run `scripts/08_export_paper_tables.py` for the headline AUC
table: that exporter still writes the old graph-only $L{=}200$ format.

| Paper table / figure | Script | Result file |
|---|---|---|
| Fig. 1 attribution pipeline | TikZ | `paper/figures/fig_architecture.tex` |
| Table aliases | hand-maintained | `paper/tables/table_aliases.tex` |
| Table AUC XES3G5M (clean L=400, test-only rebuild) | `scripts/36_rescore_a2_testonly.py` + capacity-matched + pyKT CSVs | `results/tables/table5_testonly_rebuild.csv`, `a2_testonly_rescore.csv`, `qkc_capacity_matched_matrix.csv`, `a2_multiseed_summary.csv` |
| A2/A7 test-only rescore (server 08/09) | `scripts/36_rescore_a2_testonly.py` | `results/tables/a2_testonly_rescore.csv`, `a2_testonly_rescore_summary.json` |
| A3 Δt-at-repeats diagnosis | `scripts/37_a3_dt_repeat_diag.py` | `results/tables/a3_dt_repeat_dist.json` |
| Table protocol residual (A3) | stored rescores | `p0_dt_testonly_clean.csv`, `p0_dt_fold{1,2}_testonly_clean.csv`, `xes3g5m_fold0_p0parity_protocol.csv` |
| Table attribution summary | `scripts/33_b8_holm_credit_tests.py`, `scripts/34_qkc_capacity_matched.py` | `results/tables/b8_holm_credit_tests.json`, `qkc_capacity_matched_summary.json` |
| Table QKC capacity-matched (app.) | `scripts/34_qkc_capacity_matched.py` | `results/tables/qkc_capacity_matched_{matrix.csv,summary.json}` |
| Table QKC zero-incidence + Δt (app.) | same (`zero_time` arm) | `paper/tables/table_qkc_zero_time.tex` |
| Table A1 T-real / T-zero / T-misaligned | `scripts/38_a1_timing_input_controls.py` | `results/tables/a1_timing_input_controls.csv`, `a1_timing_input_controls_summary.json` |
| Linear $\Delta t$ start-to-start gaps (B1) | P0 `_normalise_xes_timestamp_ms` then `log_time_gaps` | raw `timestamps` (ms start of answering) → parquet unix seconds → $\log(1+\Delta t)$ |
| Table A7 history-only Δt | `scripts/31_a7_query_dt_ablation.py` | `results/tables/a7_query_dt_ablation.csv`, `paper/tables/table_a7_query_dt.tex` |
| B6 DH²/pyKT occurrence join | `scripts/39_b6_join_prediction_ids.py` | `results/tables/b6_id_join_summary.json` |
| A4 event identity (occurrence join is not a stable ID) | `scripts/44_a4_event_identity.py` | `results/tables/a4_event_identity.json`, `b6_label_mismatch_rows.csv`, `b6_unmatched_pykt_rows.csv`; tests `tests/test_a4_event_identity.py` |
| A3 chunk-boundary safety | `scripts/43_a3_chunk_boundary_safety.py` | `results/tables/a3_chunk_boundary_safety.json`; tests `tests/test_a3_attempt_safe.py` |
| B8 Zero+Δt ECE | `scripts/41_b8_zero_dt_calibration.py` | `results/tables/a6_calibration_xes3g5m_fold0.csv`, `a6_reliability_zero_dt.json` |
| B9 same-ckpt destruction | `scripts/42_b9_same_ckpt_destruction.py` | `results/tables/b9_same_ckpt_destruction.{csv,json}` |
| Table AUC folds 1–2 (QKC-T only) | `scripts/03_train_tier1.py` | `paper/tables/table_auc_p0_dt_folds.tex` |
| Table calibration (DH²) | `scripts/30_a6_calibration_metrics.py` | `results/tables/a6_calibration_xes3g5m_fold0.csv` |
| Table calibration (pyKT) | `scripts/35_pykt_calibration_from_npz.py` | `results/tables/a6_calibration_pykt_xes3g5m_fold0.csv` |
| Table bootstrap (app.) | `scripts/25_bootstrap_protocol_ci.py` | `results/tables/b10_bootstrap_qkct_vs_{gikt,akt,simplekt}.csv`, `paper/tables/table_bootstrap_qkct.tex` |
| P4 concept-forget 5-seed | `scripts/03_train_tier1.py` (`--concept-forget`) | `results/tables/p4_forget_5seed.csv` |
| M4 teacher-group 5-seed | `scripts/03_train_tier1.py` (`--group-embed`) | `results/tables/m4_group_5seed.csv` |
| Fig. reliability | `scripts/_plot_reliability_xes3g5m.py` | `paper/figures/fig_reliability_xes3g5m.pdf` |
| Table negative knowledge (app.) | ablation drivers in `scripts/03_train_tier1.py` | `paper/tables/table_negative_knowledge.tex` |
| Table P4 raw 5-seed | P4 forget runs | `results/tables/p4_forget_5seed.csv`, `paper/tables/table_p4_raw.tex` |
| Fig. manipulation sweep | `scripts/09_export_paper_figures.py` | `results/tables/b9_same_ckpt_destruction.csv` |
| Table leakage taxonomy / measured | `scripts/29_a4_audit_positive_control.py`; $|\rho|$ status in `dh2a_kt/hyperedge/audit.py` | `results/tables/xes3g5m_fold0_hyperedge_audit_positive_control.json` |
| Table FoundationalASSIST | `scripts/03_train_tier1.py` | `results/tables/dh2_kt_foundational_assist.csv` |
| Table Junyi GT overlap | P0 GT overlap | `paper/tables/table_junyi_gt.tex` |
| Table node-drop (v2, app.) | `scripts/05_run_manipulation_check.py` | `results/tables/xes3g5m_fold{0,1,2}_manipulation_check.json` |
| Table B12 rewiring / permute | `scripts/32_b12_structure_checks.py` | `results/tables/xes3g5m_foldN_manipulation_check_*.json` |
| Fig. manipulation sweep | `scripts/09_export_paper_figures.py` | `paper/figures/fig_manipulation_sweep.pdf` |
| Table hparams | recorded configs | `paper/tables/table_hparams.tex` |
| Table hint IPW (supplement) | `scripts/14_run_hint_ate.py` | `results/tables/foundational_assist_fold0_hint_ate.json` |
| Table Tier-2 flags (supplement) | `scripts/18_run_kbs_faithfulness_suite.py` | `results/tables/xes3g5m_fold0_kbs_faithfulness_comparison.json` |
| Table KC-Jaccard / IG (supplement) | same | same JSON |
| Table paired diagnostics (supplement) | same | same JSON |
| Table human Likert n=100 (supplement) | pack in `results/tables/tier2_human_eval_tracer_n100/` | `paper/tables/table_human_eval.tex` |
| P0 overlap / evaluator map | hand-maintained | `docs/p0_overlap_disclosure.md` |
| Supplementary Material (Critic + IPW) | hand-maintained | `paper/supplement.tex` |
| Table graph-only v2 AUC (app.) | `scripts/05_run_manipulation_check.py` | `paper/tables/table_auc_xes3g5m_v2.tex` |
| Holm / paired credit tests | `scripts/33_b8_holm_credit_tests.py` | `results/tables/b8_holm_credit_tests.json` |
| Highlights | hand-maintained | `paper/highlights.txt` |

Capacity-matched Q←KC: incidence **0/5 PASS**, mean Δval ≈ −0.0001;
Linear Δt on zero incidence **5/5 PASS**, mean Δval +0.00255
(test-only Δ +0.00269).
P4 concept-forget **0/5 PASS**, mean Δval −0.00252;
M4 teacher-group **0/5 PASS**, mean Δval +0.00025.
pyKT ECE (seed 42, n=1,093,720): GIKT 0.0073 / AKT 0.0047 / simpleKT 0.0100.
A1 aligned-Δt isolation **0/5** (mean Δval +0.00183 vs T-zero, +0.00144 vs T-misaligned).
B6 join n=1,093,718; n_only_dh2=36; n_only_pykt=2; 8 occurrence collisions.
Stable key (user|timestamp|item|kc) on GPU: n_common=1,093,718; 0 label mismatch;
n_only_dh2_stable=37; n_only_pykt_stable=2. Table 5 / Appendix C vs pyKT remain
contextual, not event-paired.
B8 Zero+Δt ECE 0.0072 (n=1,093,755). B9 same-clean PASS (2.0e-9).
A2: 208 chunk windows start on a global repeat. A7 seed-42 lstm/query now present.
