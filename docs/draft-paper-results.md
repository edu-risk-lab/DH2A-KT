# Bản thảo — Mục Results & Discussion (DH²A-KT + P0)

> Skeleton bài báo, bám `docs/idea-D-plan.md`. Mọi số trích từ `results/tables/` hoặc `docs/M9-results-summary.md`. Cập nhật: 6 Aug 2026.

---

## Abstract (draft)

We present DH²A-KT, a two-tier knowledge tracing framework built on the leakage-controlled graph audit protocol of P0. Tier 1 encodes concept-prerequisite structure as hyperedges and enforces graph reliance via a manipulation check; Tier 2 wraps frozen Tier-1 predictions with Diagnostician–Critic–Tutor agents. On XES3G5M (primary), DH²A-KT v2 (chain hyperedges, graph-only interaction) achieves mean test AUC 0.752 versus GKT 0.834 under matched training budget, while passing the graph destruction check (ΔAUC = 0.038 at p = 0.9 node-drop). On Junyi, concept-prerequisite hyperedges align with the expert DAG at edge-F1 ≈ 0.17 (comparable to P0’s E_pre at ≈ 0.18). A 500-sample Tier-2 pilot with Qwen2.5-7B (Ollama) yields Critic flag rates of 12.4% (strict prompt) and 0% after probability-format calibration.

---

## 5. Experimental Setup (brief)

**Primary dataset:** XES3G5M — learner split 0.7/0.1/0.2, three folds (seeds 42/43/44), inherited from P0 (`configs/xes3g5m.yaml`).

**Model:** DH²-KT with two HypergraphConv layers on chain hyperedges derived from audited E_pre (446,305 hyperedges, fold 0); graph-only interaction path (no exercise embedding in the interaction term) to prevent graph-inert bypass.

**Training budget:** Matched to P0 GKT — batch size 4, 10 epochs, max sequence length 200 (`results/tables/dh2_kt_vs_p0.csv`).

**Tier 2:** 500 eval interactions sampled from fold-0 valid+test; LLM = Qwen2.5-7B (Q4_K_M) via Ollama 0.32.6; mandatory Critic gate on Diagnostician output.

**Software:** PyTorch + PyG; 42/42 unit tests pass.

---

## 6. Results

### 6.1 RQ1 — Predictive performance vs P0 baselines (XES3G5M)

**Source:** `results/tables/dh2_kt_vs_p0.csv`

Under matched budget, DH²-KT v2 mean AUC is **0.7516** (folds: 0.7501, 0.7518, 0.7531) versus GKT **0.8336** and simpleKT **0.8747** (P0 reused baselines). The gap reflects the deliberate removal of direct exercise-embedding signal in exchange for graph reliance (Section 6.2).

*Interpretation for the paper:* DH²-KT is not positioned as a pure AUC improvement over strong graph baselines on XES3G5M; the claim is methodological — hypergraph structure with leakage control and demonstrable graph use.

### 6.2 RQ2 / Manipulation check — Does the model read the hypergraph?

**Source:** `results/tables/xes3g5m_fold0_manipulation_check.json`

| Condition | AUC |
|-----------|-----|
| Clean hypergraph | 0.7527 |
| Destroyed (p=0.9 node-drop) | 0.7147 |
| **ΔAUC** | **0.0380** |

DDR = 0.990; `passes_manipulation_check = true`. Early graph-inert architectures (pairwise + exercise path) failed this gate (ΔAUC ≈ 0.0003). The graph-only redesign was required for a positive manipulation check.

### 6.3 RQ5 — Hyperedge leakage audit (concept-prerequisite)

**Source:** `results/tables/xes3g5m_fold0_hyperedge_audit.json`

446,305 chain hyperedges; ECR^flag = 0; group-membership leak rate = 0. Pairwise TBMR/|ρ| skipped due to projection cap (noted in audit JSON).

Junyi fold 0: 700,570 hyperedges, ECR^flag = 0 (`results/tables/junyi_fold0_hyperedge_audit.json`).

### 6.4 Ground-truth cross-validation (Junyi)

**Source:** `results/tables/junyi_fold0_gt_crossval/overlap_metrics_summary.json` (folds 1–2 analogous)

At top-K = |expert| = 1131:

| Representation | Edge F1 |
|----------------|---------|
| P0 E_pre | 0.183 |
| Hyperedge chain | 0.165 |

Hyperedge chain slightly underperforms pairwise E_pre on edge overlap but remains in the same order of magnitude; direction agreement and reachability metrics in the same JSON support qualitative comparison with P0 Table 15 format (`gt_crossval_table15_format.csv`).

**Note:** Full DH²-KT training on Junyi was not run (25M interactions; P0 disables GKT on Junyi). Junyi’s role in this study is structural validation, not primary predictive benchmarking.

### 6.5 Tier 2 — Agent faithfulness pilot

**Sources:**
- Stub (pipeline sanity): `results/tables/xes3g5m_fold0_tier2_pilot_summary.json` — 500 samples, flag_rate = 0.0
- Ollama Qwen2.5-7B v1: `results/tables/xes3g5m_fold0_tier2_pilot_ollama_summary.json` — flag_rate = **0.124** (62/500)
- Ollama v2 (calibrated Critic): `results/tables/xes3g5m_fold0_tier2_pilot_ollama_v2_summary.json` — flag_rate = **0.000** (0/500)
- Eval + case studies: `results/tables/xes3g5m_fold0_tier2_pilot_ollama_v2_eval.json`

Mean Tier-1 P(correct) on pilot samples: 0.746 (v1), **0.753** (v2). The Critic flagged explanations that reported probabilities as percentages (e.g. “69.3%”) when the gate expects decimal consistency with Tier-1 output — see `case_study_flagged` in the v1 eval JSON. **Critic calibration** (prompt v2 + `critic_calibration.py`) reduced flag_rate from **0.124** to **0.000** on the full 500-sample run (seed 42).

500 session hyperedges were written by TutorHintAgent (`*_session_hyperedges.json`), closing the agentic loop at pilot scale.

---

## 7. Discussion

**Graph–accuracy tradeoff.** Graph-only interaction reduces AUC ~8.2 points vs GKT but enables a passing manipulation check. For APIN-style claims, graph reliance should be reported alongside predictive metrics.

**Tier 2 / C-Human (A1+B1).** Faithfulness **4.01±1.07**, usefulness **3.53±0.71** (n=40×2). Spearman 0.71 / 0.58; within±1 92.5% / 97.5%. Summary: `results/tables/tier2_human_eval/human_eval_summary.json`.

**P2 / FoundationalASSIST (6–7 Aug 2026).** Session+Hint hyperedges: 308{,}670 (11.6% with Hint). Fold-0 DH²-KT AUC **0.7196**. Hint ATE (IPW, next-step correct): **−0.207** (naive −0.255); propensity `[0.05, 0.219]` — weak overlap, report as limited observational evidence only.

**Limitations.**
- Discussion / teacher hyperedges still blocked; session+Hint on FoundationalASSIST only.
- FoundationalASSIST AUC observational (no P0-reused baseline column).
- Critic v1 sensitivity to response format inflated flags; v2 calibration resolves this on the 500-sample pilot.
- Windows + single RTX 3090 reproduction environment.

---

## 8. Reproducibility checklist

| Step | Command / artefact |
|------|-------------------|
| Tier 1 train | `scripts/03_train_tier1.py configs/xes3g5m.yaml --all-folds --device cuda` |
| M6 | `scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 0 --device cuda` |
| M7 Junyi | `scripts/06_run_gt_crossval_junyi.py configs/junyi.yaml` |
| Tier 2 pilot | `scripts/04_run_tier2_pilot.py` + checkpoint in `results/checkpoints/` |
| M9 eval | `scripts/07_eval_tier2_pilot.py` |
| Number index | `docs/M9-results-summary.md` |

---

## Figures

1. **Fig. 1** — Two-tier architecture → `paper/figures/fig_architecture.tex` (TikZ).
2. **Fig. 2** — M6 $\Delta$AUC sweep → `paper/figures/fig_manipulation_sweep.pdf` (regen: `python scripts/09_export_paper_figures.py`).

## LaTeX tables (generated)

Run `python scripts/08_export_paper_tables.py` → `paper/tables/*.tex`; included from `paper/main.tex`:

3. **Table 1** — XES3G5M AUC vs P0 baselines (`table_auc_xes3g5m.tex`).
4. **Table 2** — Junyi GT overlap (`table_junyi_gt.tex`).
5. **Table 3** — Tier 2 flag_rate stub vs Ollama v1/v2 (`table_tier2.tex`).
6. **Table 4** — Manipulation check (`table_manipulation.tex`).
7. **Table 5** — Leakage audit (`table_leakage_audit.tex`).
