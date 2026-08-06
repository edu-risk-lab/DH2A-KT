# M9 — Tổng hợp kết quả DH2A-KT (Tier 1 + Tier 2)

> Cập nhật: 6 Aug 2026. Mọi con số dưới đây trích từ artefact trong `results/tables/` (hoặc `results/checkpoints/`). Dùng làm nguồn khi viết bản thảo — không trích số không có file nguồn.

## 1. Tóm tắt executive

| Khối | Kết luận chính | Gate |
|------|----------------|------|
| Tier 1 KT | DH2-KT v2 (chain + graph-only) mean AUC **0.7516** vs GKT **0.8336** | M5 |
| Graph reliance | M6 **PASS**: `auc_drop=0.0380` | M6 |
| Hyperedge audit | 446,305 chain hyperedges, `ecr_flag=0` | M1 |
| GT alignment (Junyi) | E_pre F1≈0.183; hyperedge chain F1≈0.165 (@K=\|expert\|) | M7 |
| Tier 2 agents | Ollama v1 **0.124** → v2 calibrated **0.000** (500/500) | M8–M9 |

**Tradeoff đã chấp nhận:** graph-only interaction giảm AUC ~3.6 điểm phần trăm so với kiến trúc cũ (exercise embed) nhưng pass manipulation check — model **graph-reliant**.

---

## 2. Tier 1 — XES3G5M (primary)

### 2.1 M5 — Predictive performance (v2 graph-only)

**Nguồn:** `results/tables/dh2_kt_vs_p0.csv`

| Fold | DH2-KT AUC | GKT (P0) | Δ vs GKT |
|------|------------|----------|----------|
| 0 | 0.7501 | 0.8336 | −0.0836 |
| 1 | 0.7518 | 0.8336 | −0.0819 |
| 2 | 0.7531 | 0.8336 | −0.0806 |
| **Mean** | **0.7516** | **0.8336** | **−0.0820** |

- Ngân sách train khớp P0 GKT: batch=4, epochs=10, max_seq_len=200.
- Kiến trúc: chain hyperedges + graph-only interaction (không `exercise_embed` trên interaction path).

### 2.2 M6 — Manipulation check (fold 0 full)

**Nguồn:** `results/tables/xes3g5m_fold0_manipulation_check.json`

| Metric | Giá trị |
|--------|---------|
| auc_clean | 0.7527 |
| auc_destroyed | 0.7147 |
| **auc_drop** | **0.0380** |
| ddr | 0.9904 |
| passes_manipulation_check | **true** |
| operator / p | node_drop / 0.9 |

Verdict: *"DH2-KT reads the hypergraph (graph-reliant)."*

### 2.3 M1 — Hyperedge construction & audit (fold 0)

**Nguồn:** `results/tables/xes3g5m_fold0_hyperedge_audit.json`

| Metric | Giá trị |
|--------|---------|
| n_hyperedges | 446,305 |
| ecr_flag | 0.0 |
| group_membership_leak_rate | 0.0 |

---

## 3. Tier 1 — Junyi (M7 GT cross-validation)

**Nguồn:** `results/tables/junyi_fold0_gt_crossval/overlap_metrics_summary.json` (fold 0; fold 1–2 tương tự trong `junyi_fold1_gt_crossval/`, `junyi_fold2_gt_crossval/`)

@ top-K = |expert| = 1131:

| Graph | edge F1 | edge precision | edge recall |
|-------|---------|----------------|-------------|
| E_pre (P0) | 0.183 | 0.183 | 0.183 |
| Hyperedge chain | 0.165 | 0.165 | 0.165 |

Bảng Table-15 format: `results/tables/junyi_fold0_gt_crossval/gt_crossval_table15_format.csv`

---

## 4. Tier 2 — Agent pilot (M8–M9)

**Checkpoint Tier 1:** `results/checkpoints/xes3g5m_fold0.pt` (~156 MB, fold 0, graph-only v2)

### 4.1 Stub backend (sanity / pipeline gate)

| Metric | Giá trị | Nguồn |
|--------|---------|-------|
| n_samples | 500 | `xes3g5m_fold0_tier2_pilot_summary.json` |
| flag_rate | 0.0 | idem |
| mean P(correct) | 0.742 | `xes3g5m_fold0_tier2_pilot_eval.json` |
| session hyperedges | 500 | summary |

Logs: `results/tables/xes3g5m_fold0_tier2_pilot.jsonl`

### 4.2 Ollama / qwen2.5:7b (LLM thật)

| Run | n | flag_rate | Nguồn |
|-----|---|-----------|-------|
| v1 (prompt strict) | 500 | **0.124** (62/500) | `xes3g5m_fold0_tier2_pilot_ollama_summary.json` |
| v2 (calibrated Critic) | 500 | **0.000** (0/500) | `xes3g5m_fold0_tier2_pilot_ollama_v2_summary.json` |

mean P(correct) v1: 0.746 · v2 full: **0.753** → `xes3g5m_fold0_tier2_pilot_ollama_v2_eval.json`

**Critic calibration:** `dh2a_kt/agents/critic_calibration.py` — chấp nhận %/decimal tương đương; giảm false FLAG format từ **12.4%** (v1, 500) xuống **0%** (v2, 500; cùng seed 42).

---

## 5. Kiểm thử phần mềm

**Nguồn:** `python -m pytest tests/ -q` → **42/42 pass** (tại thời điểm M9).

---

## 6. Acknowledged limitations (ghi trong bài)

1. **M2 skipped** — không có session hyperedge (thiếu ES-KT-24 Hint/Video đồng bộ).
2. **AUC tradeoff** — graph-only pass M6; không claim SOTA predictive trên XES3G5M.
3. **Tier 2 pilot** — một fold, một LLM (qwen2.5:7b Q4); flag_rate phụ thuộc Critic prompt (format-sensitive).
4. **Junyi GT** — hyperedge chain F1 thấp hơn E_pre một chút @|expert|; không claim thay thế expert DAG.

---

## 7. Lệnh reproduce nhanh

```powershell
# Tier 1
python scripts/03_train_tier1.py configs/xes3g5m.yaml --all-folds --device cuda
python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 0 --device cuda

# Tier 2 (cần checkpoint + Ollama)
python scripts/04_run_tier2_pilot.py configs/xes3g5m.yaml --device cuda `
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt `
  --llm-backend ollama --ollama-model qwen2.5:7b --sample-size 500 --output-tag ollama

python scripts/07_eval_tier2_pilot.py results/tables/xes3g5m_fold0_tier2_pilot_ollama_v2_summary.json

# LaTeX tables
python scripts/08_export_paper_tables.py
# LaTeX figures
python scripts/09_export_paper_figures.py
# Compile: cd paper; pdflatex main; bibtex main; pdflatex main; pdflatex main
```

---

## 8. Việc còn lại (ngoài phạm vi M9 software gate)

- Viết bản thảo đầy đủ theo cấu trúc `docs/idea-D-plan.md` (prose + figures).
- Optional: M5 Junyi GPU; M2 nếu có ES-KT-24; thử LLM khác / Critic prompt calibration.
