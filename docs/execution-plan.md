# Kế hoạch thực hiện dự án DH2A-KT

> Đây là kế hoạch **thi công phần mềm**, bám theo trạng thái thực tế của code trong repo (không phải bản lặp lại kế hoạch nghiên cứu cấp cao ở `docs/idea-D-plan.md`). Mỗi milestone dưới đây gắn với file/module cụ thể đã tồn tại trong `dh2a_kt/`, có gate nghiệm thu bằng `pytest`, và map ngược lại đúng "Pha" trong `docs/idea-D-plan.md` để không lệch khỏi kế hoạch gốc.

## 0. Trạng thái thực tế (cập nhật 6 Aug 2026)

### 0.1. Milestone Tier 1 + Tier 2

| Milestone | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| **M0** — Môi trường & dữ liệu | **Done** | Parquet processed: XES3G5M, Junyi, ASSIST2012 |
| **M1** — Concept-prerequisite hyperedge | **Done** | XES3G5M fold 0: 446,305 chain hyperedges, `ecr_flag=0` |
| **M2** — Session hyperedge | **Skipped** | thiếu ES-KT-24 |
| **M3** — `DH2KT.forward()` | **Done** | graph-only interaction + HypergraphConv ×2 |
| **M4** — Causal layer | **Done** | `tests/test_causal_layer_integration.py` pass |
| **M5** — Train + so P0 | **Done (v2)** | mean AUC **0.7516** vs GKT **0.8336** → `results/tables/dh2_kt_vs_p0.csv` |
| **M6** — Manipulation check | **PASS** | `auc_drop=0.0380` → `results/tables/xes3g5m_fold0_manipulation_check.json` |
| **M7** — GT cross-val Junyi | **Done** | 3 folds |
| **M8** — Tier 2 pilot | **Done** | stub500 `0.0`; Ollama v1 **0.124**; Ollama v2 calibrated **0.000** (500) |
| **M9** — Đánh giá Tier 2 | **Done** | stub + Ollama v1/v2 eval → `docs/M9-results-summary.md` |
| **Tests** | **42/42 pass** | `python -m pytest tests/ -q` |

### 0.2. Kiểm kê module (snapshot code)

| Module | Trạng thái | Bằng chứng |
|---|---|---|
| `dh2a_kt/p0_bridge.py` | Chạy được | `tests/test_p0_bridge.py` |
| `dh2a_kt/hyperedge/construction.py` | Chạy được (concept-prerequisite); session/forum/teacher vẫn `NotImplementedError` | `tests/test_hyperedge_audit.py`, `tests/test_build_hyperedges_real_data.py` |
| `dh2a_kt/hyperedge/indexing.py` | Chạy được | `tests/test_hyperedge_indexing.py` — destroy hyperedges cho M6 |
| `dh2a_kt/hyperedge/audit.py` | Chạy được | `tests/test_hyperedge_audit.py` |
| `dh2a_kt/baselines/loader.py` | Chạy được | `tests/test_baselines_loader.py` |
| `dh2a_kt/models/causal_layer.py` | Chạy được + nối pipeline | `tests/test_causal_layer.py`, `tests/test_causal_layer_integration.py` |
| `dh2a_kt/models/dh2_kt.py::DH2KT` | Chạy được (graph-only v2) | `tests/test_dh2_kt.py`; train/eval trên GPU |
| `dh2a_kt/train/tier1.py` | Chạy được | chain hyperedges, graph sensitivity loss, concept state cache |
| `dh2a_kt/eval/tier1_eval.py` | Chạy được | M6 dùng `trained.clean_hyperedges` |
| `dh2a_kt/eval/gt_crossval.py` | Chạy được | `tests/test_gt_crossval.py` |
| `dh2a_kt/agents/*` | Interface cố định | Cần `LLMClient` backend thật (M8) |
| `scripts/02_build_hyperedges.py` | Chạy được | audit trên dữ liệu thật |
| `scripts/03_train_tier1.py` | Chạy được | `--all-folds --device cuda` |
| `scripts/05_run_manipulation_check.py` | Chạy được | M6 gate pass fold 0 |
| `scripts/04_run_tier2_pilot.py` | Chạy được | stub + ollama backends; checkpoint save/load |
| Dữ liệu processed | **Có** | `external/p0_leakage_audit/data/processed/*.parquet` |

### 0.3. M6 graph-inert fix (tóm tắt kỹ thuật)

Ban đầu model predict tốt qua **exercise embedding + concept residual**, bỏ qua hypergraph → M6 fail (`auc_drop≈0.0003`). Fix gồm:

1. Train trên **chain hyperedges** (446k) thay pairwise `e_pre`.
2. **Participation masking** — concept state chỉ nonzero cho node có trong ≥1 hyperedge.
3. **Graph-only interaction** — `x_t = concept_proj(c_t) + response_proj(r_t)`, không cộng `exercise_embed`.
4. **Graph sensitivity loss** (`graph_sensitivity_weight=1.0`, `p=0.9`) khớp regime M6.

Tradeoff: `auc_clean` giảm (~0.75 vs ~0.79 v1) nhưng model **graph-reliant** — điều kiện cần để diễn giải ablation.

Kết luận: **Tier 1 code path (M0–M7 trừ M2/M8) đã chạy end-to-end trên GPU.** Việc còn lại chủ yếu là Tier 2 pilot (M8).

## 1. Nguyên tắc thi công

1. Không viết đè lên các module đã có test pass (M1-trở đi chỉ **thêm** logic thật vào chỗ đang `NotImplementedError`, không refactor lại phần đã chạy được trừ khi test hỏng).
2. Mỗi milestone đóng bằng cách thêm test mới vào `tests/`, không đóng bằng "chạy tay thấy có vẻ đúng".
3. Bất cứ khi nào một milestone cần dữ liệu chưa có (Forum/Teacher), dừng đúng tại ranh giới đó và ghi vào "Acknowledged Limitations" của bản thảo — không tự chế dữ liệu giả để né `NotImplementedError`.
4. Trước M5 (train Tier 1), bắt buộc chạy lại `scripts/01_verify_p0_preprocessing_match.py` — nếu preprocessing lệch khỏi P0, việc so baseline ở M5 vô nghĩa.

## 2. Lộ trình theo milestone

### M0 — Môi trường & dữ liệu thật
**Việc cụ thể**
- Cài đầy đủ `external/p0_leakage_audit/requirements.txt` (torch, scipy, matplotlib...).
- P0 vendor tĩnh không mang theo `third_party/pykt-toolkit` (bản thân P0 cũng chỉ tham chiếu nó qua submodule) — clone riêng: `git clone https://github.com/pykt-team/pykt-toolkit.git external/p0_leakage_audit/third_party/pykt-toolkit`, sau đó `pip install -e external/p0_leakage_audit/third_party/pykt-toolkit`.
- Tải raw data theo `external/p0_leakage_audit/README.md` mục 3 (link Google Drive P0 đã chia sẻ, hoặc tự tải XES3G5M/ASSISTments2012/Junyi từ nguồn gốc).
- Chạy tiền xử lý bằng chính code P0: `python -m src.preprocess --config configs/xes3g5m.yaml` (và tương tự cho assist2012, junyi) từ trong `external/p0_leakage_audit/`.
- Chạy `python -m src.split_checker --config configs/xes3g5m.yaml` — phải báo "no learner leakage".

**Gate nghiệm thu**: `scripts/01_verify_p0_preprocessing_match.py configs/xes3g5m.yaml` chạy OK **và** có file parquet thật dưới `external/p0_leakage_audit/data/processed/`.

**Rủi ro đã biết**: Junyi tiền xử lý nặng RAM (P0 tự ghi chú "memory-heavy, ưu tiên máy ~32GB RAM"). Nếu RTX 3090 gắn với máy RAM thấp hơn, cân nhắc chạy Junyi preprocessing riêng trên CPU trước khi động vào GPU.

---

### M1 — Concept-prerequisite hyperedge trên dữ liệu thật
**Việc cụ thể**: hoàn thiện `scripts/02_build_hyperedges.py` — thay đoạn `NotImplementedError` bằng: load `E_pre` đã audit của P0 (fold cụ thể) → `build_concept_prerequisite_hyperedges()` → `audit_hyperedges()` với `held_out_interaction_ids` lấy từ split thật.

**Gate nghiệm thu**: test mới `tests/test_build_hyperedges_real_data.py` (đánh dấu `@pytest.mark.slow`, cần dữ liệu M0) kiểm tra: số hyperedge > 0 trên XES3G5M fold 0, `HyperedgeLeakageReport.ecr_flag == 0.0` (không leakage cấu trúc).

---

### M2 — Session hyperedge (cần ES-KT-24)
**Việc cụ thể**: tải ES-KT-24, viết schema mapping Hint/Video vào `configs/`, hiện thực `build_session_hyperedges()` (bỏ `NotImplementedError` ở `construction.py` dòng 111).

**Gate nghiệm thu**: test mới chạy trên mẫu nhỏ ES-KT-24, audit hyperedge pass.

**Có thể làm song song với M1** nếu có thời gian, nhưng không bắt buộc trước M3 — Tier 1 vẫn train được chỉ với concept-prerequisite hyperedge (đúng phạm vi tối thiểu đã chốt ở `docs/idea-D-plan.md` mục 5: Student-Exercise-Concept-Hint(-Video)).

---

### M3 — Tier 1 model thật (`DH2KT.forward()`)
**Việc cụ thể**: cài `torch_geometric`; hiện thực trong `dh2a_kt/models/dh2_kt.py`:
1. Một `HypergraphConv` riêng cho mỗi loại hyperedge trong `config.hyperedge_kinds`.
2. Gộp bằng `HeteroConv`-style merge.
3. Cơ chế dual-gated theo thời gian (tham khảo HGKT, đã trích trong Phiên bản A mục 3.3).

**Gate nghiệm thu**: (a) unit test shape — forward pass trên batch giả không lỗi, output đúng shape `(batch, 1)`; (b) sanity overfit test — model overfit được một tập cực nhỏ (vài chục sample) trong vài trăm bước, loss giảm về gần 0 (chuẩn kiểm tra gradient chảy đúng trước khi train full).

---

### M4 — Nối causal layer vào pipeline Tier 1
**Việc cụ thể**: dùng embedding từ `DH2KT` làm confounder cho `PropensityScoreATE` (đã chạy được từ trước); treatment = có hint hay không (tạm dùng dữ liệu concept-prerequisite hyperedge vì chưa có Hint thật nếu M2 chưa xong); outcome = đúng/sai bước kế tiếp.

**Gate nghiệm thu**: test mới `tests/test_causal_layer_integration.py` — chạy `estimate_ate()` trên dữ liệu thật, log `propensity_min/max` để tự kiểm tra overlap (theo cảnh báo có sẵn trong code).

---

### M5 — Training loop Tier 1 + so sánh P0 (`scripts/03_train_tier1.py`)
**Việc cụ thể**: viết train/eval loop thật; **bắt buộc** khớp ngân sách epoch/batch với P0 cho model tương ứng (xem `configs/xes3g5m.yaml` mục `pykt:`/`baselines:` — ví dụ GKT: batch 4, max 10 epoch) trước khi so sánh; gọi `dh2a_kt.baselines.loader.compare_against_p0()`.

**Gate nghiệm thu**: bảng kết quả AUC ghi ra `results/tables/dh2_kt_vs_p0.csv`; nếu ngân sách không khớp được chính xác, ghi rõ trong báo cáo là so sánh "quan sát" (observational) — đúng cách P0 tự giới hạn phát biểu (P0 Section 5.3).

**Trạng thái (6 Aug 2026)**:
- v1 (pairwise + exercise): mean AUC **0.7874** vs GKT **0.8336** — archived baseline.
- v2 (chain + graph-only): mean AUC **0.7516** vs GKT **0.8336** (folds 0.7501/0.7518/0.7531) — **done**, tradeoff để pass M6.

---

### M6 — Manipulation check thật
**Việc cụ thể**: viết `eval_auc_fn` nối với model đã train ở M5, gọi `run_manipulation_check(hyperedges, eval_auc_fn=..., p=0.90)`.

**Gate nghiệm thu**: `ManipulationCheckResult.passes_manipulation_check` được ghi lại kèm `verdict` — nếu `False`, phải điều tra trước khi diễn giải bất kỳ kết quả ablation nào là "graph có tác dụng" (đúng cảnh báo đã viết sẵn trong `manipulation_check.py`).

**Trạng thái (5 Aug 2026)**: **PASS** trên XES3G5M fold 0 full (graph-only v2): `auc_clean=0.7527`, `auc_destroyed=0.7147`, `auc_drop=0.0380`, `ddr=0.9904`, verdict *"DH2-KT reads the hypergraph (graph-reliant)."* → `results/tables/xes3g5m_fold0_manipulation_check.json`.

---

### M7 — Ground-truth cross-validation mở rộng (Junyi)
**Việc cụ thể**: dùng `compute_overlap_metrics`/`diagnose_disagreement` (đã có qua `p0_bridge`) để đối chiếu concept-prerequisite hyperedge (không chỉ `E_pre` gốc) với expert DAG của Junyi.

**Gate nghiệm thu**: báo cáo F1/precision/recall theo top-K, định dạng tương đương P0 Table 15.

---

### M8 — Tier 2 pilot chạy thật
**Việc cụ thể**: chọn backend LLM (Qwen3-8B-Instruct quant qua vLLM/Ollama chạy local, hoặc API dự phòng — theo `docs/idea-D-plan.md`/Phiên bản C mục 7); hiện thực một lớp implement `dh2a_kt.agents.LLMClient`; chọn mẫu pilot 200-500 interaction; chạy vòng Diagnostician → Critic → Tutor/Hint; ghi log hyperedge mới do agent sinh ra qua `TutorHintAgent.write_session_hyperedge()`.

**Gate nghiệm thu**: pipeline chạy hết trên mẫu pilot không crash; log đầy đủ để tính hallucination rate ở M9.

---

### M9 — Đánh giá Tier 2 + tổng hợp bản thảo
**Việc cụ thể**: tính tỉ lệ `CriticVerdict.flagged` (proxy hallucination/drift), so khớp đường cong quên (nếu làm), viết case study; tổng hợp toàn bộ M0-M8 thành bản thảo theo cấu trúc đã có ở `docs/idea-D-plan.md`.

**Gate nghiệm thu**: bản thảo hoàn chỉnh, mọi con số trong bài trỏ được về đúng file kết quả (`results/*.csv`) — không có số nào không có nguồn.

## 3. Bảng milestone × tuần (map lại `docs/idea-D-plan.md`, đã trừ phần code-skeleton làm xong sớm)

| Milestone | Pha tương ứng (idea-D-plan) | Tuần ước tính | Phụ thuộc |
|---|---|---|---|
| M0 | Pha 0 (phần dữ liệu, chưa xong) | 1–2 | Băng thông tải dữ liệu, RAM máy |
| M1 | Pha 1 | 2–3 | M0 |
| M2 | Pha 1 | 3–4 (có thể song song M1) | ES-KT-24 |
| M3 | Pha 2 | 4–6 | M1 (+M2 nếu muốn Video ngay) |
| M4 | Pha 2 | 6–7 | M3 |
| M5 | Pha 3 | 7–9 | M3, M4, GPU RTX 3090 |
| M6 | Pha 3 | 9–10 | M5 |
| M7 | Pha 3 | 9–10 (song song M6) | M1, Junyi đã tiền xử lý |
| M8 | Pha 4 | 11–12 | M5, LLM backend |
| M9 | Pha 5–6 | 12–15 | M6, M7, M8 |

## 4. Việc cần làm tiếp (6 Aug 2026)

1. ~~LaTeX bản thảo~~ — Results/Discussion/Conclusion + tables + Fig. 1–2 trong `paper/main.tex` / `paper/main.pdf` (7 trang).
2. **Bổ sung theo review TLT** — xem [`docs/paper-revision-plan-tlt.md`](paper-revision-plan-tlt.md) (MAJOR REVISION):
   - ~~**P0 / Phase A**~~ — **Xong** (6 Aug 2026): xóa `[TODO]`; bib sạch (0 verify/TBD); §P0 self-contained + note supplementary.
   - ~~**P1 / Phase B**~~ — **Xong** (6 Aug 2026): Abstract/C khớp phạm vi; discussion AUC vs graph-reliance; Junyi GT-only + F1 underlap; Fig.2 threshold; leakage cap caption.
   - **P2 / Phase C** — **Xong** (7 Aug 2026): FoundationalASSIST session+Hint + AUC/ablation + limited ATE + C-Human A1/B1. Còn Phase D: cover letter + Short/Regular + commit/push.
3. ~~Optional — M2 session hyperedge~~ — **Xong** trên FoundationalASSIST (`build_session_hyperedges`, 308k HE, 11.6% có Hint).
4. ~~Critic prompt calibration~~ — **Xong** (Ollama v2, 500 mẫu).
5. ~~M0–M9 software path~~ — **Xong**; pytest session tests thêm.
6. ~~M5 Junyi GPU train~~ — **Bỏ qua**.

**P2 artefacts (FoundationalASSIST fold 0):**
- Ingest: `scripts/12_ingest_foundational_assist.py` → parquet + `e_pre` (280 edges)
- Session HE: `results/tables/foundational_assist_fold0_session_hyperedges.json` (308{,}670; hint 11.6%)
- Train: `results/tables/dh2_kt_foundational_assist.csv` — **AUC 0.7196**
- Session ablation: concept-only AUC 0.7196 → +session 0.7220 (Δ+0.0024); `foundational_assist_session_ablation.json`
- Hint ATE: ATE_IPW=-0.207; trimmed=-0.211; CI95=[-0.230,-0.187]
- C-Human: A1+B1 — faithfulness **4.01±1.07**, usefulness **3.53±0.71**; Spearman 0.71/0.58; within±1 92.5%/97.5% (`human_eval_summary.json`)

## 5. Rủi ro kỹ thuật cụ thể (bổ sung, gắn trực tiếp với file)

| Rủi ro | File liên quan | Giảm thiểu |
|---|---|---|
| `_flatten_to_pairwise` trong `hyperedge/audit.py` là xấp xỉ (gán weight=1.0 đều cho mọi cặp) — có thể làm loãng tín hiệu `|rho|` thật | `dh2a_kt/hyperedge/audit.py` | Ghi rõ trong `notes` (đã có) khi báo cáo; nếu kết quả `|rho|` quan trọng cho luận điểm bài, cân nhắc viết thống kê hyperedge-native thay vì tái dùng nguyên xi hàm pairwise của P0 |
| `PropensityScoreATE` giả định không có confounder ẩn — vi phạm dễ xảy ra trên dữ liệu observational | `dh2a_kt/models/causal_layer.py` | Luôn báo cáo `propensity_min/max` cùng ATE; nêu rõ giả định nhận diện trong bài, không chỉ báo con số ATE |
| `DH2KT` phụ thuộc cứng vào `torch_geometric` — cài đặt trên Windows đôi khi cần đúng bản CUDA khớp | `dh2a_kt/models/dh2_kt.py` | Cài theo đúng lệnh chính thức của PyG cho bản CUDA của máy trước khi bắt đầu M3, test bằng `import torch_geometric` trước |
