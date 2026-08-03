# Kế hoạch thực hiện dự án DH2A-KT

> Đây là kế hoạch **thi công phần mềm**, bám theo trạng thái thực tế của code trong repo (không phải bản lặp lại kế hoạch nghiên cứu cấp cao ở `docs/idea-D-plan.md`). Mỗi milestone dưới đây gắn với file/module cụ thể đã tồn tại trong `dh2a_kt/`, có gate nghiệm thu bằng `pytest`, và map ngược lại đúng "Pha" trong `docs/idea-D-plan.md` để không lệch khỏi kế hoạch gốc.

## 0. Điểm xuất phát — kiểm kê thực tế (không phải ước lượng)

| Module | Trạng thái | Bằng chứng |
|---|---|---|
| `dh2a_kt/p0_bridge.py` | Chạy được | `tests/test_p0_bridge.py` — 2/2 pass |
| `dh2a_kt/hyperedge/construction.py::build_concept_prerequisite_hyperedges` | Chạy được (trên dữ liệu toy) | `tests/test_hyperedge_audit.py` — pass |
| `dh2a_kt/hyperedge/construction.py::build_session_hyperedges` | `NotImplementedError` | dòng 111 — chặn bởi thiếu dữ liệu Hint/Video đồng bộ |
| `dh2a_kt/hyperedge/construction.py::build_discussion_thread_hyperedges` | `NotImplementedError` | dòng 122 — chặn bởi thiếu dữ liệu Forum |
| `dh2a_kt/hyperedge/construction.py::build_teacher_intervention_hyperedges` | `NotImplementedError` | dòng 128 — chặn bởi thiếu dữ liệu Teacher |
| `dh2a_kt/hyperedge/audit.py` | Chạy được (group-membership leak); cross-modal/intervention-timing cần input thật | `tests/test_hyperedge_audit.py` — pass |
| `dh2a_kt/baselines/loader.py` | Chạy được, đầy đủ | `tests/test_baselines_loader.py` — 3/3 pass |
| `dh2a_kt/models/causal_layer.py::PropensityScoreATE` | Chạy được (chưa nối vào pipeline thật) | `tests/test_causal_layer.py` — 4/4 pass (dữ liệu tổng hợp, kiểm tra IPW estimate sát ground-truth ATE hơn naive mean-difference, có cảnh báo overlap kém) |
| `dh2a_kt/models/dh2_kt.py::DH2KT.forward()` | `NotImplementedError` | dòng 72 |
| `dh2a_kt/eval/manipulation_check.py` | Chạy được nhưng cần `eval_auc_fn` từ model đã train | — |
| `dh2a_kt/agents/*` (5 agent) | Interface cố định, cần `LLMClient` backend thật | — |
| `scripts/02_build_hyperedges.py`, `03_train_tier1.py`, `04_run_tier2_pilot.py` | `NotImplementedError` — glue-code chờ dữ liệu/model thật | — |
| Dữ liệu thô (XES3G5M/ASSISTments2012/Junyi) | **Chưa tải** | `external/p0_leakage_audit/data/raw/` chỉ có `.gitkeep` |

Kết luận: phần **viết code khung** cho Pha 0-1 (theo `docs/idea-D-plan.md`) đã xong sớm hơn dự kiến trong một phiên, nhưng phần **phụ thuộc thời gian thực** (tải dữ liệu, train trên GPU, chạy LLM) vẫn cần đúng khối lượng thời gian đã ước lượng trong kế hoạch gốc — code viết nhanh không rút ngắn được thời gian máy chạy.

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

**Gate nghiệm thu**: bảng kết quả AUC ghi ra `results/dh2_kt_vs_p0.csv`; nếu ngân sách không khớp được chính xác, ghi rõ trong báo cáo là so sánh "quan sát" (observational) — đúng cách P0 tự giới hạn phát biểu (P0 Section 5.3).

---

### M6 — Manipulation check thật
**Việc cụ thể**: viết `eval_auc_fn` nối với model đã train ở M5, gọi `run_manipulation_check(hyperedges, eval_auc_fn=..., p=0.90)`.

**Gate nghiệm thu**: `ManipulationCheckResult.passes_manipulation_check` được ghi lại kèm `verdict` — nếu `False`, phải điều tra trước khi diễn giải bất kỳ kết quả ablation nào là "graph có tác dụng" (đúng cảnh báo đã viết sẵn trong `manipulation_check.py`).

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

## 4. Việc cần làm ngay tuần này (không chờ toàn bộ M0 xong)

1. Cài `external/p0_leakage_audit/requirements.txt` + clone `pykt-toolkit` (M0, không cần dữ liệu vẫn làm được).
2. Bắt đầu tải XES3G5M trước (dataset primary) thay vì cả 3 dataset cùng lúc — mở khoá M1 sớm nhất có thể.
3. ~~Viết thêm 1 test cho `models/causal_layer.py`~~ — **Xong**: `tests/test_causal_layer.py` (4 test, dữ liệu tổng hợp có confounder đã biết, không cần GPU/dữ liệu thật).

## 5. Rủi ro kỹ thuật cụ thể (bổ sung, gắn trực tiếp với file)

| Rủi ro | File liên quan | Giảm thiểu |
|---|---|---|
| `_flatten_to_pairwise` trong `hyperedge/audit.py` là xấp xỉ (gán weight=1.0 đều cho mọi cặp) — có thể làm loãng tín hiệu `|rho|` thật | `dh2a_kt/hyperedge/audit.py` | Ghi rõ trong `notes` (đã có) khi báo cáo; nếu kết quả `|rho|` quan trọng cho luận điểm bài, cân nhắc viết thống kê hyperedge-native thay vì tái dùng nguyên xi hàm pairwise của P0 |
| `PropensityScoreATE` giả định không có confounder ẩn — vi phạm dễ xảy ra trên dữ liệu observational | `dh2a_kt/models/causal_layer.py` | Luôn báo cáo `propensity_min/max` cùng ATE; nêu rõ giả định nhận diện trong bài, không chỉ báo con số ATE |
| `DH2KT` phụ thuộc cứng vào `torch_geometric` — cài đặt trên Windows đôi khi cần đúng bản CUDA khớp | `dh2a_kt/models/dh2_kt.py` | Cài theo đúng lệnh chính thức của PyG cho bản CUDA của máy trước khi bắt đầu M3, test bằng `import torch_geometric` trước |
