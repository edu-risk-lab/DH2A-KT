# Nhật ký thực nghiệm DH2-KT — XES3G5M fold 0

Tài liệu này ghi lại **quá trình thực nghiệm**, **số liệu chính**, và **các phân tích / nhận định** trong giai đoạn từ kiến trúc v3–v4 đến giao thức sạch, v5, nhóm ưu tiên hypergraph, và ablation từng thành phần (đến thời điểm ghi).

- **Corpus:** XES3G5M, fold 0 (trừ khi ghi khác)
- **Thiết bị tham chiếu:** CUDA (RTX 3090 class)
- **Mốc SOTA sạch (test, mask repeats, chunked):** GIKT **0.823807**
- **Ngày ghi / cập nhật:** 2026-08-17

---

## 1. Tóm tắt điều hành (executive)

1. **Giao thức đánh giá** (cửa sổ + dòng lặp multi-KC) chi phối AUC mạnh hơn khoảng cách kiến trúc (~0.05 vs ~0.001–0.006).
2. **DH2-KT v4q** cạnh tranh SimpleKT trên giao thức sạch (~0.818) nhưng **không SOTA** (thua AKT/GIKT).
3. **Hypergraph trên embedding tĩnh (v2–v4)** gần như **không đóng góp** (ΔAUC ≈ 0).
4. **v5** (gộp sự kiện multi-KC + memory) đúng hướng chống rò rỉ nhưng AUC thấp hơn v4q; graph vẫn gần trơ (+0.0009).
5. **Nhóm ưu tiên** (attention + star + hedge embed + kind-conditioned): có graph **hại** (Δ −0.047); không graph lại cao hơn.
6. **Ablation:** `star` phá model (0.52); `attention` trung tính; `hyperedge_embed` giúp nhẹ (0.8047) — vẫn xa GIKT ~0.019.
7. **GIKT vượt** vì dùng đồ thị câu–KC quan sát để tinh chỉnh embedding câu + LSTM/recap — không loang đáp án qua cạnh như transport memory của v5.

---

## 2. Nhật ký theo thời gian (diary)

### Giai đoạn A — v3 residual / so với GKT

- Train lại v3 với residual `concept_embed + graph_x`.
- Kết luận nội bộ: v3 residual **không** kéo AUC vượt GKT một cách thuyết phục trên track cũ; cần đổi kiến trúc (→ v4).

### Giai đoạn B — v4 / v4q (nhánh AUC)

| Hạng mục | Kết quả (fold 0, ghi nhận) |
|---|---|
| v4 KC-only | ~0.848–0.849 (giao thức leaky / matched budget cũ) |
| v4q (question Rasch) | ~0.862–0.863 (leaky) |
| Ablation no-graph v4 | ≈ graph (Δ ~0) |
| Sparse hyperedge (neigh/pair) | AUC gần như không đổi; M6 Δ nhỏ |
| Tune capacity h256 | Không vượt baseline 128 — hết headroom kiến trúc |

**Nhận định:** Sequence modeling (LSTM, alignment DKT, chunked, early-stop val AUC) là nguồn điểm; hypergraph tĩnh dư thừa với embedding khái niệm.

### Giai đoạn C — Phát hiện rò rỉ multi-KC

- ~14.4% vị trí eval là dòng lặp (cùng user, item, timestamp); ~99.9% trùng đáp án dòng trước.
- AUC trên repeat ~0.999; trên “fresh” ~0.80.
- Zero-learning (copy prev trên repeat) đã thổi AUC tổng.
- Baseline P0 (pyKT KC-level) **cùng bị** — không mask `is_repeat`.

**Hệ quả:** Bảng P0 không phải thước đo KT thuần; cần **giao thức sạch**.

### Giai đoạn D — Giao thức sạch + audit P0

**Công cụ / mã chính:**

- `dh2a_kt/train/tier1.py`: `mask_repeats`, `window_mode` first/last/chunked, seed
- `dh2a_kt/baselines/pykt_clean.py`, `scripts/24_train_baselines_clean.py`
- `scripts/23_eval_clean_protocol.py`, `scripts/25_bootstrap_protocol_ci.py`

**Lệch giao thức phát hiện thêm:**

| Vấn đề | Chi tiết |
|---|---|
| Cửa sổ | P0: tail-200; v3 tier1: first-200; v4: chunked toàn log |
| Split báo cáo | P0 metadata “valid+test” nhưng AUC thực trên **test** |
| Padding `selectmasks='0'` | pyKT `!= -1` có thể chấm pad (trơ trên xes3g5m vì log dài) |

**Tái tạo cấu hình A (leaky, last, test) — trong dung sai:**

- DKT, SimpleKT, AKT, GIKT đều reproduce sát số công bố P0.

**Bảng sạch cấu hình C (chunked + mask_repeats, test) — fold 0:**

| Model | AUC sạch |
|---|---|
| GIKT | **0.823807** |
| AKT | 0.819751 |
| SimpleKT (3 seed) | 0.817760 / 0.817858 / 0.817762 |
| DH2-KT v4q (test-only, best e40) | **0.818340** |
| DKT | 0.798366 |

**Bootstrap (learner-level):**

- v4q vs SimpleKT: hiệu dương nhỏ, CI loại trừ 0
- v4q vs AKT: hiệu âm, CI loại trừ 0 (AKT hơn)
- v4q vs GIKT: GIKT hơn ~0.0055, CI loại trừ 0

**Sụt A→C nhất quán ~−0.054 đến −0.059** trên DKT/AKT/SimpleKT/GIKT.

**Nhận định:** Phát biểu “SOTA AUC” không đứng; phát biểu mạnh là **hiệu ứng giao thức + rò rỉ multi-KC**.

### Giai đoạn E — v5 (sự kiện multi-KC + memory transport)

**Thiết kế:**

- Gộp hàng KC → một attempt (`dh2a_kt/data/events.py`)
- Siêu cạnh `question_concepts` từ tập KC câu hỏi
- Bộ nhớ mastery theo KC + transport láng giềng (clique mặc định)
- CLI: `--architecture v5`

**Chẩn đoán dữ liệu (fold 0 test):**

- ~1.28M hàng → ~1.10M sự kiện sau gộp (~14% lặp)
- ~12.6% sự kiện multi-KC; KC-set hầu như ổn định theo câu hỏi

**Kết quả v5 (CSV valid+test, sự kiện gộp; so tuyệt đối với bảng test-only cần thận trọng):**

| Tag | AUC | Ghi chú |
|---|---|---|
| v5q_clean | 0.799463 | có graph |
| v5q_nograph | 0.798600 | Δ ≈ **+0.00086** (gần trơ) |

**Nhận định:** Gộp sự kiện đúng bệnh leakage; nhánh graph vẫn chưa tạo lợi thế AUC.

### Giai đoạn F — Nhóm ưu tiên hypergraph

Bật đồng thời: `event_pool=attention`, `transport=star`, `use_hyperedge_embed`, `kind_conditioned` (`--priority-hypergraph`).

| Tag | AUC |
|---|---|
| v5q_priority | **0.758950** |
| v5q_priority_nograph | **0.805854** |
| Δ (graph − no-graph) | **−0.0469** |

Quỹ đạo priority có graph: loss nổ (epoch 2–3, 14), val đỉnh ~0.76 rồi sụp.

**Nhận định:** Star/transport qua siêu cạnh **hại**; attention+hedge+kind không cần graph vẫn học được (~0.806). Cổng “graph phải có ích” **thất bại rõ**.

### Giai đoạn G — Ablation từng thành phần (đang chạy / cập nhật)

Mốc: GIKT 0.8238; v5 0.7995; priority_nograph 0.8059.  
Cùng seed 42, batch 64, 20 epoch, neighborhood, no-session, `graph_dropout=0`, `laux=0`.

| Tag | Thành phần | AUC test | Trạng thái (2026-08-17 tối) |
|---|---|---|---|
| v5q_ab_attn | attention only | **0.799567** | xong — trung tính |
| v5q_ab_star | star only | **0.524332** | xong — phá model |
| v5q_ab_hedge | hyperedge embed | **0.804702** | xong — tốt nhất có-graph trong ablation |
| v5q_ab_kind | kind-conditioned | (chưa) | **đang chạy** |
| v5q_ab_ahk | attn+hedge+kind, clique | (chưa) | chờ |
| *_nograph | nếu AUC ≥ 0.80 | (chưa) | sẽ tự chạy (hedge đã vượt cổng) |

Log: `results/tables/v5q_ablation.log`.

---

## 3. Quá trình thực nghiệm (method log)

### 3.1 Luồng làm việc lặp lại

1. Giả thuyết (AUC / novelty graph / sạch giao thức)
2. Sửa mã + unit test
3. Smoke GPU nhỏ → full fold 0
4. Ablation / M6 / bootstrap khi cần
5. So với bảng sạch cố định (cùng window, mask, split)
6. Ghi số vào CSV dưới `results/tables/` và checkpoint `results/checkpoints/`

### 3.2 Artefact quan trọng

| Loại | Đường dẫn ví dụ |
|---|---|
| Train DH2 | `scripts/03_train_tier1.py` |
| Baseline sạch | `scripts/24_train_baselines_clean.py` |
| Eval hai giao thức | `scripts/23_eval_clean_protocol.py` |
| Bootstrap CI | `scripts/25_bootstrap_protocol_ci.py` |
| Bảng baseline | `results/tables/xes3g5m_fold0_baselines_protocol.csv` |
| v4q test-only | `results/tables/v4q_testonly_clean.csv` |
| v5 / priority CSV | `results/tables/dh2_kt_v5q_*_vs_p0.csv` |
| Log ablation | `results/tables/v5q_ablation.log` |

### 3.3 Quyết định đã chốt với người dùng (tóm tắt)

- Phạm vi baseline sạch: DKT + GKT + SimpleKT rồi mở AKT/GIKT; GKT B/C hủy vì quá chậm; giữ tái tạo A đủ tin.
- Multi-seed cấu hình C: v4q + SimpleKT (+ GIKT một seed chính).
- Định hướng sau bảng sạch: thử **v5 / novelty hypergraph** (không chuyển hẳn bài sang chỉ “protocol paper” ngay).
- Ưu tiên GPU: ablation thành phần sau khi priority thất bại cổng graph.

---

## 4. Phân tích & nhận định (tổng hợp)

### 4.1 Đặc trưng dataset giáo dục (XES3G5M)

| Đặc trưng | Hệ quả |
|---|---|
| Export KC-level: multi-KC → nhiều dòng cùng đáp án | Rò rỉ nhãn; clean/gộp sự kiện bắt buộc |
| Multi-KC thiểu số (~12–13% attempt) | Bias siêu cạnh câu hỏi chỉ phủ phần mỏng |
| Log rất dài (>200) | first/last/chunked đổi AUC rõ (~0.009) |
| E_pre suy diễn, chain hyperedge cực dày | Tín hiệu loãng / hub; dễ trùng embedding |
| Ít teacher/forum/hint đồng bộ | Heterogeneous hypergraph gần như không có data |

### 4.2 Ưu / nhược mô hình

**v4q — ưu:** LSTM + Rasch + chunked; cạnh SimpleKT trên sạch; ổn định seed.  
**v4q — nhược:** không SOTA; graph trơ.

**v5 — ưu:** gộp multi-KC đúng bệnh; siêu cạnh câu hỏi quan sát; tách “graph mang lịch sử” khỏi embedding tĩnh (hướng novelty).  
**v5 — nhược:** AUC thấp hơn v4q; graph gần trơ; chậm (memory theo bước).

**Priority — ưu:** đụng đúng chỗ hypergraph “đúng nghĩa” hơn mean+clique; no-graph ~0.806 gợi ý attention/embed có ích cho chuỗi.  
**Priority — nhược:** star/graph **hại nặng**; tối ưu không ổn định khi bật cả gói.

### 4.3 Có nguy cơ code sai / ngược suy luận?

**Khó giải thích toàn bộ bằng “AUC đảo”:** pipeline tái tạo baseline; attention ≈ baseline; hedge **tăng** đúng chiều; star hại có điều kiện.

**Rủi ro thiết kế/mã đáng kể hơn:**

1. Transport ghi cùng evidence đáp án sang láng giềng → **khuếch tán nhãn** (khớp AUC star 0.52).
2. Star write có thể ghi trùng / không loại writer → khuếch đại nhiễu.
3. So v5 CSV (valid+test gộp) vs baseline test-only — lệch tuyệt đối nhẹ.
4. Bật 4 thay đổi + lr cũ → collapse tối ưu (một phần “thấp” là train, không phải lý thuyết).

### 4.4 Vì sao GIKT vượt trội?

Triển khai P0 (`external/p0_leakage_audit/src/models/gikt.py`):

1. Đồ thị **hai phần câu–KC quan sát** (`A_qs`), không phải chuỗi tiên quyết suy diễn.
2. GCN chỉ tinh chỉnh **embedding câu** (`q + Agg(skills)`), **không** đẩy đáp án học sinh sang KC khác.
3. LSTM trên `(q_gcn ⊕ response)` — động cơ vẫn là chuỗi.
4. **Recap attention** theo câu hỏi kế tiếp trên lịch sử hidden — khớp log dài.
5. Trên sạch vẫn dẫn sau khi mất ~0.054 như mọi model → lợi thế kiến trúc thật.

**Đối chiếu một câu:** GIKT = graph như **bộ mã hóa quan hệ câu–KC**; DH2 v5 priority = graph như **ống dẫn mastery động** — trên XES3G5M cái trước khớp, cái sau dễ hại.

### 4.5 Hệ quả hướng SOTA “theo hypergraph”

- Không kỳ vọng thắng bằng transport mạnh hơn (star đã bác bỏ).
- Hướng gần GIKT: hypergraph/bipartite **refine biểu diễn câu/KC**; prediction = LSTM/attention + recap; ΔAUC khi bỏ incidence câu–KC phải **dương và đủ lớn**.
- Hedge embed (0.8047) là manh mối: *câu-như-object giúp*, chưa đủ cho SOTA.

---

## 5. Bảng số liệu tổng hợp nhanh

### Giao thức sạch (test) — baselines + v4q

| Model | AUC |
|---|---|
| GIKT | 0.823807 |
| AKT | 0.819751 |
| v4q (e40, test-only clean) | 0.818340 |
| SimpleKT (mean 3 seed ~) | ~0.8178 |
| DKT | 0.798366 |

### Họ v5 (CSV `dh2_kt_*_vs_p0`, n≈1.635M valid+test events)

| Tag | AUC |
|---|---|
| v5q_priority_nograph | 0.805854 |
| v5q_ab_hedge | 0.804702 |
| v5q_ab_attn | 0.799567 |
| v5q_clean | 0.799463 |
| v5q_nograph | 0.798600 |
| v5q_priority | 0.758950 |
| v5q_ab_star | 0.524332 |

---

## 6. Việc còn mở (checklist)

- [ ] Xong `v5q_ab_kind`, `v5q_ab_ahk`
- [ ] Nograph twin cho hedge (và mọi run ≥ 0.80) — đo ΔAUC graph
- [ ] (Nếu theo SOTA hypergraph) thiết kế lại gần GIKT: GCN câu←KC + recap; **cấm** loang đáp án qua cạnh
- [ ] Chấm lại v5/hedge **test-only** cho bảng công bằng tuyệt đối với GIKT
- [ ] Commit/push nhật ký + số liệu ablation khi hàng đợi xong (theo yêu cầu người dùng)

---

## 7. Ghi chú phiên bản tài liệu

| Phiên bản | Nội dung |
|---|---|
| 2026-08-17 | Khởi tạo: toàn bộ quá trình, phân tích, nhật ký; ablation đến hedge xong, kind đang chạy |

*File này là nhật ký nghiên cứu nội bộ, không thay thế `paper/main.tex`.*
