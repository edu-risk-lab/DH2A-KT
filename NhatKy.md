# Nhật ký thực nghiệm DH2-KT — XES3G5M fold 0

Tài liệu này ghi lại **quá trình thực nghiệm**, **số liệu chính**, và **các phân tích / nhận định** trong giai đoạn từ kiến trúc v3–v4 đến giao thức sạch, v5, nhóm ưu tiên hypergraph, và ablation từng thành phần (đến thời điểm ghi).

- **Corpus:** XES3G5M, fold 0 (trừ khi ghi khác)
- **Thiết bị tham chiếu:** CUDA (RTX 3090 class)
- **Mốc SOTA sạch (test, mask repeats, chunked):** GIKT **0.823807**
- **Ngày ghi / cập nhật:** 2026-08-18

---

## 1. Tóm tắt điều hành (executive)

1. **Giao thức đánh giá** (cửa sổ + dòng lặp multi-KC) chi phối AUC mạnh hơn khoảng cách kiến trúc (~0.05 vs ~0.001–0.006).
2. **DH2-KT v4q** cạnh tranh SimpleKT trên giao thức sạch (~0.818) nhưng **không SOTA** (thua AKT/GIKT).
3. **Hypergraph trên embedding tĩnh / mean-add (v2–v4 E3, prereq, transport)** gần như **không đóng góp**; **Q←KC refine** (`--question-graph`, giai đoạn I) thì **PASS** cổng Δval +0.006.
4. **v5** (gộp sự kiện multi-KC + memory) đúng hướng chống rò rỉ nhưng AUC thấp hơn v4q; graph vẫn gần trơ (+0.0009).
5. **Nhóm ưu tiên** (attention + star + hedge embed + kind-conditioned): có graph **hại** (Δ −0.047); không graph lại cao hơn.
6. **Ablation:** `star` phá model (0.52); `attention` trung tính; `hyperedge_embed` giúp nhẹ (0.8047) — vẫn xa GIKT ~0.019.
7. **GIKT vượt** vì dùng đồ thị câu–KC quan sát để tinh chỉnh embedding câu + LSTM/recap — không loang đáp án qua cạnh như transport memory của v5.
8. **Giai đoạn H (SOTA ladder E0–E5):** không vượt GIKT ở L=200; E2/E3 FAIL; cải thiện từ L=400 + batch + ensemble.
9. **Giai đoạn I (Q←KC refine, 2026-08-18):** `--question-graph` (GIKT-style, không transport) **PASS cổng** Δval ≥ +0.002 (Δval **+0.00612**); test-only clean **0.826984** vs twin off **0.820249**. Chưa tuyên bố SOTA tuyệt đối vs GIKT L=200 (khác cửa sổ L=400).

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
| v5q_ab_kind | kind-conditioned | **0.799561** | xong — trung tính |
| v5q_ab_ahk | attn+hedge+kind, clique | (chưa / đang chạy) | xem log ablation |
| *_nograph | nếu AUC ≥ 0.80 | (chưa) | hedge vượt cổng → sẽ có twin |

Log: `results/tables/v5q_ablation.log`.

### Giai đoạn H — Hướng SOTA v4q (`PROMPT_cursor_v4q_sota.md`)

Thực nghiệm theo thứ tự E0→E5; cổng quyết định đặt trước khi nhìn số.

#### E0 — Đối chứng "bất biến giao thức" (xong)

Cùng checkpoint train với `mask_repeats=True`, chấm lại trên chunked **leaky**:

| Model | clean AUC | leaky AUC (cùng ckpt) | n_scored leaky |
|---|---|---|---|
| SimpleKT (`C_simplekt_clean`) | 0.817760 | **0.814198** | 1.273.545 |
| AKT (`C_akt_clean`) | 0.819751 | **0.809396** | 1.273.545 |

**Cổng:** nếu leaky ≈ 0.818 → lợi thế không tồn tại; nếu ~0.87 → thật.

**Kết luận:** cả hai đều ≈ 0.81 (thậm chí *thấp hơn* clean), **không** nhảy lên ~0.87 như B_* (retrain leaky). Lợi thế "bất biến giao thức" của v4q so với baseline **không tồn tại** khi so đối xứng (cùng ckpt, đổi eval). Xóa mọi phát biểu kiểu đó khỏi nháp bài báo. File: `results/tables/sota_E0_protocol_invariance_control.csv`.

#### E1–E5

- E1: quét L∈{100,150,200,300,400} (batch=64, epochs=20). **Chốt L\*=400** theo val AUC (0.818629); test AUC 0.819285. File: `results/tables/sota_E1_chunk_sweep.csv`.
- E2: `--recap-attention` tại L=400 → val **0.8175** vs baseline 0.8186 (Δ **−0.0011**). **GATE FAIL** (cần ≥ +0.002) → **tắt cờ**, ghi nhật ký thất bại.
- E3: `--question-kc-agg` tại L=400 → val **0.8183** / test **0.8192** vs baseline 0.8186 / 0.8193 (Δ ≈ **0**). **GATE FAIL** → ghi thẳng: graph Q–KC agg **không đóng góp AUC**.
- E4: batch 16/32, L=400, epochs=30. **bs16** val 0.8201 / test **0.8209** (44 phút); **bs32** val 0.8198 / test 0.8207 (23 phút). bs16 tốt nhất trong ladder nội bộ.
- E5: ensemble 3 seed v4q_clean (logit avg, test-only) → **0.820721** (đơn seed ~0.818). Dòng riêng “ensemble of 3 seeds”.

#### Đánh giá tổng thể giai đoạn H (2026-08-18)

- **Ladder E0→E5 đã chạy xong.** E6 (gộp multi-KC không transport) không bắt buộc vì E2+E3 đã fail cổng.
- **Không đạt SOTA sạch** so với GIKT **0.823807**. Số gần nhất trên protocol sạch test-only: ensemble **0.820721**; E4 bs16 đạt **0.820908** trên CSV valid+test (không đồng protocol tuyệt đối với bảng GIKT).
- **Hai đòn kiến trúc gần GIKT đều FAIL:** recap Δval −0.001; Q–KC agg Δ≈0 → **mean-add graph không đóng góp AUC** theo cổng đã đặt trước.
- **Cải thiện có thật** đến từ cửa sổ L=400, batch nhỏ hơn, và ensemble — không từ E2/E3.
- **Hệ quả bài báo (sau H):** không viết tiêu đề từ E2/E3. Trục vững: rò rỉ multi-KC, bảng sạch, audit protocol. File: `results/tables/sota_summary.csv`.

### Giai đoạn I — Novelty graph vòng cuối: Q←KC refine (2026-08-18)

**Thiết kế (đã chốt):** embedding câu riêng + `q_gcn = tanh(W(q + A_norm @ concept))` nạp vào LSTM/query v4q; **không** transport đáp án, không star, không prereq-as-novelty. Cờ `--question-graph`. Twin ablation: cùng protocol, `--no-graph` concept stack, **không** `--question-graph`.

**Protocol:** fold 0, mask-repeats, chunked, L=400, batch=16, epochs=30, val_frac=0.1, seed=42, `--use-questions`, `--no-session --no-graph`.

**Cổng (val, đăng ký trước):** `val(on) − val(off) ≥ +0.002`.

| Run | tag | val AUC | test AUC (CSV valid+test) | test-only clean |
|---|---|---|---|---|
| B | `hg_qkc_off` | 0.819862 | 0.820497 | **0.820249** |
| A | `hg_qkc_on` | **0.825982** | 0.827184 | **0.826984** |
| Δ A−B | | **+0.006120** | +0.006687 | +0.006736 |

**Kết luận cổng:** **PASS** (Δval ≫ +0.002). Graph Q←KC kiểu GIKT **có đóng góp AUC** trên ablation cứng — khác E3 (mean-add absorbable). Bootstrap learner-level (1000×): Δtest-only **+0.00674**, 95% CI **[+0.00626, +0.00721]** (loại trừ 0). File: `bootstrap_hg_qkc_on_vs_off.json`.

**So GIKT 0.823807 (clean test, L≈200):** số thô `hg_qkc_on` test-only **0.826984** cao hơn, nhưng **không** tuyên bố SOTA tuyệt đối cho đến khi (i) GIKT (hoặc baseline) chạy cùng L=400 và (ii) bootstrap CI của hiệu số vs GIKT loại trừ 0. File: `results/tables/dh2_kt_hg_qkc_*_vs_p0.csv`, `hg_qkc_testonly_clean.csv`, `hg_qkc_ablation.log`.

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
| Ensemble E5 | `scripts/26_sota_E5_ensemble.py` |
| Bảng baseline | `results/tables/xes3g5m_fold0_baselines_protocol.csv` |
| v4q test-only | `results/tables/v4q_testonly_clean.csv` |
| v5 / priority CSV | `results/tables/dh2_kt_v5q_*_vs_p0.csv` |
| SOTA ladder | `results/tables/sota_*.csv`, `dh2_kt_sota_*_vs_p0.csv` |
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

### E0 protocol-invariance control (cùng ckpt clean → eval leaky)

| Model | clean | leaky |
|---|---|---|
| SimpleKT | 0.817760 | 0.814198 |
| AKT | 0.819751 | 0.809396 |

→ **GATE FAIL** cho tuyên bố bất biến giao thức.

### E1 chunk length (val+test CSV; chốt theo val)

| L | val_auc | test_auc | phút |
|---|---|---|---|
| 100 | 0.8147 | 0.8154 | 26.1 |
| 150 | 0.8166 | 0.8172 | 18.5 |
| 200 | 0.8172 | 0.8181 | 15.5 |
| 300 | 0.8183 | 0.8191 | 11.5 |
| **400** | **0.8186** | **0.8193** | 9.2 |

→ **L\* = 400** cho mọi thí nghiệm sau.

### E2 recap attention (L=400)

| | val_auc | test_auc |
|---|---|---|
| baseline E1 L=400 | 0.818629 | 0.819285 |
| +recap | 0.817501 | 0.818182 |
| Δ | **−0.00113** | −0.00110 |

→ **GATE FAIL** (cần Δval ≥ +0.002). Tắt `--recap-attention`.

### E3 question–KC agg (L=400, incidence quan sát)

| | val_auc | test_auc |
|---|---|---|
| baseline E1 L=400 | 0.818629 | 0.819285 |
| +question_kc_agg | 0.818296 | 0.819185 |
| Δ | **−0.00033** | −0.00010 |

→ **GATE FAIL** (cần ΔAUC ≥ +0.002). **Graph không đóng góp AUC** trên cổng này — báo cáo trung thực, không biện minh.

### E4 matched budget (L=400, epochs=30)

| batch | val_auc | test_auc (valid+test CSV) | phút |
|---|---|---|---|
| 64 (E1) | 0.8186 | 0.8193 | ~9 |
| **16** | **0.8201** | **0.8209** | 43.9 |
| 32 | 0.8198 | 0.8207 | 22.8 |

→ Giảm batch giúp nhẹ; vẫn dưới GIKT 0.8238 (bảng sạch test-only).

### E5 ensemble 3 seed (logit avg; test-only chunked + mask)

| Model | AUC | n_scored |
|---|---|---|
| v4q_clean s42 | 0.818243 | 1.090.610 |
| v4q_clean s17 | 0.818109 | 1.090.610 |
| v4q_clean s1234 | 0.817975 | 1.090.610 |
| **ensemble_of_3_seeds** | **0.820721** | 1.090.610 |

→ Báo cáo thành **một dòng riêng** “ensemble of 3 seeds”; vẫn dưới GIKT 0.823807 (~−0.003).

### Giai đoạn H — bảng tóm tắt cổng

| Exp | Kết luận cổng |
|---|---|
| E0 | FAIL — không có lợi thế bất biến giao thức |
| E1 | PASS chọn L\* — **L=400** |
| E2 | FAIL — tắt recap |
| E3 | FAIL — graph ΔAUC≈0 |
| E4 | bs16 tốt nhất nội bộ (0.8209 valid+test) |
| E5 | ensemble 0.8207 test-only; chưa vượt GIKT |
| I (`question_graph`) | **PASS** Δval +0.00612; test-only 0.8270; chưa SOTA tuyệt đối vs GIKT@L200 |

---

## 6. Việc còn mở (checklist)

- [x] E0–E5 SOTA ladder (giai đoạn H)
- [x] Giai đoạn I: `--question-graph` vs twin off → **GATE PASS**
- [x] Test-only clean rescore L=400 (`hg_qkc_testonly_clean.csv`)
- [x] Bootstrap CI on−off (learner-level) → Δ CI [+0.00626, +0.00721], loại trừ 0
- [ ] (Trước tuyên bố SOTA) GIKT / AKT cùng L=400 + bootstrap vs `hg_qkc_on`
- [ ] (Tùy chọn) variant (1) event-collapse — chỉ nếu muốn sau khi I đã PASS
- [x] Commit nhật ký + số liệu hg_qkc (sau cổng)
- [x] Push `main`: `809c76b` / `e7ffbcc` (question-graph + diary)

---

## 7. Ghi chú phiên bản tài liệu

| Phiên bản | Nội dung |
|---|---|
| 2026-08-17 | Khởi tạo: toàn bộ quá trình, phân tích, nhật ký; ablation đến hedge xong, kind đang chạy |
| 2026-08-17 tối | Giai đoạn H: E0 xong (bác bỏ protocol-invariance); bắt đầu E1; thêm cờ E2/E3 |
| 2026-08-18 | E1–E5 xong; đánh giá tổng thể: không SOTA; E2/E3 FAIL; push `sota_summary.csv` |
| 2026-08-18 sáng | Giai đoạn I: Q←KC `--question-graph` GATE PASS Δval +0.006; test-only 0.8270; push `809c76b` |

*File này là nhật ký nghiên cứu nội bộ, không thay thế `paper/main.tex`.*
