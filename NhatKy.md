# Nhật ký thực nghiệm DH2-KT — XES3G5M fold 0

Tài liệu này ghi lại **quá trình thực nghiệm**, **số liệu chính**, và **các phân tích / nhận định** trong giai đoạn từ kiến trúc v3–v4 đến giao thức sạch, v5, nhóm ưu tiên hypergraph, và ablation từng thành phần (đến thời điểm ghi).

- **Corpus:** XES3G5M, fold 0 (trừ khi ghi khác)
- **Thiết bị tham chiếu:** CUDA (RTX 3090 class)
- **Mốc SOTA sạch (test, mask repeats, chunked):** GIKT **0.823807**
- **Ngày ghi / cập nhật:** 2026-08-19

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
9. **Giai đoạn I (Q←KC refine, 2026-08-18):** `--question-graph` **PASS cổng** Δval **+0.00612**; test-only **0.826984**.
10. **Giai đoạn J (fair L=400, 2026-08-18):** GIKT sạch L=400 test **0.825953**; matched budget **0.825815**. `hg_qkc_on` hơn GIKT **+0.001**, CI loại trừ 0.
11. **Giai đoạn K (HypergraphConv multi-KC cô lập):** GATE **FAIL** Δval **+0.00025** (cần ≥ +0.002). Slice `multi_kc_item` Δ **+0.0003**, CI chứa 0. Trên XES3G5M, hypergraph multi-way **không** thêm gì ngoài bipartite GIKT.
12. **Giai đoạn L (Hint-item hypergraph, FoundationalASSIST):** GATE **FAIL** Δval **+0.00007** (cần ≥ +0.002). Slice `has_hint` Δ **−0.014**, CI chứa 0. Hyperedge hành vi thật, nhánh non-absorbable, **không** dịch AUC trên protocol đã đăng ký.
13. **Giai đoạn M (thuộc tính rẻ→đắt):** M2 Δt **PASS** Δval **+0.021**; M3 saw t−1 **PASS** Δval **+0.0047**; M1 Q-matrix / M4 teacher / M5 Junyi DAG **FAIL**. Không nới cổng.
14. **Hướng P0–P4 (2026-08-19):** P0 time-gap XES **PASS** Δval **+0.00280**. P1: query giữ phần lớn M2, LSTM nhỏ. P2 time+saw **PASS** +0.00297. P3 duration/idle ASSIST **PASS** +0.00442. P4 per-concept forget **FAIL**. Không gán cho hypergraph.

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

**So GIKT L=200:** số thô cao hơn nhưng không dùng làm tuyên bố SOTA (cửa sổ khác). Fair so sánh: giai đoạn J.

### Giai đoạn J — GIKT/AKT sạch @ L=400 (2026-08-18)

Protocol: fold 0, chunked, mask-repeats, test-only, **L=400**. GIKT/AKT giữ hyperparam P0 (`scripts/24_train_baselines_clean.py --max-seq-len 400`).

| Model | tag | test AUC | n | phút |
|---|---|---|---|---|
| DH2 `hg_qkc_on` | question-graph | **0.826984** | 1.093.755 | (trước đó) |
| GIKT | `C_gikt_clean_L400` | 0.825953 | 1.093.720 | 155.2 |
| AKT | `C_akt_clean_L400` | 0.822140 | 1.093.720 | 13.4 |
| DH2 `hg_qkc_off` | twin | 0.820249 | 1.093.755 | (trước đó) |
| GIKT L=200 (cũ) | `C_gikt_clean` | 0.823807 | 1.090.574 | 142.7 |

Bootstrap learner-level 1000×, 3614 learners:

- `hg_qkc_on` − GIKT_L400 = **+0.001031**, 95% CI **[+0.000633, +0.001400]** → loại trừ 0.
- `hg_qkc_on` − AKT_L400 = **+0.004844**, 95% CI **[+0.004388, +0.005295]** → loại trừ 0.

**Kết luận:** trên giao thức sạch L=400, `--question-graph` **vượt GIKT** một khe nhỏ nhưng có ý nghĩa thống kê. GIKT L=400 chỉ +0.0021 so với GIKT L=200. **Lưu ý ngân sách:** GIKT 10 epoch / batch 8; DH2 30 epoch / batch 16 — không gán phần vượt cho “hypergraph” nếu đối thủ chưa matched budget. File: `sota_baselines_L400.csv`, `bootstrap_hg_qkc_on_vs_gikt_L400.json`.

**Matched budget (2026-08-18 chiều):** `C_gikt_clean_L400_e30b16` — batch 16, cap 30, patience 5 (như DH2). Early-stop epoch 8/13; test **0.825815** (100 phút) — **không** hơn bản P0 10ep/bs8 (0.825953). Bootstrap: `hg_qkc_on` − GIKT_e30b16 = **+0.001169**, 95% CI **[+0.000776, +0.001528]** (loại trừ 0). Khe không phải do GIKT thiếu epoch/batch. File: `sota_gikt_L400_e30b16.log`, `bootstrap_hg_qkc_on_vs_gikt_e30b16.json`.

### Giai đoạn K — HypergraphConv multi-KC cô lập (2026-08-18)

Backbone = `hg_qkc_on`. Thêm `--question-hypergraph`: 1 lớp PyG HypergraphConv trên **1017** siêu cạnh `question_concepts` (train, min_size=2); Linear `tanh(W(·))` mới; không E_pre, không v5 transport.

Cổng val (đăng ký trước): Δval ≥ +0.002 vs `hg_qkc_on` 0.825982.

| Run | val AUC | test CSV (valid+test) | test-only (slice) |
|---|---|---|---|
| `hg_qkc_on` | 0.825982 | 0.827184 | 0.8270 |
| `hg_qkc_hconv_on` | 0.826232 | 0.827443 | 0.8273 |
| Δ | **+0.000250** | +0.000259 | +0.0004 |

**GATE FAIL.** Slice test-only L=400 mask-repeats (`hg_qkc_hconv_slice.csv`):

| Slice | n | AUC hconv | AUC qgraph | Δ | CI 95% |
|---|---|---|---|---|---|
| overall | 1.093.755 | 0.8273 | 0.8270 | +0.0004 | [−0.0008, +0.0012] chứa 0 |
| single_kc_item | 956.199 | 0.8264 | 0.8260 | +0.0004 | chứa 0 |
| **multi_kc_item** | 137.556 | 0.8338 | 0.8336 | **+0.0003** | [−0.0007, +0.0015] chứa 0 |

**Kết luận:** ngay trên đúng phân nhóm multi-KC, HypergraphConv **không** đóng góp. Bipartite GIKT đã lấy hết tín hiệu Q–KC trên XES3G5M. Đóng cánh cửa “hypergraph multi-way AUC” (không nới cổng).

### Giai đoạn L — Hint-item hypergraph trên FoundationalASSIST (2026-08-18)

**Giả thuyết:** hyperedge hành vi thật (session có Hint) dịch AUC khi **không** chiếu về concept — khác mọi cửa Q–KC đã cạn trên XES3G5M.

**Thiết kế:** backbone v4q + `--question-graph` + `--no-graph` (không E_pre, không session→concept). Thêm `--hint-hypergraph`: HypergraphConv 1 lớp trên **item** của session train-only có `hint_used`; Linear `tanh(W(·))` + `hint_in_proj` riêng (`hint_item_embed`, không hấp thụ vào `concept_embed`).

**Corpus:** FoundationalASSIST fold 0 (`foundational_assist_extended.parquet`, hint_rate ≈ 5.2%). **Không** kỳ vọng số AUC so được với `hg_qkc_on` trên XES3G5M.

**Protocol:** mask-repeats, chunked, L=200, batch=16, cap 30 epoch, patience 5, val_frac=0.1, seed=42, `--use-questions`, `--no-session --no-graph`.

**Cổng (val, đăng ký trước):** `val(hint_on) − val(fa_off) ≥ +0.002`. Slice bắt buộc: `has_hint` vs `no_hint` trên target row. Không nới cổng.

Siêu cạnh: **30 958** session-hint item HE (train), chọn **20 000** (incidence 149 431). Không E_pre, không chiếu concept.

| Run | val AUC | test CSV (valid+test) | test-only (slice) |
|---|---|---|---|
| `hg_qkc_fa_off` | 0.808609 | 0.804109 | 0.800683 |
| `hg_qkc_fa_hint_on` | 0.808677 | 0.804124 | 0.800704 |
| Δ | **+0.000067** | +0.000015 | +0.000021 |

**GATE FAIL.** Slice test-only L=200 mask-repeats (`hg_qkc_fa_hint_slice.csv`):

| Slice | n | AUC hint | AUC off | Δ | CI 95% |
|---|---|---|---|---|---|
| overall | 340 356 | 0.8007 | 0.8007 | +0.0000 | [−0.0007, +0.0011] chứa 0 |
| **has_hint** | 17 678 | 0.6980 | 0.7120 | **−0.0140** | [−0.048, +0.013] chứa 0 |
| no_hint | 322 678 | 0.7903 | 0.7904 | −0.0001 | chứa 0 |

`has_hint` có positive rate ≈ 0.2% (gần như mọi bước có hint đều sai) nên AUC slice này ồn; CI rộng và chứa 0. Single-skill FA: `multi_kc_item` n=0.

**Kết luận:** trên đúng corpus có Hint thật và đúng khuôn non-absorbable, nhánh item-hypergraph **không** đóng góp AUC. Δval cũ +0.00236 (session HE chiếu về concept, kiến trúc khác) **không** tái lập khi tách Hint khỏi Q–KC. Đóng cánh cửa “hint hyperedge AUC” trên protocol này (không nới cổng). Không so số với XES3G5M.

### Giai đoạn M — thuộc tính còn lại, rẻ thử trước (2026-08-18)

Nguồn *thông tin* còn lại (không GNN Q–KC mới, không same-step label proxy). Cổng **đăng ký trước**, không nới: Δval ≥ +0.002 vs twin khớp vocab; nhánh Linear/embed riêng (không hấp thụ vào `concept_embed`). Backbone FA = v4q + `--question-graph` + `--no-graph --no-session`, L=200, bs16, cap 30, patience 5, val_frac=0.1, seed 42, mask-repeats, `--graph-dropout 0`.

| # | Tín hiệu | Twin val | ON val | Δval | Cổng |
|---|---|---|---|---|---|
| M1 | FA full Q-matrix `Skills.csv` → `A_qs` (chuỗi vẫn 1 KC; vocab 224) | 0.808340 | 0.808239 | **−0.00010** | **FAIL** |
| M2 | log1p Δt (timestamp `end_time`) | 0.808609 (`fa_off`) | 0.829722 | **+0.02111** | **PASS** |
| M3 | `saw_answer` bước *trước* (LSTM only) | 0.808609 | 0.813295 | **+0.00469** | **PASS** |
| M4 | ASSIST2012 `teacher_id` (726 nhóm, join 100% theo user) | 0.764349 | 0.765765 | **+0.00142** | **FAIL** |
| M5 | Junyi expert DAG (1131 cặp, 5000 users, 695 KC) | 0.7640 | 0.764557 | **+0.00056** | **FAIL** |

M1: mean degree `A_qs` 1.00 → 1.13. Đa-skill metadata **không** thêm AUC trên FA.

M2: Linear `log1p(Δt)` vào LSTM **và** query bước t+1 (khoảng cách đến item sắp chấm — biết trước khi dự đoán). Timestamp FA là `end_time` nên Δt gồm idle + thời gian làm bài trước. **Không** tuyên bố hypergraph.

M3: `saw[t]` khi dự đoán `t+1`; unit test chặn đưa `saw[t+1]` vào query. train saw-rate ≈ 21.8%. Khác same-step proxy (P(correct|saw)≈0).

M4: join `(user,item,timestamp)` ban đầu 0% (P0 lưu ts = unix//1000). Sửa: teacher theo `user_id` hashed. 726 giáo viên, khớp mọi dòng train. Δval +0.0014 **không** đủ cổng. Không so 0.76 với P0 GKT 0.96 (giao thức khác).

M5: `--expert-graph` 1131 cặp trên 695 concept; subsample 5000 user (cỡ FA), không phải full 26M. Δval ≈ 0.

**Kết luận:** tín hiệu hành vi rẻ (thời gian, saw bước trước) **có** AUC; Q-matrix đầy đủ / teacher grouping / expert DAG **không** qua cổng +0.002. Không nới cổng. Không gán PASS cho hypergraph.

### Hướng P0–P4 — cổng đăng ký trước khi train (2026-08-19)

Không nhầm **P0 (wave này)** với baseline pyKT/P0. Cổng **không nới**: Δval ≥ +0.002 vs twin khớp; Linear/embed/θ_c riêng (không hấp thụ vào `concept_embed`). Không GNN Q–KC mới, không hint-hypergraph, không gán time/saw cho hypergraph.

| Pri | Tag | Việc | Twin val (đăng ký) | Ghi chú |
|---|---|---|---|---|
| P0 | `p0_dt_on` | `--time-gap --time-gap-mode both` trên XES `hg_qkc_on` **L=400** | **0.825982** | Protocol I: v4q + `--question-graph --mask-repeats --window-mode chunked --max-seq-len 400 --batch-size 16 --epochs 30 --val-frac 0.1 --early-stop-patience 5 --seed 42 --graph-dropout 0 --graph-sensitivity-weight 0 --no-session --no-graph`. Slice tứ phân vị Δt. Đây là bước duy nhất có thể đổi câu SOTA corpus chính. |
| P1 | `p1_dt_lstm` / `p1_dt_query` | Ablation M2: LSTM-only vs query-only | M2 both **0.829722**; cũng báo vs `fa_off` **0.808609** | FA, cùng protocol M. Quyết định claim forgetting vs ngữ cảnh attempt. |
| P2 | `p2_dt_saw` | `--time-gap --saw-input` trên FA | Twin **M2** 0.829722 (cộng tính?); phụ vs `fa_off` | Bootstrap learner-level nếu additive. |
| P3 | `p3_split_on` | ASSIST duration (`ms_first_response`) + idle; query **chỉ** idle t+1 | `m4_group_off` **0.764349** | P0 ts = unix//1000. Không đưa duration t+1 vào query. |
| P4 | `p4_forget_on` | `exp(-softplus(θ_c)·gap)` thay Linear(1,H) | Twin = Linear time-gap hiện tại | **Chỉ sau P0/P1.** Nếu P0 PASS → XES; nếu FAIL → FA vs M2. |

`--time-gap-mode {both,lstm,query}`; `--time-split`; `--concept-forget`. `make_sequence_loader` phải forward time-gap từ checkpoint (trước đây im lặng bỏ gap khi rescore).

**Kết quả:**

| Pri | Twin | ON val | Δval | Cổng |
|---|---|---|---|---|
| P0 `p0_dt_on` | `hg_qkc_on` 0.825982 | **0.828779** | **+0.00280** | **PASS** |
| P1 `p1_dt_lstm` | `fa_off` 0.808609 / M2 0.829722 | 0.811405 | **+0.00280** / −0.01832 | PASS vs off; kém both |
| P1 `p1_dt_query` | `fa_off` / M2 | 0.823318 | **+0.01471** / −0.00640 | PASS vs off; gần most of M2 |
| P2 `p2_dt_saw` | M2 0.829722 | **0.832696** | **+0.00297** | **PASS** (cộng tính) |
| P3 `p3_split_on` | `m4_group_off` 0.764349 | **0.768771** | **+0.00442** | **PASS** |
| P4 `p4_forget_on` | Linear P0 0.828779 | 0.826195 | **−0.00258** | **FAIL** (cũng +0.00021 vs `hg_qkc_on`) |

P0: early-stop epoch 18/23. valid+test **0.829754** vs twin 0.827184 (Δ **+0.00257**). Slice `p0_dt_slice.csv`: overall Δ CI (bootstrap 100k cap) loại trừ 0. qcut Δt chỉ còn **2** bin (trùng giá trị): gap ngắn log1p≤6.98 (≈18 phút, n=1.23M) Δ **+0.0028**; gap dài Δ **+0.0019**. Cả hai CI loại trừ 0. Time-gap trên XES **không** phải forgetting dài hạn là chính — khớp P1 (query t+1 chiếm phần lớn M2 trên FA). **Không** gán cho hypergraph (`--no-graph`).

P1: M2 both vẫn tốt nhất. Query (khoảng đến bước chấm) giữ ~70% Δ vs off; LSTM-only chỉ +0.0028. Claim paper: tín hiệu thời gian chủ yếu là **ngữ cảnh attempt**, không phải decay mastery ẩn.

P2: `--time-gap --saw-input` cộng trên M2 đủ cổng +0.002. vs `fa_off` Δval +0.02409.

P3: join duration 1 860 290 / 1 894 651 train (98.2%), median 25.7s. Query chỉ idle t+1. Twin ASSIST v4q off, không so 0.77 với P0 GKT 0.96.

P4: `exp(-softplus(θ_c)·gap)` thay Linear(1,H) trên XES **không** hơn Linear; dưới cổng vs cả Linear lẫn `hg_qkc_on`. Giữ `--time-gap` Linear.

**Kết luận hướng P:** AUC còn lại trên corpus chính là **time-aware next-step** (Linear log1p Δt) chồng Q←KC LSTM, cộng saw trên FA, duration/idle trên ASSIST. Per-concept forget không thắng Linear. Hypergraph vẫn đóng.

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
| I (`question_graph`) | **PASS** Δval +0.00612; test-only 0.8270 |
| J (GIKT/AKT @ L=400) | GIKT 0.82595; AKT 0.82214; `hg_qkc_on` +0.00103 vs GIKT, CI loại trừ 0 |
| K (`question_hypergraph`) | **FAIL** Δval +0.00025; multi-KC slice Δ +0.0003, CI chứa 0 |
| L (`hint_hypergraph` @ FA) | **FAIL** Δval +0.00007; has_hint Δ −0.014, CI chứa 0 |
| M1–M5 (thuộc tính rẻ→đắt) | M2 PASS +0.021; M3 PASS +0.0047; M1/M4/M5 FAIL |
| P0 (`time_gap` @ XES L=400) | **PASS** Δval +0.00280; valid+test +0.00257, slice CI loại trừ 0 |
| P1 (lstm / query @ FA) | query +0.01471 vs off; lstm +0.00280; both (M2) vẫn tốt nhất |
| P2 (time+saw @ FA) | **PASS** Δval +0.00297 vs M2 |
| P3 (duration+idle @ ASSIST) | **PASS** Δval +0.00442 vs `m4_group_off` |
| P4 (`concept_forget` @ XES) | **FAIL** −0.00258 vs Linear P0 |

---

## 6. Việc còn mở (checklist)

- [x] E0–E5 SOTA ladder (giai đoạn H)
- [x] Giai đoạn I: `--question-graph` vs twin off → **GATE PASS**
- [x] Test-only clean rescore L=400 (`hg_qkc_testonly_clean.csv`)
- [x] Bootstrap CI on−off (learner-level) → Δ CI [+0.00626, +0.00721], loại trừ 0
- [x] GIKT / AKT sạch @ L=400 + bootstrap vs `hg_qkc_on` → `hg_qkc_on` hơn GIKT +0.001, CI loại trừ 0
- [x] GIKT matched budget L=400 bs16 cap30 patience5 → test **0.825815** (không hơn bản 10ep/bs8); `hg_qkc_on` +0.00117, CI [+0.00078, +0.00153]
- [x] Giai đoạn K: HypergraphConv cô lập trên `question_concepts` → **GATE FAIL**; slice multi-KC Δ≈0
- [x] Giai đoạn L: Hint-item hypergraph trên FoundationalASSIST → **GATE FAIL**; slice has_hint Δ CI chứa 0
- [x] Giai đoạn M: M1 full Q-matrix **FAIL**; M2 Δt **PASS** +0.021; M3 saw t−1 **PASS** +0.0047; M4 teacher **FAIL**; M5 Junyi DAG **FAIL**
- [x] Hướng P0: `--time-gap` trên XES `hg_qkc_on` L=400 vs val 0.825982 → **PASS** Δval **+0.00280**
- [x] Hướng P1: FA `--time-gap-mode lstm` / `query` vs M2 both → query chiếm phần lớn; lstm nhỏ
- [x] Hướng P2: FA `--time-gap --saw-input` vs M2 → **PASS** Δval **+0.00297**
- [x] Hướng P3: ASSIST `--time-split` vs `m4_group_off` → **PASS** Δval **+0.00442**
- [x] Hướng P4: `--concept-forget` trên XES vs Linear P0 → **FAIL** Δval **−0.00258**
- [ ] (Tùy chọn) variant (1) event-collapse — không ưu tiên (K đã đóng hypergraph multi-way AUC)
- [x] Commit nhật ký + số liệu hg_qkc (sau cổng)
- [x] Push `main`: `809c76b` / `e7ffbcc` (question-graph + diary)
- [x] Push giai đoạn J: GIKT/AKT L=400 + bootstrap vs `hg_qkc_on`
- [x] Push GIKT matched budget e30b16 + bootstrap
- [x] Push giai đoạn K: HypergraphConv multi-KC GATE FAIL + slice

---

## 7. Ghi chú phiên bản tài liệu

| Phiên bản | Nội dung |
|---|---|
| 2026-08-17 | Khởi tạo: toàn bộ quá trình, phân tích, nhật ký; ablation đến hedge xong, kind đang chạy |
| 2026-08-17 tối | Giai đoạn H: E0 xong (bác bỏ protocol-invariance); bắt đầu E1; thêm cờ E2/E3 |
| 2026-08-18 | E1–E5 xong; đánh giá tổng thể: không SOTA; E2/E3 FAIL; push `sota_summary.csv` |
| 2026-08-18 sáng | Giai đoạn I: Q←KC `--question-graph` GATE PASS Δval +0.006; test-only 0.8270; push `809c76b` |
| 2026-08-18 trưa | Giai đoạn J: GIKT L=400 test 0.8260; `hg_qkc_on` +0.001 vs GIKT (CI loại trừ 0) |
| 2026-08-18 tối | Giai đoạn K: HypergraphConv multi-KC GATE FAIL; slice Δ≈0 |
| 2026-08-18 đêm | Giai đoạn L: Hint-item FA GATE FAIL Δval +0.00007; slice has_hint CI chứa 0 |
| 2026-08-18 đêm | Giai đoạn M: M2/M3 PASS (Δt, saw t−1); M1/M4/M5 FAIL |
| 2026-08-19 | Hướng P0–P4: P0/P2/P3 PASS; P1 query chiếm M2; P4 forget FAIL. Không gán time cho hypergraph. |

*File này là nhật ký nghiên cứu nội bộ, không thay thế `paper/main.tex`.*
