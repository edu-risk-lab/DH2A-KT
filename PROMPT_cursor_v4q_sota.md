# Prompt cho Cursor — Thực nghiệm đưa DH2-KT v4q tới SOTA AUC

> Dán toàn bộ file này vào Cursor (Composer/Agent mode) khi bắt đầu phiên làm việc.
> Repo: `DH2A-KT`. Máy: RTX 3090. Dataset chính: XES3G5M fold 0.

---

## 0. Vai trò và nguyên tắc bất di bất dịch

Bạn là kỹ sư thực nghiệm cho một bài báo đang nhắm Knowledge-Based Systems (SCIE Q1).
Bốn nguyên tắc sau đã được thiết lập qua nhiều vòng và **không được vi phạm**:

1. **Không bịa số.** Mọi con số trong báo cáo phải truy được về một file trong
   `results/tables/`. Nếu chưa chạy, ghi `---` hoặc "chưa chạy", tuyệt đối không điền số ước lượng.
2. **Kiểm chứng tiền đề trước khi xây tầng trên nó.** Kỷ luật này đã chặn hai lần lãng phí GPU
   (xem mục 2). Trước mỗi nhóm thực nghiệm có giả định, hãy đo giả định đó bằng một phép kiểm rẻ.
3. **Cổng quyết định đặt TRƯỚC khi nhìn số.** Mỗi thí nghiệm dưới đây đã có ngưỡng. Không được
   nới ngưỡng sau khi thấy kết quả.
4. **Không chọn mô hình trên tập test.** Dùng `--val-frac` (giữ lại từ TRAIN users). Test chỉ
   chấm một lần cuối.

Khi kết thúc mỗi thí nghiệm: cập nhật `NhatKy.md` (mục 2 diary + mục 5 bảng số) và commit.

---

## 1. Trạng thái đã xác minh — KHÔNG cần chạy lại

Tất cả số dưới đây đã được đối chiếu với file gốc. Dùng làm mốc, đừng suy diễn lại.

### 1.1 Bảng sạch (chunked + mask_repeats, eval trên test, fold 0)

| Model | AUC sạch | n_scored | batch | epochs | phút |
|---|---|---|---|---|---|
| **GIKT** | **0.823807** | 1.090.574 | 8 | 10 | 142,7 |
| AKT | 0.819751 | 1.090.574 | 64 | 30 | 10,9 |
| **v4q (e40)** | **0.818340** | 1.090.610 | 64 | — | — |
| v4q seed 42 / 17 / 1234 | 0.818243 / 0.818109 / 0.817975 | 1.090.610 | 64 | — | — |
| SimpleKT (3 seed) | 0.817760 / 0.817858 / 0.817762 | 1.090.574 | 64 | 30 | ~5 |
| DKT | 0.798366 | 1.090.574 | 64 | 30 | 1,2 |

- v4q 3 seed: SD ≈ 0.00016 → **rất ổn định**, không phải may mắn một lần.
- Bootstrap learner-level (n_learners = 3.614, n_boot = 1.000):
  - v4q − GIKT = **−0.005467**, CI [−0.005903, −0.005009] → khoảng cách THẬT, không phải nhiễu.
  - v4q − AKT = **−0.001509**, CI [−0.001900, −0.001103].

### 1.2 Kiến trúc v4q hiện tại (đọc từ `dh2a_kt/models/dh2_kt.py`)

```
interaction = Emb(c_t + n_concepts · y_t)
observed    = item_scale(e_t)·concept_var(c_t) + concept_emb[c_t]      # Rasch
x_t         = LayerNorm(interaction + W_in · observed)
hidden      = LSTM(x)                                                   # n_lstm_layers
queried     = Rasch(c_{t+1}, e_{t+1})                                   # câu KẾ TIẾP
logit_t     = ⟨hidden_t, W_q · queried_t⟩/√H + concept_bias(c_{t+1}) + item_bias(e_{t+1})
```

**Thiếu so với hai mô hình đứng trên:** (a) attention trên lịch sử hidden;
(b) tổng hợp câu–KC quan sát. Đó chính là hai thí nghiệm E2 và E3.

### 1.3 Đặc trưng dữ liệu

- 865 khái niệm, **7.652 bài tập** (gấp 8,8 lần) — v4q chỉ dùng Rasch vô hướng cho bài tập.
- ~14,4% vị trí eval là dòng lặp (cùng user+item+timestamp), ~99,9% trùng đáp án dòng trước.
- ~12,6% attempt là multi-KC.
- Log rất dài (>200) → chế độ cửa sổ đổi AUC ~0,009 (**lớn hơn khoảng cách 0,0055 tới GIKT**).
- KC-repeat thấp (~21%) → lịch sử liên quan nằm **thưa và xa**.

---

## 2. Đã loại trừ — ĐỪNG thử lại

Ba nhóm dưới đây đã có bằng chứng phủ định rõ ràng. Chạy lại là lãng phí GPU.

### 2.1 Hypergraph tiên quyết không đóng góp AUC (8 cấu hình)

| Cấu hình | ΔAUC do graph |
|---|---|
| v2–v4 embedding tĩnh | ≈ 0 |
| sparse hyperedge (neigh/pair) | ≈ 0 |
| v5 clique transport | +0,00086 (0,799463 vs 0,798600) |
| attention only | 0,799567 (trung tính) |
| hyperedge embed | 0,804702 (tốt nhất có-graph, vẫn −0,019 so GIKT) |
| priority (attn+star+hedge+kind) | **−0,0469** (0,758950 vs 0,805854) |
| star only | **0,524332** — phá model |

**Chẩn đoán cấu trúc:** 466.305 siêu cạnh trên 865 khái niệm = **539 siêu cạnh mỗi khái niệm**.
Ở mật độ đó mean-pool biến mọi embedding khái niệm thành xấp xỉ cùng một vector — graph trở thành
**toán tử làm mượt** xóa danh tính khái niệm. Cộng thêm Edge F1 = 0,183 so với DAG chuyên gia Junyi
→ đồ thị vừa quá dày vừa phần lớn là nhiễu.

**Hệ quả:** nếu vẫn muốn thử graph, chỉ thử **đồ thị câu–KC QUAN SÁT ĐƯỢC** (E3), không thử
thêm biến thể nào của hypergraph tiên quyết suy diễn, và tuyệt đối không thử transport/star.

### 2.2 Cổng tin cậy C_B của GreyKT — tiền đề bị bác bỏ

| Tín hiệu C_B | Spearman(C_B, sai số²) | Kết luận |
|---|---|---|
| Tần suất huấn luyện N/(N+κ_B) | −0,0321 | BÁC BỎ |
| MC-dropout (8 mẫu) | −0,0199 | BÁC BỎ (yếu hơn cả tần suất) |
| Xác suất dự báo \|2p−1\| | −0,5877 | **DƯƠNG TÍNH GIẢ** |

- Kết quả tần suất **bác bỏ cả họ tham số**: C_B = N/(N+κ_B) đơn điệu theo N, mà Spearman chỉ
  phụ thuộc thứ hạng → tinh chỉnh κ_B hay đổi sang log(N) đều cho đúng −0,0321. Không thử nữa.
- \|2p−1\| là dương tính giả: với mô hình hiệu chỉnh tốt, E[(p−y)²|p] = p(1−p) = (1−C_B²)/4,
  tức quan hệ giảm là **đồng nhất thức toán học**. Mô phỏng đối chứng (mô hình hiệu chỉnh hoàn hảo,
  vô thông tin) cho ρ từ −0,59 đến −0,76; giá trị quan sát −0,5877 nằm ở đầu thấp khoảng đó.
  **Không dùng chế độ này cho số liệu báo cáo.**

### 2.3 GreyKT hợp nhất không tăng AUC (và không thể tăng)

`fused_auc = 0,7346` vs `black_auc = 0,7347` — chênh 0,0001. Nhưng
`ECE: 0,0867 → 0,0088` (**giảm 10 lần**), NLL 0,509 → 0,468, Brier 0,166 → 0,151.

GreyKT là **phương pháp hiệu chỉnh, không phải phương pháp tăng AUC**. Temperature scaling đơn điệu
nên AUC bất biến về mặt toán học. Đừng kỳ vọng hướng này đóng góp cho SOTA AUC.

---

## 3. Nhiệm vụ — 6 thí nghiệm, chạy theo đúng thứ tự

Ghi kết quả vào `results/tables/`, tên file có tiền tố `sota_`.

### E0 — Đối chứng: lợi thế "bất biến giao thức" có thật không? (ưu tiên cao nhất, ~15 phút)

**Bối cảnh.** v4q có `auc_attributable_to_repeats` **âm** (−0,0003 đến −0,0022): cùng một
checkpoint cho 0,8179 (leaky) và 0,8182 (clean). Trong khi baseline rớt −0,054 đến −0,059.
**Nhưng** với baseline đó là **hai lần huấn luyện khác nhau** (`A_dkt_repro` vs `C_dkt_clean`),
còn với v4q là **cùng một checkpoint chấm hai lần**. Hai phép so sánh không đối xứng.
Thêm nữa, `pykt_clean.py` cho thấy baseline cấu hình C **cũng** được mask repeat lúc train.

**Việc cần làm.** Lấy checkpoint `C_simplekt_clean` (đã train với mask_repeats) và chấm nó trên
**giao thức leaky** (mask_repeats=False, cùng cửa sổ chunked). Làm tương tự cho `C_akt_clean`.

**Cổng quyết định:**
- Nếu `C_simplekt_clean` trên leaky cũng ≈ 0,818 → **lợi thế không tồn tại**, nó chỉ là hệ quả của
  train-time masking mà ai cũng có. Xóa mọi phát biểu về "bất biến giao thức" khỏi nháp bài báo.
- Nếu nó vẫn vọt lên ~0,87 → lợi thế của v4q là **thật và riêng có**, và đây là đóng góp mạnh
  cần đưa lên phần đầu bài báo.

Ghi: `results/tables/sota_E0_protocol_invariance_control.csv` với cột
`model, trained_with_mask, eval_protocol, auc, n_scored`.

---

### E1 — Quét độ dài cửa sổ (rẻ nhất, làm trước khi đổi kiến trúc, ~1–2h)

Nhật ký ghi chế độ cửa sổ đổi AUC ~0,009 — lớn hơn khoảng cách tới GIKT — nhưng **độ dài chunk
chưa từng được sweep**.

```bash
for L in 100 150 200 300 400; do
  python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda \
    --architecture v4 --use-questions --mask-repeats --window-mode chunked \
    --val-frac 0.1 --early-stop-patience 5 --seed 42 \
    --max-seq-len $L --tag sota_E1_chunk$L
done
```

> **Đã kiểm tra: `--max-seq-len` CHƯA tồn tại trong `scripts/03_train_tier1.py`** (chỉ có
> `--batch-size`, `--epochs`, `--lr`, `--hidden-dim`, `--dropout`, `--lstm-layers`).
> Việc đầu tiên của E1 là thêm flag này. Đường đi đã có sẵn:
> `resolve_training_budget(p0_cfg, ..., max_seq_len=<giá trị>)` ở `dh2a_kt/train/tier1.py:86`
> nhận tham số này và mặc định lấy `p0_cfg.pykt.max_seq_len` (=200, khớp `configs/xes3g5m.yaml:36`).
> Chỉ cần thêm `argparse` rồi truyền xuống — **đừng sửa file config**, vì config đang là mốc
> tái lập của bảng baseline.

**Cổng:** chốt L tốt nhất theo **val AUC**, không theo test. Mọi thí nghiệm sau dùng L này.

---

### E2 — Recap attention trên lịch sử (đòn bẩy lớn nhất, ~2–3h)

**Lý do có cơ sở trên chính dataset này:** log >200 bước và KC-repeat chỉ ~21%, nghĩa là lịch sử
liên quan tới khái niệm kế tiếp nằm **thưa và xa** — đúng thứ attention truy hồi được còn trạng thái
LSTM thiên vị gần thì quên. Đây cũng là cơ chế nhật ký quy cho GIKT (mục 4.4), và AKT (có attention)
đang đứng trên v4q.

**Sửa `_readout_v4`** — hiện chỉ dùng `hidden_t`:

```python
# Thêm vào __init__ khi architecture == "v4":
self.recap_proj = nn.Linear(hidden, hidden)

# Trong _readout_v4, thay logits = ⟨hidden, W_q·queried⟩ bằng:
q = self.query_proj(queried)                                   # (B, T, H)
scores = torch.einsum('bth,bsh->bts', q, hidden) / math.sqrt(H) # (B, T, S)
causal = torch.ones(T, T, dtype=torch.bool, device=...).tril()  # s <= t
scores = scores.masked_fill(~causal, float('-inf'))
attn = scores.softmax(dim=-1)
context = torch.einsum('bts,bsh->bth', attn, hidden)            # (B, T, H)
logits = ((hidden + self.recap_proj(context)) * q).sum(-1, keepdim=True) / math.sqrt(H)
logits = logits + self.concept_bias(next_concepts) + self.item_bias(next_items)
```

Bổ sung `--recap-attention` làm cờ bật/tắt để chạy ablation sạch.

**Bắt buộc kiểm tra tính nhân quả** trước khi tin số: viết một unit test đặt một giá trị cực đại
vào `hidden[:, t+1:]` và xác nhận `logits[:, :t]` không đổi. Nếu test này trượt thì AUC cao là do
rò rỉ tương lai, không phải do mô hình tốt.

**Cổng:** ΔAUC ≥ **+0,002** trên val thì giữ; nhỏ hơn thì tắt cờ và ghi vào nhật ký.

---

### E3 — Tổng hợp câu–KC QUAN SÁT ĐƯỢC (~2–3h)

**Đây là đồ thị khác hoàn toàn với hypergraph tiên quyết đã bị bác bỏ ở mục 2.1.** Nó là
incidence *quan sát được* (bài tập nào chạm KC nào), phủ 100% bài tập, không suy diễn.

```python
# Trong _concept_vectors_v4, sau khi có Rasch vector:
#   q_vec = item_scale(e) * concept_var(c) + concept_emb[c]
# thêm:
#   q_vec = q_vec + mean_{k in KC(e)} concept_embed[k]
```

Xây `KC(e)` từ chính dữ liệu huấn luyện (không dùng E_pre). Bổ sung cờ `--question-kc-agg`.

**Cổng — đây là cổng quan trọng nhất cho tuyên bố graph của cả bài:**
chạy cả `--question-kc-agg` và bản không có nó, cùng seed. **ΔAUC(graph) ≥ +0,002** thì bài có
một tuyên bố graph đứng vững; nhỏ hơn thì **ghi thẳng vào bài rằng graph không đóng góp AUC** —
đừng che giấu.

---

### E4 — Khớp ngân sách tính toán (~4h)

GIKT dùng `batch=8, 10 epoch, 142,7 phút`; v4q dùng `batch=64`. Chênh ~8 lần số bước gradient.
Nguyên tắc "matched budget" mà dự án áp dụng ở chỗ khác đã không được áp ở đây.

```bash
for BS in 16 32; do
  python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda \
    --architecture v4 --use-questions --mask-repeats --window-mode chunked \
    --batch-size $BS --epochs 30 --val-frac 0.1 --early-stop-patience 5 \
    --seed 42 --tag sota_E4_bs$BS
done
```

Ghi cả **thời gian chạy** vào CSV. Bài báo phải báo cáo compute, nếu không phản biện sẽ hỏi.

---

### E5 — Ensemble 3 seed (chi phí 0, đã có checkpoint)

Trung bình **logit** (không phải xác suất) của 3 seed đã có. Báo cáo thành **một dòng riêng**
ghi rõ "ensemble of 3 seeds", không trộn lẫn với dòng đơn mô hình.

---

### E6 — Gộp sự kiện multi-KC, KHÔNG transport (tùy chọn, chỉ khi E2+E3 chưa đủ)

v5 chẩn đoán đúng bệnh (12,6% multi-KC) nhưng kê sai thuốc (memory transport — đã bị bác bỏ).
Tách riêng phần đúng: gộp các dòng KC của cùng một attempt thành một sự kiện, lấy **trung bình**
vector khái niệm, giữ nguyên LSTM + readout của v4q. **Không** transport, **không** memory bank.

---

## 4. Báo cáo cuối phiên

Tạo `results/tables/sota_summary.csv`:

```
exp_id, tag, architecture, flags, window_L, batch, seed,
val_auc, test_auc, n_scored, delta_vs_v4q_baseline, minutes, gate_passed, note
```

Và cập nhật `NhatKy.md`:
- Mục 2 (diary): thêm "Giai đoạn H — hướng SOTA v4q", ghi cả thí nghiệm **thất bại**.
- Mục 5 (bảng số): thêm dòng mới.
- Mục 6 (checklist): tick việc đã xong.

**Chạy bootstrap cho mọi so sánh mà bạn định phát biểu:**

```bash
python scripts/25_bootstrap_protocol_ci.py --a <tag_moi> --b gikt --n-boot 1000 --seed 42
```

Chỉ tuyên bố "vượt" khi CI của **hiệu** loại trừ 0. Đừng dùng CI của từng AUC riêng lẻ —
chúng chồng lấn ngay cả khi hiệu có ý nghĩa (đã thấy ở bootstrap v4q vs GIKT).

---

## 5. Cảnh báo cuối — cái bẫy về novelty

E2 và E3 làm v4q **hội tụ về phía GIKT** về mặt kiến trúc. Nếu vượt được SOTA nhờ chúng, phản biện
sẽ hỏi ngay *"khác GIKT ở đâu?"*.

Càng nguy hiểm hơn: nếu vượt SOTA nhờ E2 (attention, không liên quan graph) trong khi E3 cho
ΔAUC(graph) ≈ 0, thì bài có tiêu đề về hypergraph mà cơ chế đó đóng góp bằng 0 — tình huống này
**tệ hơn** là không vượt SOTA.

Vì vậy: dù kết quả thế nào, **báo cáo trung thực ΔAUC(graph) từ E3**. Nếu nó ≈ 0, đóng góp thật
của bài nằm ở chỗ khác — phát hiện rò rỉ multi-KC, bảng sạch tái lập được, và đồ thị đã kiểm toán
rò rỉ — và bài nên được viết theo trục đó ngay từ đầu thay vì biện minh sau.
