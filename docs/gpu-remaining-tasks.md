# GPU remaining tasks — DH²A-KT (2026-09-05)

**Đợt 08–09/09 (Hau ver4 A1/A2/B6–B9):** **đã xong** trên
`gpu-hau-ver4-abc-2026-09-08` (`e3a2382`). Không chạy lại
38–42 / S5–S6. Quyết định A1: `aligned_dt_isolated_from_branch=false`
(0/5). Laptop đã điền số vào bản thảo.

**Đợt 08/09 (test-only rescore):** đã xong trên
`gpu-rescore-testonly-2026-09-08`. Không chạy lại
`scripts/36_rescore_a2_testonly.py` trừ bước S5 (chỉ A7 seed 42).

**Bản thảo:** `origin/main` (attribution / QKC-T).  
**Máy:** một RTX 3090. Seed cố định `{0, 17, 42, 1234, 2024}`.  
**Headline hiện tại không phụ thuộc danh sách P1.** Capacity-matched Q←KC, Zero+Δt, B12 rewiring, A2 5-seed (hàng cạnh tranh + DKT/DGEKT), A7 5-seed, ECE tracer, leakage full-log, **S1–S5 Gate 0 GPU (07/09)** đã xong — **cấm chạy lại** trừ khi `--force` có lý do ghi rõ.

**Đợt 07/09/2026 (Gate 0 máy):** xong trên nhánh `gpu-checklist-2026-09-07` (`d613fd6`), đã fast-forward vào `main`. S1 ECE pyKT; S2 bootstrap học viên; S3 P4 0/5; S4 M4 0/5; S5 SKIP. File này giữ hàng đợi lịch sử; **không** bắt đầu P1 A3.

Sau `git pull` trên `main`:

```bash
cd /path/to/DH2A-KT
git pull --ff-only
```

Ghi kết quả vào `results/tables/` (CSV/JSON allowlist) + log; không sửa số bằng tay trong `paper/tables/` nếu exporter đã có.

---

## Không chạy lại

| Việc | Driver | Kết quả đã có |
|---|---|---|
| Q←KC capacity-matched + Zero+Δt | `scripts/34_qkc_capacity_matched.py` | `qkc_capacity_matched_{matrix.csv,summary.json}` |
| A2 5-seed tracer / QKC / twin / GIKT / AKT / simpleKT | `scripts/27_a2_multiseed_matrix.py` | `a2_multiseed_matrix.csv` |
| A7 4 nhánh Δt seed 42 | `scripts/31_a7_query_dt_ablation.py` | `a7_query_dt_ablation.csv` |
| ECE/Brier/NLL 3 nhánh DH² | `scripts/30_a6_calibration_metrics.py` | `a6_calibration_xes3g5m_fold0.csv` |
| B12 rewire + permute folds 0–2 | `scripts/32_b12_structure_checks.py` | JSON `*_degree_preserving_rewire` / `*_relation_label_permute` |
| Faithfulness Ollama n=500 | `scripts/18_run_kbs_faithfulness_suite.py` | `xes3g5m_fold0_kbs_faithfulness_comparison.json` |

---

## Hàng đợi (làm theo thứ tự)

Ưu tiên **P2 trước P1** trừ khi tác giả quyết định đổi giao thức chuẩn. P1 A3 có thể buộc chạy lại toàn bộ Bảng 5 — đừng bắt đầu P1 nếu chưa chốt.

### P2 — rẻ, củng cố bản đang nộp (khuyến nghị)

#### P2.1  A4 còn lại — ECE / Brier / NLL cho pyKT (GIKT, AKT, simpleKT)

Cùng checkpoint A2 đã train. `scripts/30_a6_calibration_metrics.py` hiện load checkpoint DH² (`.pt`). Với pyKT: lấy `test_predictions.npz` từ thư mục tag A2 rồi chạy `scripts/25_bootstrap_protocol_ci.py` **không** thay ECE — ECE pyKT cần một bước chấm lại xác suất trên giao `n=1,093,720`.

Nếu driver ECE pyKT chưa có: chấm `scripts/24_train_baselines_clean.py` **không train** không tồn tại; dùng npz đã lưu:

```text
results/pykt_work_clean/xes3g5m/fold_0/C_gikt_clean_L400_e30b16/
results/pykt_work_clean/xes3g5m/fold_0/C_akt_clean_L400/
results/pykt_work_clean/xes3g5m/fold_0/C_simplekt_clean_L400/
```

Tiêu chí xong: bảng calibration thêm 3 hàng pyKT; ECE > 0.05 thì ghi Limitations, không temperature-scale trừ khi tác giả yêu cầu.

#### P2.2  A2 còn lại — DKT + DGEKT 5-seed (clean L=400)

```bash
python scripts/24_train_baselines_clean.py configs/xes3g5m.yaml --fold 0 \
  --model dkt --window-mode chunked --mask-repeats --max-seq-len 400 \
  --tag C_dkt_clean_L400 --device cuda --seed 42
# lặp seed 17, 1234, 0, 2024; tương tự --model dgekt --tag C_dgekt_clean_L400
```

Ghi mean±SD vào hàng DKT/DGEKT Bảng AUC (hiện một seed). Không bắt buộc để giữ khung attribution.

#### P2.3  A7 còn lại — 5-seed cho 4 nhánh Δt (không phải planned-gap)

`scripts/31_a7_query_dt_ablation.py` mặc định seed 42. Lặp:

```bash
python scripts/31_a7_query_dt_ablation.py --device cuda --seed 17
python scripts/31_a7_query_dt_ablation.py --device cuda --seed 1234
python scripts/31_a7_query_dt_ablation.py --device cuda --seed 0
python scripts/31_a7_query_dt_ablation.py --device cuda --seed 2024
```

Tiêu chí xong: mỗi mode `both` / `lstm` / `query` có 5 Δval; cổng +0.002 vẫn chỉ PASS ở `both` (hoặc báo nếu khác).

#### P2.4  B10 còn lại — bootstrap trên giao 1,093,720

Không train. Cặp npz QKC-T vs GIKT/AKT/simpleKT **cùng tập vị trí** (giao pyKT), rồi:

```bash
python scripts/25_bootstrap_protocol_ci.py \
  --a "QKC-T=..." \
  --b "simpleKT=..."
```

Tiêu chí xong: CI in trong bài là hiệu trên **cùng n scored** với hàng ± 5-seed, không còn cặp A1 `0.829593` vs `0.819934`.

---

### P1 — Gate 1 protocol (chỉ khi đổi giao thức chuẩn)

Đừng chạy nếu giữ headline **mask-repeats + chunked + L=400**.

#### P1.1  A3 — ba giao thức, một fold, một seed (42)

`scripts/03_train_tier1.py --window-mode` chỉ nhận `first` | `chunked` (không có `last`). So:

| Nhánh | Cờ | Ý |
|---|---|---|
| raw / leaky | `--window-mode chunked` **không** `--mask-repeats` | chấm mọi hàng KC mở rộng |
| mask-target (headline) | `--window-mode chunked --mask-repeats` | đang dùng |
| cửa sổ một đoạn | `--window-mode first --mask-repeats` | gần P0 “một cửa sổ / user” |

```bash
# (d) cùng fold 0, tracer khuyến nghị
python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda \
  --architecture v4 --use-questions --question-graph --no-graph --no-session \
  --time-gap --time-gap-mode both --max-seq-len 400 --batch-size 16 --epochs 30 \
  --val-frac 0.1 --early-stop-patience 5 --seed 42 \
  --window-mode chunked
# rồi thêm --mask-repeats; rồi --window-mode first --mask-repeats
```

**(a)(b)** phân bố Δt tại hàng lặp / raw loại Δt=0: chưa có driver riêng — ghi `results/tables/a3_dt_repeat_dist.json` bằng script chẩn đoán (có thể bắt đầu từ `results/tables/_diag_slice_auc.py`).  
**(c) attempt-level collapse:** có test `tests/test_events.py::test_collapse_yields_one_row_per_attempt_*` nhưng **chưa** nối vào `03_train_tier1.py`. Phải viết driver trước khi train.

**(e)** nếu attempt-collapse (hoặc `first`) lệch AUC đáng kể so với headline → **dừng**, báo tác giả, **không** tự đổi Bảng 5.

#### P1.2  A7 planned-gap (deployment-realistic)

Bài đã viết: không đánh giá proxy “gap kế hoạch trước khi learner mở item”. Chỉ chạy nếu tác giả chốt định nghĩa proxy trong code. Không bịa cờ CLI.

#### P1.3  B9 — rescore simpleKT L=200 P0, không train

Chỉ khi còn checkpoint P0. Xác nhận đường dẫn với tác giả trước.

```bash
# DH² checkpoint only (driver reports masked vs unmasked AUC side by side):
python scripts/23_eval_clean_protocol.py configs/xes3g5m.yaml --fold 0 --device cuda \
  --eval-split test --window-mode chunked --max-seq-len 400 \
  --checkpoint NAME=path/to.pt
```

simpleKT P0 đi qua pyKT: tìm checkpoint/tag L=200 rồi chấm trên export clean — **không train lại**. `23_eval_clean_protocol.py` không nhận checkpoint pyKT. Không có file thì bỏ qua, ghi `SKIP: checkpoint missing`.

---

### P3 — tuỳ quỹ máy (không chặn nộp)

#### P3.1  B11 — tuning, không chỉ inherit

- AKT: search nhỏ tại `L=400` (lr / dropout / d_model), ghi `n` configs vào `table_hparams.tex`.
- GIKT: nới cap epoch (patience 5 giữ), so với run e30b16 đã early-stop epoch 13.
- simpleKT `L=200` clean vs `L=400` (một seed đủ để tách độ dài chuỗi).

```bash
python scripts/24_train_baselines_clean.py configs/xes3g5m.yaml --fold 0 \
  --model simplekt --window-mode chunked --mask-repeats --max-seq-len 200 \
  --tag C_simplekt_clean_L200 --device cuda --seed 42
```

#### P3.2  Tracer + GIKT: 3 fold × 3 seed

Chỉ QKC-T và GIKT. Fold 1–2 QKC-T đã có một seed (`table_auc_p0_dt_folds.tex`).

#### P3.3  LLM-scale sweep (Ollama)

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode ollama \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt \
  --device cuda --run-scale
```

Không phải việc attribution.

#### P3.4  A5 đắt (Gate 3) — Algorithm 1 có thứ tự + M5 Junyi ≥3 seed

Chỉ khi tác giả chọn **không** giữ phương án rẻ (path-derived, tập không thứ tự). Đòi sửa `build_concept_prerequisite_hyperedges` rồi train Junyi — ngoài hàng đợi mặc định.

---

## Việc không phải GPU (đừng đưa vào hàng đợi máy)

- Caption `pre-registered` / “registered gate” → `pre-specified`.
- In *p* một phía có nhãn, hoặc in *p* hai phía.
- `\input{tables/table_attribution_summary.tex}`.
- Đổi tên config `hyperedge.concept_prerequisite` cho khớp “path-derived”.

B12–B15 / C3 trên kế hoạch ver3: **đã chọn hướng** trên bản attribution; không chờ P1–P3.

---

## Nghiệm thu khi đẩy số về laptop

1. CSV/JSON mới nằm dưới `results/tables/` và có mặt trong `git status`.
2. Không overwrite `a2_multiseed_matrix.csv` / `qkc_capacity_matched_*.csv` trừ khi hàng mới là DKT/DGEKT (append).
3. Mọi AUC headline vẫn `mask-repeats` + `chunked` + `L=400` trừ khi P1.1 **(e)** đã được tác giả chốt.
4. Cập nhật một dòng trong `docs/paper-artifacts.md` cho file mới.
