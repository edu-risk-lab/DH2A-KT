# Hướng dẫn GPU — nghiệm thu GS Hậu ver4 (A1, A2, B6–B9)

Đọc cùng: `docs/Nhan_xet_DH2A_KT_ver4_Hop_nhat_ABC_2026Sep08.pdf` và
`docs/HuongDan_Rescore_TestOnly_2026-09-08.md` (đợt 08/09 **đã xong**, không chạy lại).

**Mục tiêu:** lấy bằng chứng máy cho 6 mục còn mở. Không đổi giao thức
headline (mask-repeats, chunked, L=400). Không bịa số nếu một bước SKIP.

Nhánh đề xuất: **`gpu-hau-ver4-abc-2026-09-08`**. Seed: `{42, 17, 1234, 0, 2024}`.
Một RTX 3090. Python: `export DH2A_PYTHON=python` nếu cần.

Log tổng: `results/tables/gpu_hau_ver4_abc_2026-09-08.log`.

---

## 0. Kéo code (làm trước mọi lệnh)

```bash
cd /path/to/DH2A-KT
git fetch origin
git checkout gpu-hau-ver4-abc-2026-09-08
git pull --ff-only
git log -1 --oneline
```

Nếu nhánh chưa có trên remote: checkout commit laptop đã push, rồi
`git checkout -b gpu-hau-ver4-abc-2026-09-08`.

---

## Cấm chạy lại

| Việc | Script | Lý do |
|---|---|---|
| Capacity-matched Q←KC + Zero+Δt | `scripts/34_qkc_capacity_matched.py` | T-real **tái sử dụng** `qkc_cm_zero_time` |
| A2 5-seed train (QKC-T / Q←KC / GIKT / …) | `scripts/27_a2_multiseed_matrix.py` | đã có |
| A7 `both` (mọi seed) | `scripts/31_a7_query_dt_ablation.py` không `--arms` đủ 3 | `both` = QKC-T |
| ECE 3 nhánh DH² seed 42 (trừ Zero+Δt) | `scripts/30_a6_calibration_metrics.py` ghi đè CSV cũ | dùng `41_*` để *append* |
| B12 rewire/permute kiểu cũ | `scripts/32_b12_structure_checks.py` | thay bằng `42_*` |
| Attempt-collapse / đổi giao thức chuẩn | — | không chạy |
| Planned-gap | — | không chạy |
| `--force` trên 27 / 31 / 34 | — | cấm trừ khi log ghi rõ file hỏng |

Không sửa tay `paper/tables/*.tex` trên máy GPU.

---

## Hàng đợi — thứ tự bắt buộc

`S0 → S1 → S2 → S3 → S4` (không train) rồi `S5 → S6` (train).
Có thể dừng sau S4 nếu hết ca; S5/S6 là phần đắt.

### S0 — Liệt kê checkpoint (vài giây)

```bash
python scripts/38_a1_timing_input_controls.py --discover
python scripts/36_rescore_a2_testonly.py --discover
ls results/checkpoints/xes3g5m_fold{0,1,2}.pt
ls results/checkpoints/xes3g5m_fold0_qkc_cm_zero_time_s{42,17,1234,0,2024}.pt
ls results/checkpoints/xes3g5m_fold0_a7_dt_{lstm,query}_s42.pt
```

**Xong khi:** log có `OK`/`MISSING` từng arm×seed.

| Nếu thiếu | Việc làm |
|---|---|
| Cả 5 `qkc_cm_zero_time` | `STOP: no T-real`; **không** gọi script 34 |
| `xes3g5m_fold{0,1,2}.pt` | S4 SKIP từng fold; không train v2 |
| `a7_dt_lstm_s42` / `a7_dt_query_s42` | đúng mục S5; không train seed khác |
| pyKT `test_predictions.npz` | S2 SKIP; không train baseline |

### S1 — B8: ECE Zero+Δt (không train)

```bash
python scripts/41_b8_zero_dt_calibration.py --device cuda
```

Ghi thêm hàng `Zero+Δt` vào `results/tables/a6_calibration_xes3g5m_fold0.csv`.
Không temperature-scale. Không xóa hàng QKC-T / Q←KC.

**Xong khi:** CSV có hàng `Zero+Δt`; `n_scored=1093755`. Thiếu ckpt → `SKIP`, không train.

### S2 — B6: join 35 hàng (không train)

```bash
python scripts/39_b6_join_prediction_ids.py --device cuda
```

Cần: `xes3g5m_fold0_p0_dt_on_s42.pt` + một
`results/pykt_work_clean/xes3g5m/fold_0/C_gikt_clean_L400*/test_predictions.npz`
(ưu tiên `s42`).

Ghi:

- `results/tables/b6_id_join_summary.json`
- `results/tables/b6_unmatched_dh2_rows.csv`

Khóa join: `user|kc|occurrence` trên tập đã chấm (chưa phải `attempt_id` gốc).

**Xong khi:** JSON có `n_dh2`, `n_pykt`, `n_joined`, `n_only_dh2`.
`n_only_dh2` kỳ vọng 35; nếu khác, giữ số máy, không sửa tay.

### S3 — A2: trace luồng thông tin (CPU, không train)

```bash
python scripts/40_a2_infoflow_trace.py
```

Ghi `results/tables/a2_infoflow_trace.json`.

**Xong khi:** JSON có `n_attempts_split_across_chunk` và
`n_windows_starting_on_global_repeat`. Đây **không** phải evaluator
all-in-one; không đổi tên giao thức headline.

### S4 — B9: cùng checkpoint, một AUC clean (GPU, không train)

```bash
python scripts/42_b9_same_ckpt_destruction.py --device cuda
```

Ghi `results/tables/b9_same_ckpt_destruction.csv` và `*.json`.
Cột `auc_clean_shared` phải giống nhau trên 3 operator cùng fold.
`clean_auc_abs_diff` lớn → ghi vào JSON; không bịa một AUC clean mới.

**Xong khi:** 3 fold × 3 operator, hoặc SKIP fold thiếu `.pt`.

### S5 — B7: train đúng seed 42 history / query

Chỉ hai arm thiếu checkpoint. **Không** `--force`. **Không** train `dt_both`.

```bash
python scripts/31_a7_query_dt_ablation.py --device cuda --seed 42 --arms dt_lstm,dt_query
```

Script 31 đã đổi: có `.pt` + hàng CSV thì SKIP; thiếu `.pt` thì train dù CSV đã có val.

**Xong khi:**

```text
results/checkpoints/xes3g5m_fold0_a7_dt_lstm_s42.pt
results/checkpoints/xes3g5m_fold0_a7_dt_query_s42.pt
```

Rồi chấm test-only (không train):

```bash
python scripts/36_rescore_a2_testonly.py --device cuda --arms a7_dt_lstm,a7_dt_query --seeds 42
```

Không overwrite `a2_multiseed_matrix.csv`.

### S6 — A1: train T-zero và T-misaligned (phần đắt)

Cùng backbone zero-incidence + nhánh Linear `both`. Chỉ đổi *input* gap.
T-real = reuse `qkc_cm_zero_time` (S0 phải `OK` cả 5 seed).

```bash
python scripts/38_a1_timing_input_controls.py --device cuda
```

Một seed × một arm mới ≈ một lần train QKC-T (batch 16, cap 30, patience 5).
Hai arm mới × 5 seed ≈ **10 lần train**. Dự trữ cả ca.

Ghi:

- `results/tables/a1_timing_input_controls.csv`
- `results/tables/a1_timing_input_controls_summary.json`
- checkpoint `results/checkpoints/xes3g5m_fold0_a1_t_{zero,misaligned}_s{SEED}.pt`

**Nghiệm thu A1 (máy, không sửa conclusion trên GPU):**

Trong `*_summary.json`:

- `decision.aligned_dt_isolated_from_branch == true` chỉ khi
  `t_real` vs `t_zero` **và** `t_real` vs `t_misaligned` đều `5/5` gate val.
- Nếu không: giữ câu package (nhánh + Δt), đúng PDF thầy.

Mọi arm phải có `has_time_gap_proj=true` và `incidence_nonzero=0`.
`state_numel` T-zero / T-misaligned / T-real phải khớp.

Thiếu T-real → script trả exit 3; **dừng**, không gọi 34.

---

## Việc không phải máy (laptop / tác giả)

1. Đọc CSV/JSON rồi sửa Abstract / Table 5 / calibration / A.20.
2. Holm / gate vs *t*-test (đã khóa trên laptop).
3. GitHub / Zenodo.
4. Compile PDF.
5. Viết evaluator all-in-one cấp câu hỏi — **ngoài đợt này**
   (S3 chỉ trace). Nếu S3 báo `n_attempts_split_across_chunk > 0`,
   ghi Limitations, không collapse attempt trên máy.

---

## Đẩy số về laptop

```bash
git status
git add \
  results/tables/a1_timing_input_controls.csv \
  results/tables/a1_timing_input_controls_summary.json \
  results/tables/a1_timing_logs \
  results/tables/a6_calibration_xes3g5m_fold0.csv \
  results/tables/a6_reliability_xes3g5m_fold0.json \
  results/tables/a6_calibration_zero_dt_only.csv \
  results/tables/a6_reliability_zero_dt.json \
  results/tables/b6_id_join_summary.json \
  results/tables/b6_unmatched_dh2_rows.csv \
  results/tables/a2_infoflow_trace.json \
  results/tables/b9_same_ckpt_destruction.csv \
  results/tables/b9_same_ckpt_destruction.json \
  results/tables/a2_testonly_rescore.csv \
  results/tables/a2_testonly_rescore_summary.json \
  results/tables/a7_query_dt_ablation.csv \
  results/tables/a7_dt_lstm_s42.csv \
  results/tables/a7_dt_query_s42.csv \
  results/tables/gpu_hau_ver4_abc_2026-09-08.log
# không add checkpoints .pt, pykt_work_clean, jsonl Ollama
git commit -m "Add Hau ver4 GPU artefacts: A1 timing controls, A2 trace, B6-B9."
git push origin gpu-hau-ver4-abc-2026-09-08
```

Checklist trước khi push:

1. Không đụng `qkc_capacity_matched_*`, `a2_multiseed_matrix.csv` (trừ rescore test-only riêng).
2. Headline vẫn mask-repeats + chunked + L=400.
3. Log S5/S6 không chứa lệnh `27_a2` / `34_qkc`.
4. `a1_timing_input_controls_summary.json` có `decision` rõ true/false.
5. Calibration vẫn có hàng cũ + hàng Zero+Δt.

---

## Ánh xạ mục thầy → script

| Mục | Script | Train? |
|---|---|---|
| A1 T-real / T-zero / T-misaligned | `38_a1_timing_input_controls.py` | chỉ zero + misaligned |
| A2 trace + biên chunk | `40_a2_infoflow_trace.py` | không |
| B6 35 hàng | `39_b6_join_prediction_ids.py` | không |
| B7 seed 42 lstm/query | `31_a7_query_dt_ablation.py --arms dt_lstm,dt_query` + `36_*` | có, 2 run |
| B8 ECE Zero+Δt | `41_b8_zero_dt_calibration.py` | không |
| B9 cùng clean AUC | `42_b9_same_ckpt_destruction.py` | không |
