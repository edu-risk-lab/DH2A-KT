# Hướng dẫn chấm lại test-only — GPU server (08/09/2026)

Đọc cùng: `docs/NhanXet_DH2A-KT_attribution_Tuan_07Sep2026.md` (A1/A3) và `docs/gpu-remaining-tasks.md` (cấm chạy lại).

**Mục tiêu đợt này:** chấm lại checkpoint A2/A7 *đã train* trên đúng split test (mask-repeats, chunked, L=400), rồi chẩn đoán rẻ Δt tại hàng lặp. **Không train. Không đổi giao thức headline.**

Nhánh: **`gpu-rescore-testonly-2026-09-08`**. Seed: `{42, 17, 1234, 0, 2024}`. Một RTX 3090. Eval có thể dùng CUDA; S2 chỉ CPU.

---

## 0. Kéo code (làm trước mọi lệnh)

```bash
cd /path/to/DH2A-KT
git fetch origin
git checkout gpu-rescore-testonly-2026-09-08
git pull --ff-only
git log -1 --oneline
# kỳ vọng commit có message: Rescore A2 test-only
```

Python: `export DH2A_PYTHON=python` nếu cần.

Log:

- `results/tables/rescore_testonly_2026-09-08.log`
- `results/tables/a3_dt_repeat_diag.log`

---

## Cấm chạy lại (đã có số; đừng `--force`)

| Việc | Script |
|---|---|
| Capacity-matched Q←KC + Zero+Δt | `scripts/34_qkc_capacity_matched.py` |
| A2 5-seed train (QKC-T / Q←KC / twin / GIKT / AKT / simpleKT / DKT / DGEKT) | `scripts/27_a2_multiseed_matrix.py` |
| A7 5-seed train `both` / `lstm` / `query` | `scripts/31_a7_query_dt_ablation.py` |
| ECE DH² / pyKT | `scripts/30_a6_calibration_metrics.py`, `scripts/35_pykt_calibration_from_npz.py` |
| B12 rewire/permute | `scripts/32_b12_structure_checks.py` |
| P4 / M4 5-seed | `scripts/03_train_tier1.py` (`--concept-forget` / `--group-embed`) |
| Attempt-collapse / đổi giao thức chuẩn | không chạy |
| Planned-gap A7 | không chạy |

Không sửa tay `paper/tables/table_auc_xes3g5m.tex` / `table_a7_query_dt.tex` trên máy. Không overwrite `a2_multiseed_matrix.csv` hay `qkc_capacity_matched_*`.

---

## Hàng đợi — thứ tự

Làm **S0 → S1 → S2**. S1 cần checkpoint `.pt`. S2 không cần GPU.

### S0 — Liệt kê checkpoint (vài giây)

```bash
python scripts/36_rescore_a2_testonly.py --discover
```

Đường dẫn kỳ vọng:

```text
results/checkpoints/xes3g5m_fold0_p0_dt_on_s{42,17,1234,0,2024}.pt
results/checkpoints/xes3g5m_fold0_hg_qkc_on_s{42,17,1234,0,2024}.pt
results/checkpoints/xes3g5m_fold0_a7_dt_lstm_s{42,17,1234,0,2024}.pt
results/checkpoints/xes3g5m_fold0_a7_dt_query_s{42,17,1234,0,2024}.pt
```

`a7_dt_both` thường không có file riêng (A7 import từ A2): script sẽ dùng `p0_dt_on` làm alias.

**Xong khi:** log in `OK` / `MISSING` từng cặp arm×seed. Nếu **cả 5** `p0_dt_on` đều `MISSING`: ghi `STOP: no QKC-T A2 checkpoints`, **dừng, không train**.

### S1 — Chấm test-only (không train)

```bash
python scripts/36_rescore_a2_testonly.py --device cuda
```

Hết VRAM → `--device cpu`. Một checkpoint khoảng vài phút trên 3090.

Script ghi:

- `results/tables/a2_testonly_rescore.csv`
- `results/tables/a2_testonly_rescore_summary.json`

Cột bắt buộc: `arm,seed,test_only_auc,n_test_predictions,status`.  
`status=ok` phải có `n_test_predictions=1093755` (native DH²). Không trừ combined cho test-only.

Kiểm seed 42 QKC-T: `test_only_auc` phải khớp `a6_calibration` **0.829593** trong ±5e-5. Nếu lệch: ghi `WARN` vào log, vẫn giữ số vừa chấm, không sửa CSV tay.

**Xong khi:**

| Arm | Ý nghĩa | Cần |
|---|---|---|
| `p0_dt_on` | QKC-T — thay ô seed-42 trên Bảng 5 khi đủ 5 seed | 5 `ok` |
| `hg_qkc_on` | Q←KC A2 (khác run capacity-matched) | 5 nếu có `.pt`; thiếu thì `missing_checkpoint` |
| `a7_dt_lstm` / `a7_dt_query` | A7 cùng metric test-only | 5 nếu có `.pt` |
| `a7_dt_both` | alias QKC-T | theo `p0_dt_on` |

Thiếu từng seed: để `status=missing_checkpoint`, **không** gọi `03_train_tier1.py`.

### S2 — Chẩn đoán rẻ Δt tại hàng lặp (CPU, không checkpoint)

```bash
python scripts/37_a3_dt_repeat_diag.py
```

Ghi `results/tables/a3_dt_repeat_dist.json`.

Kỳ vọng: `frac_repeat_raw_dt_zero` ≈ 1.0 (cùng timestamp ⇒ Δt = 0). Đó là giả thuyết residual +0.0101, không phải lệnh đổi giao thức.

**Xong khi:** JSON có `counts` + `delta_t`. `decision.run_attempt_collapse` phải là `false`. Không train collapse.

---

## Việc không phải máy (laptop / tác giả)

Không đưa vào hàng đợi:

1. Sửa Abstract / Bảng 5 từ số S1 (làm trên laptop sau khi `git pull` CSV).
2. Mở GitHub / Zenodo.
3. Ghép 35 hàng DH² ∩ pyKT (1,093,755 vs 1,093,720) — ngoài đợt này.
4. Compile PDF.

---

## Đẩy số về laptop

```bash
git status
git add \
  results/tables/a2_testonly_rescore.csv \
  results/tables/a2_testonly_rescore_summary.json \
  results/tables/rescore_testonly_2026-09-08.log \
  results/tables/a3_dt_repeat_dist.json \
  results/tables/a3_dt_repeat_diag.log
# không add checkpoints .pt, pykt_work_clean, jsonl Ollama
git commit -m "Add A2/A7 test-only rescore and A3 Δt-at-repeats diagnosis."
git push origin gpu-rescore-testonly-2026-09-08
```

Checklist nghiệm thu:

1. Không đụng `qkc_capacity_matched_*`, `a2_multiseed_matrix.csv`.
2. Headline vẫn mask-repeats + chunked + L=400.
3. Không có lần gọi train trong log S1/S2.
4. Seed 42 QKC-T khớp 0.829593 ±5e-5, hoặc có `WARN` giải thích.
