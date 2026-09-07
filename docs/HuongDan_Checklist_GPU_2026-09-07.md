# Hướng dẫn thực hiện checklist còn lại — GPU server (07/09/2026)

Đọc cùng: `docs/Checklist_Gate0_review_Hau_2026-09-07.md` (chữ đã xong trên laptop) và `docs/gpu-remaining-tasks.md` (cấm chạy lại).

**Mục tiêu đợt này:** máy chỉ làm 4 việc còn `[ ]` trong checklist + ghi CSV/JSON. Không đụng headline attribution.

Nhánh: **`main`**. Seed: `{42, 17, 1234, 0, 2024}`. Một RTX 3090.

---

## 0. Kéo code (làm trước mọi lệnh)

```bash
cd /path/to/DH2A-KT
git fetch origin
git checkout main
git pull --ff-only
git log -1 --oneline
# kỳ vọng commit có message Gate 0 / GPU checklist
```

Python: `export DH2A_PYTHON=python` nếu cần. Log: `results/tables/gpu_checklist_2026-09-07.log`.

---

## Cấm chạy lại (đã có số trong bài)

| Việc | Script |
|---|---|
| Capacity-matched Q←KC + Zero+Δt | `scripts/34_qkc_capacity_matched.py` |
| A2 5-seed: QKC-T, Q←KC, twin, GIKT, AKT, simpleKT, DKT, DGEKT | `scripts/27_a2_multiseed_matrix.py` |
| A7 5-seed `both` / `lstm` / `query` | `scripts/31_a7_query_dt_ablation.py` |
| ECE 3 nhánh DH² seed 42 | `scripts/30_a6_calibration_metrics.py` |
| B12 rewire/permute | `scripts/32_b12_structure_checks.py` |
| A3 attempt-collapse / đổi giao thức chuẩn | không chạy |
| Planned-gap A7 | không chạy (bài đã ghi không đánh giá) |

Không `--force` các script trên. Không sửa tay `paper/tables/table_qkc_*.tex` / `table_auc_xes3g5m.tex`.

---

## Hàng đợi — thứ tự

Làm **S1 → S2 trên CPU** (vài phút nếu npz đã có), rồi **S3 → S4 trên GPU**. S5 chỉ khi còn checkpoint P0.

### S1 — ECE / Brier / NLL pyKT (CPU, không train)

Checkpoint A2 đã train. Tìm npz (tên thư mục có thể có `_s42`):

```bash
ls results/pykt_work_clean/xes3g5m/fold_0/C_gikt_clean_L400_e30b16*/test_predictions.npz
ls results/pykt_work_clean/xes3g5m/fold_0/C_akt_clean_L400*/test_predictions.npz
ls results/pykt_work_clean/xes3g5m/fold_0/C_simplekt_clean_L400*/test_predictions.npz
```

Ưu tiên **seed 42** (cùng checkpoint calibration DH²). Thay PATH cho khớp máy:

```bash
python scripts/35_pykt_calibration_from_npz.py \
  --npz "GIKT=results/pykt_work_clean/xes3g5m/fold_0/C_gikt_clean_L400_e30b16_s42/test_predictions.npz" \
  --npz "AKT=results/pykt_work_clean/xes3g5m/fold_0/C_akt_clean_L400_s42/test_predictions.npz" \
  --npz "simpleKT=results/pykt_work_clean/xes3g5m/fold_0/C_simplekt_clean_L400_s42/test_predictions.npz" \
  --output results/tables/a6_calibration_pykt_xes3g5m_fold0.csv
```

**Xong khi:** CSV có 3 hàng; `n_scored` ≈ 1,093,720. Nếu ECE > 0.05: ghi vào log, **không** temperature-scale. Không có npz → `SKIP: npz missing` + đường dẫn đã tìm, không train lại baseline.

### S2 — Bootstrap trên giao học viên (CPU, không train)

Script hiện **ghép theo learner id**, không phải 35 hàng. Chạy để có CI trên cùng tập học viên; ghi `n` mỗi file vào log.

```bash
python scripts/25_bootstrap_protocol_ci.py \
  --a "QKC-T=PATH_DH2_NPZ" \
  --b "GIKT=PATH_GIKT_NPZ" \
  --n-boot 1000 --seed 42 \
  --output results/tables/b10_bootstrap_qkct_vs_gikt.csv

python scripts/25_bootstrap_protocol_ci.py \
  --a "QKC-T=PATH_DH2_NPZ" \
  --b "AKT=PATH_AKT_NPZ" \
  --n-boot 1000 --seed 42 \
  --output results/tables/b10_bootstrap_qkct_vs_akt.csv

python scripts/25_bootstrap_protocol_ci.py \
  --a "QKC-T=PATH_DH2_NPZ" \
  --b "simpleKT=PATH_SIMPLEKT_NPZ" \
  --n-boot 1000 --seed 42 \
  --output results/tables/b10_bootstrap_qkct_vs_simplekt.csv
```

NPZ DH²: nếu chưa có file `ts/ps/us`, chấm lại **không train** bằng `scripts/30_a6_calibration_metrics.py` (cần `.pt` seed 42) rồi lưu npz từ log; hoặc dùng dump đã có dưới `results/predictions/`.

**Xong khi:** 3 CSV; log in `n=` hai bên. Nếu lệch 35 hàng, ghi đúng số, **không** bịa giao vị trí. Không overwrite `qkc_capacity_matched_*`.

### S3 — P4 5-seed (GPU) — quên hàm mũ vs Linear Δt

Cùng cờ lần chạy seed 42 (`p_stage.log`): `--concept-forget`, **không** `--time-gap`, Q←KC, clean L=400. Twin đã có: hàng QKC-T / Linear trong `a2_multiseed_matrix.csv` — **không train lại twin**.

```bash
SEEDS="42 17 1234 0 2024"
for s in $SEEDS; do
  python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda \
    --architecture v4 --use-questions --question-graph --no-graph --no-session \
    --concept-forget --mask-repeats --window-mode chunked --max-seq-len 400 \
    --batch-size 16 --epochs 30 --val-frac 0.1 --early-stop-patience 5 \
    --graph-dropout 0 --graph-sensitivity-weight 0 \
    --seed $s --tag p4_forget_s$s
done
```

Ghi val AUC từng seed vào `results/tables/p4_forget_5seed.csv` (cột: seed, forget_val, linear_twin_val từ A2 `p0_dt_on`, delta_val, gate). Twin Linear seed `s` = hàng `p0_dt_on` seed `s` trong `a2_multiseed_matrix.csv`.

**Xong khi:** 5 seed; cổng +0.002; k/5 PASS. Kỳ vọng vẫn FAIL (seed 42 Δval −0.00258). Nếu ≥2 seed PASS: dừng, báo tác giả, không tự sửa Abstract.

### S4 — M4 5-seed (GPU) — teacher grouping ASSIST2012

Protocol gốc (không phải XES L=400): L=200, mask-repeats, question-graph, no-graph. Mỗi seed: **off rồi on**.

```bash
SEEDS="42 17 1234 0 2024"
for s in $SEEDS; do
  python scripts/03_train_tier1.py configs/assist2012.yaml --fold 0 --device cuda \
    --architecture v4 --use-questions --question-graph --no-graph --no-session \
    --mask-repeats --window-mode chunked --max-seq-len 200 \
    --batch-size 16 --epochs 30 --val-frac 0.1 --early-stop-patience 5 \
    --graph-dropout 0 --graph-sensitivity-weight 0 \
    --seed $s --tag m4_group_off_s$s
  python scripts/03_train_tier1.py configs/assist2012.yaml --fold 0 --device cuda \
    --architecture v4 --use-questions --question-graph --no-graph --no-session \
    --group-embed --mask-repeats --window-mode chunked --max-seq-len 200 \
    --batch-size 16 --epochs 30 --val-frac 0.1 --early-stop-patience 5 \
    --graph-dropout 0 --graph-sensitivity-weight 0 \
    --seed $s --tag m4_group_on_s$s
done
```

Seed 42 đã có (`m4_group.log`: Δval +0.00142). Có thể `--seed 42` để xác nhận khớp ±0.0002; nếu khớp thì 4 seed còn lại đủ.

Ghi `results/tables/m4_group_5seed.csv`.

**Xong khi:** 5 cặp; k/5 PASS. M4 gần cổng — nếu lật ≥2 seed PASS: báo tác giả.

### S5 — B9 rescore simpleKT P0 L=200 (không train)

Chỉ khi còn checkpoint/tag pyKT L=200 protocol cũ. Tìm:

```bash
find results external/p0_leakage_audit/results -iname '*simplekt*' | head
```

Chấm trên export clean (mask-repeats, chunked, L=400) **không train lại**. `scripts/23_eval_clean_protocol.py` **không** nhận checkpoint pyKT.

Không có file → `SKIP: checkpoint missing` trong log. Không train simpleKT L=200 clean (đó là P3.1, ngoài checklist này).

---

## Việc không phải GPU (laptop / tác giả)

Không đưa vào hàng đợi máy:

1. Mở `https://github.com/edu-risk-lab/DH2A-KT` hoặc kho ẩn danh sống (hiện 404).
2. Sau commit này: sửa Data availability pin từ `ca8bc2b` sang `git rev-parse --short HEAD`.
3. Zenodo DOI lúc nộp.
4. Compile `paper/main.pdf`, lướt hidelinks.
5. Không sửa Abstract bằng số S3/S4 trừ khi tác giả chốt (sáu FAIL một seed đã ra khỏi Abstract).

---

## Đẩy số về laptop

```bash
git status
# add chỉ CSV/JSON mới dưới results/tables/ (allowlist)
# không add checkpoints .pt, jsonl Ollama, pykt_work_clean dumps nặng
```

Checklist nghiệm thu:

1. Không đụng `qkc_capacity_matched_*`, `a2_multiseed_matrix.csv` (trừ khi chỉ *đọc* twin P4).
2. Headline vẫn mask-repeats + chunked + L=400 trên XES.
3. Một dòng mới trong `docs/paper-artifacts.md` cho mỗi CSV mới.

Khi xong S1–S4: commit message gợi ý: `Add pyKT calibration, learner bootstrap CSVs, and P4/M4 five-seed tables.`
