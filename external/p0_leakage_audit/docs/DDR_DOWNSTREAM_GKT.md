# Playbook: DDR → downstream stress test cho GKT

Mục tiêu reviewer **M4**: bổ sung ít nhất **một backbone graph-KT thứ hai** (GKT) cho C2 — liên kết **DDR (cấu trúc)** với **AUC downstream** sau retrain, song song với sweep DGEKT đã có trong paper (§4.7, Table `tab:ddr-downstream`, Figure S3).

**Trạng thái repo:** `scripts/ddr_downstream.py` hỗ trợ `--model gkt` (mặc định). `results/tables/ddr_downstream.csv` hiện chỉ có **60 dòng DGEKT** (ASSIST + XES3G5M). Chưa có dòng `model=gkt`.

---

## 0. Thiết kế thí nghiệm (khớp paper DGEKT)

| Thành phần | Giá trị (giữ parity với DGEKT run) |
|------------|-------------------------------------|
| Dataset | **ASSISTments 2012**, **XES3G5M** (bỏ Junyi — GKT impractical) |
| Folds | 3 (learner CV, seed base 42) |
| Operators | `edge_drop`, `node_drop`, `prereq_preserve` |
| Strengths `p` | 0.10, 0.20, 0.30 |
| Baseline graph | `operator=none`, `p=0.0`, DDR=0 |
| Model | **`gkt`** (pyKT graph branch) |
| Hyperparams | Đọc từ `configs/<ds>.yaml` → `baselines[name=gkt].hyperparams` |
| | ASSIST: batch **32**, 10 epochs |
| | XES3G5M seed 42: batch **4**, 10 epochs (primary budget) |
| | XES3G5M multiseed (17/1234, Jul 2026): batch **8**, 10 epochs |
| Output | Append/resume CSV; key `(dataset, model, fold, operator, p)` |

**Số run huấn luyện:** 2 dataset × 3 fold × (1 baseline + 3×3 perturb) = **60** forward passes GKT.

**Không dùng GIKT:** script từ chối — GIKT dùng bipartite Q–C graph, không đọc `E_pre` KC–KC.

---

## 1. Chuẩn bị GPU server

Giống [Q1_GPU_EXPERIMENTS.md](Q1_GPU_EXPERIMENTS.md) §0:

```bash
git clone https://github.com/edu-risk-lab/leakage-controlled-kt-audit.git
cd leakage-controlled-kt-audit && git pull
git submodule update --init --recursive

python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt && pip install -e . && pip install -e ".[pykt]"
pip install torch --index-url https://download.pytorch.org/whl/cu121   # chỉnh theo driver

python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Dữ liệu bắt buộc:**

- `data/processed/assist2012.parquet`, `data/processed/xes3g5m.parquet`
- Graph train-only per fold:
  - `data/processed/assist2012/fold_{0,1,2}/e_pre_train_only.csv`
  - `data/processed/xes3g5m/fold_{0,1,2}/e_pre_train_only.csv`
  - (và `e_sim_train_only.csv` nếu có — XES3G5M có E_sim)

Nếu thiếu graph:

```bash
python -m src.graph_builder --config configs/assist2012.yaml
python -m src.graph_builder --config configs/xes3g5m.yaml
```

---

## 2. Smoke test (không GPU / không train)

Xác minh đường dẫn graph, DDR, npz:

```bash
python -m scripts.ddr_downstream \
  --config configs/xes3g5m.yaml \
  --model gkt \
  --max-folds 1 \
  --dry-run \
  --out results/q1/ddr_downstream_gkt/smoke_dryrun.csv
```

Kỳ vọng: 10 dòng/fold (1 none + 9 perturb), cột `status=dry_run`, `auc=NaN`.

---

## 3. Calibrate 1 fold (ước lượng thời gian)

```bash
mkdir -p results/q1/ddr_downstream_gkt

python -m scripts.ddr_downstream \
  --config configs/xes3g5m.yaml \
  --model gkt \
  --max-folds 1 \
  --out results/q1/ddr_downstream_gkt/ddr_downstream_gkt.csv \
  2>&1 | tee results/q1/ddr_downstream_gkt/xes_fold0_calib.log
```

- GKT trên XES3G5M (batch **4** primary / **8** multiseed): thường **~8–20 phút/run** trên GPU 24GB (phụ thuộc GPU và early stopping).
- 10 variant/fold → **~1.5–3 h/dataset-fold**.
- Full 60 run GKT: **~12–24 h** (chậm hơn DGEKT đáng kể).

Script **resume-safe**: chạy lại cùng lệnh sẽ skip các key đã có trong CSV.

---

## 4. Full sweep (production)

### Cách A — script một lệnh (khuyến nghị)

```bash
bash scripts/run_ddr_downstream_gkt.sh
```

### Cách B — từng dataset

```bash
OUT=results/q1/ddr_downstream_gkt/ddr_downstream_gkt.csv

python -m scripts.ddr_downstream \
  --config configs/assist2012.yaml \
  --model gkt \
  --operators edge_drop node_drop prereq_preserve \
  --ps 0.10 0.20 0.30 \
  --experiment-seed 42 \
  --perturb-seed 42 \
  --out "$OUT"

python -m scripts.ddr_downstream \
  --config configs/xes3g5m.yaml \
  --model gkt \
  --operators edge_drop node_drop prereq_preserve \
  --ps 0.10 0.20 0.30 \
  --experiment-seed 42 \
  --perturb-seed 42 \
  --out "$OUT"
```

**Không cần config yaml mới** — dùng `configs/assist2012.yaml` và `configs/xes3g5m.yaml` (hyperparams GKT đã khai báo trong `baselines`).

**Work dir trung gian:** `results/pykt_work/<dataset>/fold_<k>_seed_<s>/ddr_downstream/gkt/` (sequence CSV + graph npz per perturbation).

---

## 5. Merge vào CSV paper + plot

Sau khi 60 dòng GKT xong:

```bash
python scripts/merge_ddr_downstream.py \
  --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt.csv

python scripts/plot_ddr_downstream.py
```

`merge_ddr_downstream.py` gộp vào `results/tables/ddr_downstream.csv` (giữ DGEKT cũ, thêm GKT, dedupe theo key).

Kiểm tra:

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/tables/ddr_downstream.csv')
print(df.groupby(['dataset','model']).size())
assert (df.model=='gkt').sum() == 60
"
```

Artefacts:

| File | Mô tả |
|------|--------|
| `results/tables/ddr_downstream.csv` | Raw (DGEKT + GKT) |
| `results/tables/ddr_downstream_summary.csv` | Mean DDR / AUC / AUC drop |
| `results/tables/ddr_downstream.tex` | Table main (cả hai model) |
| `results/figures/fig_ddr_downstream.pdf` | Scatter DDR vs AUC drop |

Regenerate ANOVA supplement (hiện hardcode caption DGEKT — cần sửa sau khi có GKT):

```bash
python scripts/generate_phase_c_tables.py   # hoặc pipeline paper artifacts
```

---

## 6. Pull về máy local / commit

Trên server:

```bash
tar czf ddr_gkt_results.tgz \
  results/q1/ddr_downstream_gkt/ \
  results/tables/ddr_downstream.csv \
  results/tables/ddr_downstream_summary.csv \
  results/tables/ddr_downstream.tex \
  results/figures/fig_ddr_downstream.pdf
```

Local:

```bash
scp user@gpu:~/p0_project/ddr_gkt_results.tgz .
tar xzf ddr_gkt_results.tgz -C p0_project/
```

Commit (khi sẵn sàng): `results/tables/ddr_downstream*.csv/tex`, `results/figures/fig_ddr_downstream.pdf`, cập nhật `main_APIN.tex` §4.7.

---

## 7. Cập nhật paper (sau khi có số liệu)

### §4.7 `sec:exp-ddr-downstream`

- Đổi "DGEKT-only" → **"GKT + DGEKT"** (hoặc "two graph consumers").
- Báo cáo correlation **riêng từng model** (plot script đã tách `assist2012/gkt` vs `xes3g5m/gkt`).
- Nếu GKT vẫn null sensitivity: củng cố C2 ở mức **structural**; nếu GKT sensitive hơn DGEKT: điểm mới quan trọng.

### Table S20 (`anova_ddr_downstream.tex`)

- Chạy ANOVA filter `model==gkt` hoặc thêm cột Model; sửa caption không ghi "DGEKT-only".

### Figure S3

- Scatter sẽ có thêm điểm GKT (màu theo operator); cân nhắc facet theo model nếu quá dày.

---

## 8. Kịch bản kết quả & diễn giải

| Kết quả GKT | Ý nghĩa cho reviewer M4 |
|-------------|-------------------------|
| `r` ≈ 0, AUC drop ≈ 0 (giống DGEKT) | DDR vẫn là metric **cấu trúc**; predictive link không universal — nhưng đã có **2 backbones** |
| `r` > 0.3, AUC drop tăng theo DDR | C2 có giá trị applied; GKT nhạy cấu trúc hơn DGEKT |
| `prereq_preserve` drop < `node_drop` @ cùng p | Khớp narrative DDR operator (như §4.6) — báo cáo per-model |

**Headline cần trích (ví dụ):**

```
assist2012/gkt: Pearson r=..., p=...
xes3g5m/gkt:    Pearson r=..., p=...
```

So sánh với DGEKT hiện tại: r=0.13–0.15, p≈0.5 (null).

---

## 9. Troubleshooting

| Triệu chứng | Xử lý |
|-------------|--------|
| `No module named torch` | Cài pyKT stack trên GPU |
| `e_pre_train_only.csv` missing | Chạy `graph_builder` |
| CUDA OOM (GKT XES) | Giảm batch trong yaml: `baselines[gkt].hyperparams.batch_size: 8` (ghi chú trong paper) |
| Run bị ngắt giữa chừng | Chạy lại cùng lệnh — skip keys done |
| AUC GKT baseline lệch ~0.71 | Graph stale / sai seed — rebuild graph (xem Q1 doc §0.2) |
| CSV trùng dòng | `merge_ddr_downstream.py` dedupe; hoặc xóa dòng GKT lỗi rồi rerun |

---

## 10. Checklist nhanh

- [ ] Graph ASSIST + XES3G5M fold 0–2
- [ ] Dry-run OK
- [ ] Calibrate 1 fold XES, ghi thời gian/run
- [ ] Full 60 GKT runs → `results/q1/ddr_downstream_gkt/ddr_downstream_gkt.csv`
- [ ] Merge + `plot_ddr_downstream.py`
- [ ] Verify 60 gkt + 60 dgekt rows
- [ ] Update §4.7 + S20 caption + (optional) facet figure
- [ ] Push / PR

---

## Tham chiếu code

- Runner: [`scripts/ddr_downstream.py`](../scripts/ddr_downstream.py) — `--model gkt`
- Plot: [`scripts/plot_ddr_downstream.py`](../scripts/plot_ddr_downstream.py)
- Shell: [`scripts/run_ddr_downstream_gkt.sh`](../scripts/run_ddr_downstream_gkt.sh)
- Merge: [`scripts/merge_ddr_downstream.py`](../scripts/merge_ddr_downstream.py)
