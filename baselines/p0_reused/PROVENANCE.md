# Provenance — Baseline reused from P0

Các file trong thư mục này là **bản sao trực tiếp, không chỉnh sửa**, của kết quả thực nghiệm baseline đã công bố trong bài P0:

> Dao Minh Tuan et al., *Leakage-Controlled Concept Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing*, APIN submission.

## Nguồn

| File ở đây | File gốc trong P0 |
|---|---|
| `baseline_results.csv` | `results/tables/baseline_results.csv` |
| `baseline_fold_results.csv` | `results/tables/baseline_fold_results.csv` |

- Repo gốc: `https://github.com/edu-risk-lab/leakage-controlled-kt-audit`
- Commit đã vendor: `5e3d6a01c32a649f1655ac8712f03359bddfe478` (xem `external/p0_leakage_audit/.p0_vendored_commit.txt`)
- Thời điểm vendor: 2026-08-02T15:36:04Z

## Nội dung

- `baseline_results.csv` — 69 dòng, tổng hợp AUC/ACC/NLL (mean ± CI, 3-fold) cho các model **bkt, akt, dkt, simplekt, gkt, gikt, skt, dygkt, dgekt** trên 5 dataset: **assist2012, junyi, xes3g5m** (3 benchmark công khai chính) + **synthetic_c2, synthetic_c5** (sanity). Mỗi model có cả hai điều kiện `graph_construction ∈ {train_only, full_log}`.
- `baseline_fold_results.csv` — 205 dòng, chi tiết theo từng fold (0/1/2, seed 42/43/44) — dùng khi cần kiểm định thống kê theo fold (paired t-test/Wilcoxon) thay vì chỉ dùng số mean đã gộp.

## Cách dùng lại (không train lại trên RTX 3090)

Dùng `dh2a_kt.baselines.loader` để nạp trực tiếp các file này làm cột so sánh cho RQ1 (Tier 1 DH²-KT vs baseline). **Chỉ hợp lệ nếu** DH2A_KT dùng đúng preprocessing/split của P0 (xem `external/p0_leakage_audit/README.md` mục 3 "Data download and preparation" và mục 4 "Running experiments") — nếu tự tiền xử lý lại dataset theo cách khác, các số này **không còn so sánh được trực tiếp** và phải nêu rõ là "quan sát" (observational) trong bài, đúng cách P0 tự giới hạn phát biểu của họ (Section 5.3 của P0).

## Giới hạn đã biết

- Baseline này chỉ phủ quan hệ graph **cặp đôi KC–KC** (train_only/full_log của P0), **không** bao gồm bất kỳ biến thể hypergraph/dị thể nào — vì P0 không có kiến trúc đó. DH²-KT (Tier 1 của D) vẫn phải tự train và so với các số này.
- `status=pending_data` (dòng `bkt` đầu tiên trong `baseline_results.csv` không có dataset) là artefact export gốc của P0, giữ nguyên để tránh sửa dữ liệu đã công bố — bỏ qua dòng này khi phân tích.
