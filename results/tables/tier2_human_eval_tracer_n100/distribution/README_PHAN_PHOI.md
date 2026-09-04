# Gói chấm Human Eval — B10 tracer QKC-T (n=100)

Dành cho **người tổ chức**. Phân phối cho **≥3 người chấm độc lập**.
Không gửi chung một file; mỗi người nhận đúng 1 bộ A/B/C.

## Khác pilot n=40 cũ
| | Pilot cũ (`tier2_human_eval/`) | Pack này |
|--|-------------------------------|----------|
| Checkpoint | v2 graph-only | **QKC-T (tracer khuyến nghị)** |
| Số mẫu | 40 | **100** |
| Số rater tối thiểu | 2 | **3** |
| Thời gian ước lượng | 45–90 phút | **~2–3.5 giờ** |

**Không** ghi đè thư mục pilot cũ.

## Nội dung `distribution/`

| File | Người nhận |
|------|------------|
| `HUONG_DAN_RATER_A.md` + `human_eval_rater_A.csv` | Rater A |
| `HUONG_DAN_RATER_B.md` + `human_eval_rater_B.csv` | Rater B |
| `HUONG_DAN_RATER_C.md` + `human_eval_rater_C.csv` | Rater C |
| `RUBRIC_CHUNG.md` | Đính kèm mọi người |
| `MAU_TIN_NHAN.md` | Copy-paste khi gửi |
| `README_PHAN_PHOI.md` | File này (chỉ tổ chức) |

## Cách phân phối

1. Chọn **3** người quen ngữ cảnh giáo dục / KT (không cần biết code).
2. Gửi mỗi người một cặp: hướng dẫn + CSV + `RUBRIC_CHUNG.md`.
3. Nhắc: **không trao đổi điểm** cho đến khi cả nhóm nộp xong.
4. Có thể chia 2 đợt (50+50) nếu cần giảm mệt — giữ nguyên `sample_id`.
5. Thu về tên `human_eval_rater_X_done.csv` (hoặc giữ tên gốc, miễn đủ cột).

## Thu thập

Đặt file đã chấm vào:

`results/tables/tier2_human_eval_tracer_n100/distribution/returned/`

Rồi chạy:

```bash
python scripts/15_aggregate_human_eval.py ^
  --returned-dir results/tables/tier2_human_eval_tracer_n100/distribution/returned ^
  --out-dir results/tables/tier2_human_eval_tracer_n100
```

Sau đó cập nhật `paper/tables/table_human_eval.tex` và đoạn Tier 2 trong `paper/main.tex`.

## Quy tắc tổ chức
- Cùng 100 `sample_id` (seed=42) cho mọi rater.
- Chỉ điền 4 cột cuối: `rater_id`, `faithfulness_1to5`, `usefulness_1to5`, `notes`.
- Không sửa explanation / hint / xác suất.
- 9 dòng `critic_flagged=True` là bình thường — vẫn chấm Likert độc lập.
