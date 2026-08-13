# Gói chấm Human Eval — AgentDG-KT (C-Human)

Phân phối cho **2–3 người chấm độc lập**. Không gửi chung một file; mỗi người nhận đúng 1 bộ.

## Nội dung thư mục `distribution/`

| File | Người nhận |
|------|------------|
| `HUONG_DAN_RATER_A.md` + `human_eval_rater_A.csv` | Rater A |
| `HUONG_DAN_RATER_B.md` + `human_eval_rater_B.csv` | Rater B |
| `HUONG_DAN_RATER_C.md` + `human_eval_rater_C.csv` | Rater C (tuỳ chọn nếu đủ 3 người) |
| `RUBRIC_CHUNG.md` | Đính kèm mọi người (thang điểm) |
| `README_PHAN_PHOI.md` | File này — chỉ dành cho người tổ chức |

## Cách phân phối (bạn làm)

1. Chọn 2 hoặc 3 người quen với ngữ cảnh giáo dục / KT (không cần biết code).
2. Gửi mỗi người **một cặp**:
   - file hướng dẫn `HUONG_DAN_RATER_X.md`
   - file CSV `human_eval_rater_X.csv`
   - (tuỳ chọn) `RUBRIC_CHUNG.md`
3. Nhắc: **không trao đổi điểm** cho đến khi cả nhóm nộp xong.
4. Thời gian ước lượng: **45–90 phút** cho 40 mẫu.
5. Thu về đúng tên file (hoặc đổi tên `..._rater_A_done.csv` cũng được, miễn giữ cột).

## Thu thập & gửi lại cho AI / paper

Sau khi nhận đủ file đã chấm, đặt vào:
`results/tables/tier2_human_eval/distribution/returned/`

Rồi báo lại để gộp + tính mean/SD và agreement (Spearman/ICC).

## Quy tắc tổ chức

- Cùng 40 `sample_id` cho mọi rater (đã cố định seed=42).
- Chỉ điền 4 cột cuối: `rater_id`, `faithfulness_1to5`, `usefulness_1to5`, `notes`.
- Không sửa nội dung giải thích / hint / xác suất.
- Nếu chỉ có 2 người: bỏ Rater C.
