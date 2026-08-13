# Hướng dẫn chấm — Rater A

Cảm ơn bạn tham gia đánh giá. Đây là gói **độc lập**: vui lòng **không thảo luận điểm** với người khác trước khi nộp.

## Bạn nhận được
1. File này (`HUONG_DAN_RATER_A.md`)
2. File chấm: `human_eval_rater_A.csv` (40 dòng)
3. (Nếu có) `RUBRIC_CHUNG.md`

## Việc cần làm (~45–90 phút)

1. Mở `human_eval_rater_A.csv` bằng Excel / Google Sheets / LibreOffice.
2. Ở **mọi dòng**, điền:
   - `rater_id` → viết **cùng một** mã của bạn, ví dụ `TN` hoặc `A1` (chữ cái viết tắt tên).
   - `faithfulness_1to5` → số **1, 2, 3, 4 hoặc 5**
   - `usefulness_1to5` → số **1, 2, 3, 4 hoặc 5**
   - `notes` → để trống nếu không cần; ghi ngắn nếu có lý do đặc biệt
3. **Không sửa** các cột khác (`sample_id`, xác suất, giải thích, hint, …).
4. Lưu file (giữ định dạng CSV nếu được).

## Cách đọc từng mẫu (thứ tự)

Với mỗi `sample_id`:

1. Xem `predicted_correct_prob` (vd. `0.67` ≈ 67%).
2. Đọc `recent_history_summary` (lịch sử gần đây đúng/sai).
3. Đọc `diagnostician_explanation` → chấm **faithfulness**.
4. Đọc `hint_text` → chấm **usefulness**.

## Thang điểm nhanh

| | 1 | 3 | 5 |
|---|---|---|---|
| Faithfulness | Sai số / bịa | Khớp một phần | Khớp đầy đủ |
| Usefulness | Không dùng được | Hơi hữu ích | Dùng ngay được |

Chi tiết: xem `RUBRIC_CHUNG.md`.

## Nộp bài
Gửi lại file CSV đã điền đủ 40 dòng (có thể đổi tên thành `human_eval_rater_A_done.csv`).

**Checklist trước khi gửi**
- [ ] 40 dòng đều có `rater_id`
- [ ] `faithfulness_1to5` và `usefulness_1to5` chỉ chứa 1–5
- [ ] Không xoá / đổi `sample_id`
