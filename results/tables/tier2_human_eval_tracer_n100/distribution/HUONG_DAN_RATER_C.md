# Hướng dẫn chấm — Rater C (B10, n=100)

Cảm ơn bạn tham gia. Đây là gói **độc lập**: vui lòng **không thảo luận điểm**
với người khác trước khi nộp.

## Bạn nhận được
1. File này (`HUONG_DAN_RATER_C.md`)
2. File chấm: `human_eval_rater_C.csv` (**100** dòng)
3. `RUBRIC_CHUNG.md`

## Việc cần làm (~2–3.5 giờ; có thể chia 2 buổi)

1. Mở `human_eval_rater_C.csv` bằng Excel / Google Sheets / LibreOffice.
2. Ở **mọi dòng**, điền:
   - `rater_id` → **cùng một** mã của bạn (vd. `DN`, `C1`)
   - `faithfulness_1to5` → **1–5** (số nguyên)
   - `usefulness_1to5` → **1–5** (số nguyên)
   - `notes` → tuỳ chọn
3. **Không sửa** các cột khác.
4. Lưu CSV; nộp với tên `human_eval_rater_C_done.csv` nếu tiện.

## Cách đọc từng mẫu
1. `predicted_correct_prob` (vd. `0.60` ≈ 60%)
2. `recent_history_summary`
3. `diagnostician_explanation` → chấm **faithfulness**
4. `hint_text` → chấm **usefulness**
5. `critic_flagged` chỉ tham khảo — không copy thành điểm

## Thang nhanh

| | 1 | 3 | 5 |
|---|---|---|---|
| Faithfulness | Sai số / bịa | Khớp một phần | Khớp đầy đủ |
| Usefulness | Không dùng được | Hơi hữu ích | Dùng ngay được |

## Checklist trước khi gửi
- [ ] 100 dòng đều có `rater_id`
- [ ] Hai cột điểm chỉ chứa 1–5
- [ ] Không đổi / xoá `sample_id`
