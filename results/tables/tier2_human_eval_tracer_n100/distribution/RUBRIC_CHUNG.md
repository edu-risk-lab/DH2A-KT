# Rubric chung — Likert 1–5 (B10, tracer QKC-T)

Chấm **hai** thang độc lập trên mỗi mẫu. Chỉ dùng số nguyên 1–5.

## Faithfulness (độ trung thực với mô hình)

So khớp `diagnostician_explanation` với:

- `predicted_correct_prob` (vd. 0.67 ≈ 67%), và
- `recent_history_summary` (đúng/sai gần đây).

| Điểm | Ý nghĩa |
|------|---------|
| **1** | Mâu thuẫn xác suất / bịa lịch sử hoặc sự kiện không có trong history |
| **2** | Lệch lớn (sai số đáng kể, claim không có căn cứ) |
| **3** | Khớp một phần; có phóng đại hoặc bỏ sót quan trọng |
| **4** | Khớp tốt; chỉ lệch nhỏ về wording |
| **5** | Khớp đầy đủ số liệu + history; không bịa |

## Usefulness (hữu ích cho giáo viên / tutor)

Dựa chủ yếu vào `hint_text` (có thể đọc kèm explanation).

| Điểm | Ý nghĩa |
|------|---------|
| **1** | Không dùng được / gây hiểu nhầm |
| **2** | Mơ hồ, generic, khó hành động |
| **3** | Hơi hữu ích nhưng còn chung chung |
| **4** | Rõ, phần lớn hành động được |
| **5** | Có bước tiếp theo cụ thể, dùng ngay được |

## Lưu ý nhanh
- Không cần biết code hay kiến trúc mô hình.
- Cột `critic_flagged` chỉ để tham khảo — **không** copy thành điểm Likert.
- Nếu explanation khớp số nhưng hint vô dụng → faithfulness cao, usefulness thấp (và ngược lại).
