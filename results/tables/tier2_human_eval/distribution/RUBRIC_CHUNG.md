# Rubric chấm điểm (chung cho mọi rater)

## Nhiệm vụ
Đánh giá chất lượng giải thích của hệ thống AgentDG-KT (Tier 2):
- **Faithfulness**: giải thích có khớp với xác suất Tier-1 và lịch sử làm bài không?
- **Usefulness**: gợi ý / hint có hữu ích cho giáo viên không?

## Thang Likert 1–5 (chỉ số nguyên)

### Faithfulness (đối chiếu `predicted_correct_prob` + `recent_history_summary` với `diagnostician_explanation`)

| Điểm | Ý nghĩa |
|-----:|---------|
| 1 | Mâu thuẫn số P(correct), bịa thêm sự kiện không có trong history |
| 2 | Lệch lớn hoặc claim không được history hỗ trợ |
| 3 | Khớp một phần; có phóng đại / thiếu sót |
| 4 | Khớp tốt, chỉ lỗi diễn đạt nhỏ |
| 5 | Khớp đầy đủ số liệu và history |

### Usefulness (đánh giá `hint_text` góc nhìn giáo viên)

| Điểm | Ý nghĩa |
|-----:|---------|
| 1 | Không dùng được / gây hiểu nhầm |
| 2 | Mơ hồ, hầu như không actionable |
| 3 | Hơi hữu ích |
| 4 | Rõ, phần lớn dùng được |
| 5 | Gợi ý bước tiếp theo cụ thể, dùng ngay được |

## Lưu ý
- Critic đã “OK” không có nghĩa giải thích tự động được 5 điểm — bạn vẫn phải đọc.
- Không cần biết học sinh thật là ai; ID chỉ để tham chiếu.
- `notes` chỉ ghi khi cần (ví dụ: “nói 67.5% nhưng ý khác với history”).
