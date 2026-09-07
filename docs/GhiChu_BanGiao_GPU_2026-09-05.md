# Ghi chú bàn giao — tiếp tục xử lý DH²A-KT sau review vòng 2 (ver3)

**Ngày:** 05/09/2026
**Nguồn:** `docs/NhanXet_DH2A-KT_ver3_Tuan_05Sep2026.pdf` (PGS.TS. Nguyễn Văn Hậu)
**Kế hoạch đầy đủ:** `docs/KeHoach_CaiTien_BanThao_ver3_2026-09-05.md` (đã có trong kho)

Ghi chú này dành cho bất kỳ ai (người hoặc agent) tiếp tục chạy GPU/sửa bản thảo sau phiên làm việc này, để không lặp lại việc đã làm và không bỏ sót một cảnh báo quan trọng về phân nhánh git bên dưới.

---

## ⚠️ CẢNH BÁO QUAN TRỌNG NHẤT — đọc trước khi làm gì khác

Các sửa Gate 0 mô tả bên dưới được thực hiện trên **bản `main.tex`/`references.bib`/`table_auc_xes3g5m.tex` đúng như phiên bản ver3 đã gửi cho GS Hậu** — phiên bản này **KHÔNG chứa** các nội dung đã có sẵn trên nhánh git `qkc-zero-time` (đã merge PR #5), gồm:

- Tiêu đề/Abstract kiểu "Tier 1 / Tier 2" (khác hẳn tiêu đề ver3 "Leakage-Audited Knowledge Attribution for Knowledge Tracing")
- Kết quả zero-incidence Linear Δt (PASS 5/5)
- Kết quả capacity-matched QKC (FAIL, đã sửa lại tuyên bố ghi công incidence)
- B12 rewiring/label-permute cho fold 1–2
- B10 tracer Likert (n=100, 3 raters) — **lưu ý: số hiệu "B10" trong các commit này rất có thể thuộc numbering của review VÒNG TRƯỚC (ver2), KHÔNG phải B10 trong review ver3 hiện tại (ver3's B10 là việc lấy giao tập ID đánh giá — một việc khác hẳn). Đừng giả định hai "B10" là cùng một việc — cần đối chiếu lại nội dung, không chỉ số hiệu.**

**Do đó, các file `main.tex`/`references.bib`/`table_auc_xes3g5m.tex` đã sửa (giao qua chat trong phiên này, KHÔNG commit vào git theo yêu cầu của tác giả) chưa được hợp nhất (merge) với nội dung trên nhánh `qkc-zero-time`.** Trước khi tiếp tục bất kỳ việc gì ở Gate 1/2/3 bên dưới:

1. Đọc kỹ nội dung hiện tại của `main.tex` trên nhánh `qkc-zero-time` (không phải bản ver3) — rất có thể một số việc trong Gate 1/2 dưới đây đã được thực hiện một phần ở đó (đặc biệt các mục liên quan Δt/incidence/rewiring).
2. Đối chiếu từng điểm A/B/C trong `docs/NhanXet_DH2A-KT_ver3_Tuan_05Sep2026.pdf` với nội dung nhánh `qkc-zero-time`, KHÔNG giả định dựa trên tên nhánh hay số hiệu commit cũ.
3. Việc hợp nhất bản ver3+Gate-0-fixes với nhánh `qkc-zero-time` cần một người (tác giả) quyết định — đây là việc chưa làm, xem mục "Việc còn treo" bên dưới.
4. Kho GitHub `https://github.com/edu-risk-lab/DH2A-KT` hiện trả về 404 (chưa public). Tác giả xác nhận: **sẽ public khi nộp bài (submission)**, trước đó phản biện có thể xin quyền truy cập qua tác giả liên hệ. Câu Data Availability trong `main.tex` (bản ver3+fixes) đã được sửa theo đúng tinh thần này.

---

## 1. Gate 0 — đã sửa xong trong bản ver3 (chưa merge vào qkc-zero-time)

Các điểm sau đã sửa trực tiếp vào bản `main.tex`/`references.bib`/`table_auc_xes3g5m.tex` (dựa trên ver3), biên dịch sạch (41 trang, không lỗi, không cảnh báo overfull mới), đã giao cho tác giả qua chat:

- **A1** — sửa câu §7.1 nói ngược hướng simpleKT vs. twin (twin=0.8202 < simpleKT=0.8214, không phải ngược lại)
- **A6** — thêm trích dẫn pyKT (Liu et al., NeurIPS 2022) ở 4 vị trí + đoạn ghi công/phân biệt đóng góp mới
- **B2** — sửa "+0.047" → "+0.051 to +0.056" (nguồn đúng là Bảng 3, không phải dải manipulation-check ΔAUC 0.038/0.043/0.047 trong `docs/kbs-gpu-runbook.md`); sửa "Tables 8–7"→"Tables 7–8"
- **B3** — bỏ mệnh đề sai về đại số tuyến tính ("the mix cannot collapse into e_c")
- **B4** — tách rõ 5,549,635 lượt thử canonical vs. 6,413,353 hàng mở rộng theo KC (XES3G5M)
- **B5** — đổi "pre-registered" → "pre-specified"; thêm quy ngưỡng +0.002 ra bội số SD (≈2.9×SD, SD=0.0007 từ simpleKT 5-seed)
- **B6** — thêm trích dẫn AKT, simpleKT, DGEKT, SKT, BKT
- **B7** — sửa 2 câu Abstract nói FAIL mạnh hơn mức cổng cho phép; thêm chú thích dưới Bảng 7 giải thích ý nghĩa FAIL
- **B8** — thêm phạm vi tra cứu cho các phát biểu phủ định toàn xưng; sửa "agentic closed loop" thành mô tả đúng phạm vi
- **A5 (phần rẻ)** — đổi tên "concept-prerequisite hyperedge" → "path-derived concept hyperedge"; thêm ℓmin=3/ℓmax=8 (lấy từ `configs/xes3g5m.yaml`) vào mô tả Algorithm 1
- **B1** — sửa câu Data Availability: kho sẽ public khi nộp bài; trước đó phản biện xin quyền qua tác giả liên hệ (quyết định của tác giả, 05/09/2026)
- **C1** — thêm `\usepackage[hidelinks]{hyperref}` để bỏ khung viền màu hyperlink trong PDF
- **C2** — sửa metadata tham khảo [9] (FoundationalASSIST: Worden, Heffernan, Heffernan, Sonkar 2026, arXiv:2602.00070), [13] (ACL Anthology), [15] (qua OpenAlex, vì IEEE Xplore không fetch được trực tiếp)

**Còn treo trong Gate 0 (cần tác giả xác nhận, chưa sửa):**
- **A5** — quyết định cuối cùng: giữ tên đổi (rẻ, đã làm) hay sửa cả Algorithm 1 phát danh sách có thứ tự + chạy lại M5 trên toàn bộ Junyi ≥3 seed (đắt, dời sang Gate 3)?
- **B5** — có hiện vật đăng ký thật (OSF/AsPredicted/commit mốc thời gian) cho ngưỡng +0.002 không, hay giữ nguyên "pre-specified" không có hiện vật?
- Câu hỏi #2, #3, #5, #6, #10, #11, #12 của GS Hậu (xem mục 6 của kế hoạch cải tiến) — cần tác giả xác nhận thủ công hoặc đọc thêm code.

---

## 2. Gate 1 — thí nghiệm protocol cần GPU (~1 tuần), làm TRƯỚC Gate 2

**Ưu tiên A3 trước A7** — vì A3 có thể buộc đổi giao thức đánh giá chuẩn; nếu đổi, phải xong trước khi vào Gate 2 (tránh lặp lỗi "sửa rời rạc" đã sinh ra A1).

1. **A3** — Dư lượng +0.0101 ở Table 3 hàng 4 (p0_dt_on): đã xác nhận hàng 3 (v4q_clean) và hàng 4 khác nhau cả ở `window_mode` (last/parity vs. chunked) và số vị trí chấm (599,546 vs 1,093,755) — không chỉ khác Δt. Cần: (a) trích phân bố Δt tại vị trí lặp dưới giao thức raw; (b) chấm tracer khuyến nghị dưới raw loại bỏ Δt=0, so residual với +0.0101; (c) xây bản gộp mức lượt thử (attempt-level collapse); (d) chạy tracer trên 3 giao thức (raw/mask-target/attempt-collapse) cùng 1 fold; (e) nếu attempt-collapse khác biệt đáng kể → dùng làm giao thức chuẩn mới và chạy lại Bảng 5 theo giao thức đó TRƯỚC Gate 2.
2. **A7** — Δt phía query: FoundationalASSIST cho thấy nhánh query-only giữ +0.01471 (LSTM-only chỉ +0.00280). Cần: định nghĩa "attempt-onset next-response prediction" (§4.3); chạy 4 nhánh trên XES3G5M (không thời gian / chỉ Δt lịch sử / chỉ Δt query / cả hai) + nhánh "deployment-realistic" (gap tính tới thời điểm dự đoán); viết §8.1 "Deployment availability of the query-side gap".
3. **B9** — Chấm lại checkpoint simpleKT L=200 của P0 (nếu còn lưu — cần xác nhận với tác giả) dưới clean protocol, không train lại, để có điểm dữ liệu chéo-kiến-trúc đầu tiên.

---

## 3. Gate 2 — MỘT đợt chạy 5-seed thống nhất (~1–2 tuần)

**Không chia nhỏ thành nhiều đợt riêng lẻ** — đây là nguyên nhân chính sinh lỗi A1 ở vòng trước. Dùng đúng bộ seed `{0, 17, 42, 1234, 2024}`. Chỉ bắt đầu sau khi Gate 1 (đặc biệt A3) đã chốt giao thức chuẩn.

1. **A2** — Chạy cả 6 cấu hình Bảng 5 (tracer khuyến nghị, Q←KC, twin, GIKT, AKT, simpleKT) trên cùng 5 seed; báo cáo mean±SD cho mọi ô; bỏ "single canonical run".
2. **A4** — Trong CÙNG đợt: xuất thêm NLL, Brier, ECE (15 bin) cho cả 6 cấu hình; thêm 3 cột vào Bảng 5; thêm reliability diagram; nếu ECE > 0.05 cân nhắc temperature scaling hoặc ghi vào Limitations.
3. **B10** (ver3 numbering — khác B10 trong git log cũ) — Lấy giao tập ID đánh giá giữa DH²-KT (n=1,093,755) và baseline (n=1,093,720, lệch 35 hàng đã xác nhận là hệ thống, không phải nhiễu); chấm lại tất cả trên tập giao; tính lại CI bootstrap.
4. **B11** — Phụ lục bảng siêu tham số đầy đủ cho mọi mô hình; tinh chỉnh AKT ở L=400; nới trần epoch GIKT; simpleKT L=200 clean để tách ảnh hưởng độ dài chuỗi.
5. Nếu quỹ máy cho phép: mở rộng tracer khuyến nghị + GIKT thành 3 fold × 3 seed.

---

## 4. Gate 3 — quyết định phạm vi khoa học (~3–4 ngày viết lại), sau khi có số Gate 1–2

Không viết lại các phần này trước khi có số liệu, để tránh viết lại hai lần:

- **A5** (nếu chọn phương án đắt ở Gate 0) — sửa Algorithm 1 + chạy lại M5 trên toàn bộ Junyi ≥3 seed.
- **B12** — kiểm toán rò rỉ hyperedge: thêm đối chứng dương (tập hyperedge nhiễm bẩn cố ý) — hoặc hạ tuyên bố C1 thành "taxonomy + 1 corpus".
- **B13** — Manipulation check: thêm phá hủy giữ phân bố bậc (permutation cạnh) + hoán vị nhãn quan hệ.
- **B14** — Tier 2 (Critic-Gated Explanations): củng cố (n=80–100, ≥3 người chấm mù, Krippendorff's α) hoặc hạ cấp khỏi tiêu đề/đóng góp.
- **B15** — Chốt MỘT câu luận đề (trục "quy công tri thức có kiểm soát" thay vì "giao thức đánh giá"); viết lại tiêu đề+Abstract; chuyển ATE xuống Phụ lục.
- **C3** — giải nghĩa DH²A ngay câu đầu; thống nhất cách gọi tên mô hình; cấu trúc lại Abstract.

---

## 5. Việc còn treo (cần tác giả quyết định, không phải việc kỹ thuật)

1. **Hợp nhất bản ver3+Gate-0-fixes với nhánh `qkc-zero-time`** — chưa làm. File đã sửa (`main.tex`, `references.bib`, `table_auc_xes3g5m.tex`) hiện chỉ tồn tại dưới dạng đã giao qua chat, KHÔNG commit vào git theo yêu cầu tác giả (05/09/2026). Cần tác giả (hoặc agent được tác giả ủy quyền rõ ràng) quyết định cách hợp nhất — ví dụ tạo nhánh riêng, hay áp từng sửa chữ vào bản `qkc-zero-time` hiện tại.
2. **Xác thực push GitHub** — môi trường thực hiện các sửa Gate 0 này không có thông tin đăng nhập git (`git ls-remote` báo lỗi thiếu Username). Việc push lên `origin` cần thực hiện từ máy/môi trường đã có sẵn xác thực (ví dụ Cursor hoặc terminal cá nhân của tác giả).
3. Ba quyết định Gate 0 còn treo ở mục 1 (A5, B5) và các câu hỏi #2,3,5,6,10,11,12.

---

*File này chỉ là ghi chú bàn giao, không phải đã commit vào git — vui lòng đọc cùng `docs/KeHoach_CaiTien_BanThao_ver3_2026-09-05.md` để có đầy đủ chi tiết từng điểm.*
