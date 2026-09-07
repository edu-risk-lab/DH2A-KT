# Kế hoạch cải tiến bản thảo DH²A-KT — sau nhận xét vòng 2 (ver3, 05/09/2026)

Nguồn: `docs/NhanXet_DH2A-KT_ver3_Tuan_05Sep2026.pdf` (PGS.TS. Nguyễn Văn Hậu, hợp nhất 3 nguồn nhận xét, 25 điểm: 7 mức A, 15 mức B, 3 mức C, tổ chức theo 4 Gate).

Kết luận của thầy Hậu: **chưa nộp được, cần thêm một vòng major revision.** Lý do chính: vòng sửa ver3 đã gỡ đúng điểm chặn lớn nhất của ver2 (thiếu simpleKT trong Bảng 5), nhưng đồng thời sinh ra một lỗi mới (A1) và làm nặng thêm một điểm cũ (A2) — đúng như thầy cảnh báo, **sửa rời rạc từng mối lo bằng từng vòng chạy riêng lẻ chính là nguyên nhân sinh lỗi mới.** Kế hoạch này vì vậy đi theo đúng khung 4 Gate của thầy, và gộp toàn bộ việc cần chạy máy của Gate 2 vào **một đợt chạy duy nhất**.

Tài liệu này là kế hoạch — chưa sửa gì vào `main.tex`. Sau khi xác nhận, có thể bắt đầu thực hiện Gate 0 ngay (không cần GPU).

---

## 0. Việc cần làm ngay lập tức (5 phút, không cần đợi kế hoạch)

**A1 — câu ở §7.1 nói ngược hướng với Bảng 5.** Đây là lỗi tôi (Claude) tự đưa vào bản thảo ở phiên làm việc trước (khi thêm simpleKT vào Bảng 5), và đã lọt qua chính vòng "review lại tính thống nhất" mà tôi tự làm sau đó. Thầy Hậu xếp đây là loại lỗi nguy hiểm nhất: nằm ở câu MỚI thêm, ở đúng chỗ dễ tổn thương nhất của bài, và sai theo hướng CÓ LỢI cho tác giả (làm simpleKT trông tệ hơn thực tế so với twin).

Câu hiện tại (`main.tex`, ngay sau `\input{tables/table_auc_xes3g5m}`):
> "Under this same clean protocol, simpleKT reaches only $0.8214{\pm}0.0007$ (5-seed mean) — below AKT, GIKT, and both DH$^2$-KT arms, and slightly below even the no-graph twin."

Sự thật theo Bảng 5: twin = 0,8202; simpleKT = 0,8214 → simpleKT **cao hơn** twin 0,0012 (≈1,7×SD của chính nó), không phải thấp hơn.

Câu thay thế thầy đề xuất:
> "Under this same clean protocol, simpleKT reaches $0.8214{\pm}0.0007$ (5-seed mean) — below AKT, GIKT and both DH$^2$-KT arms, and only 0.0012 above the no-graph twin, a margin of roughly 1.7 SD of its own seed variation."

Tôi đề xuất sửa câu này ngay khi bạn xác nhận kế hoạch — đây là việc chỉ sửa chữ, không cần chạy gì, và mỗi ngày để lỗi này trong bản thảo là một ngày rủi ro nếu bản thảo được chia sẻ hay nộp nhầm bản.

---

## 1. Bảng điều hướng theo mức hậu quả

| Mức | Số điểm | Ý nghĩa |
|---|---|---|
| A — BUỘC PHẢI SỬA | 7 | Không sửa thì một kết luận không còn được chống đỡ, hoặc bài không tái lập / không qua phản biện |
| B — NÊN SỬA | 15 | Bài vẫn đứng được nhưng yếu đi, người đọc phải tự đoán thay tác giả |
| C — TÙY CHỌN | 3 | Giúp bài đẹp hơn, bỏ qua cũng không sao |

Nhãn A/B/C là **mức hậu quả**; nhãn G0–G3 là **thứ tự làm** (độc lập với A/B/C — có điểm mức B nằm ở Gate 0 vì sửa nhanh, có điểm mức A nằm ở Gate 2 vì phải chờ máy).

Ước lượng thời gian của thầy: G0 ≈ 2–3 ngày (không cần GPU) · G1 ≈ 1 tuần · G2 ≈ 1–2 tuần nếu chạy liên tục · G3 ≈ 3–4 ngày viết lại.

---

## 2. GATE 0 — sửa chữ/số, không cần GPU (làm trước tiên, ước lượng 2–3 ngày)

Đây là phần có thể bắt tay vào ngay. Tôi đã tra một số điểm trực tiếp trên máy của bạn (kho `D:\0. NCS\CODE\DH2A_KT`) để biết điểm nào chỉ cần sửa câu chữ và điểm nào cần bạn cung cấp thêm thông tin.

| # | Mã | Việc | Đã tra được gì | Việc còn lại |
|---|---|---|---|---|
| 1 | **A1** | Sửa câu §7.1 nói ngược hướng simpleKT vs twin | Xác nhận số đúng: twin=0,8202 < simpleKT=0,8214 (Bảng 5) | Sửa câu (mẫu ở mục 0), rà toàn bộ câu mới thêm ở ver3 đối chiếu từng số với bảng |
| 2 | **A6** | Thêm trích dẫn pyKT (đang dùng làm nguồn siêu tham số mà không trích dẫn) | — | Thêm pyKT vào references.bib + trích dẫn ở 4 chỗ (chú thích Bảng 5, §2.3, đầu §4.4, chú thích Bảng 3); viết đoạn 4-6 câu ghi công pyKT rồi liệt kê 3 điểm mới của bài so với pyKT |
| 3 | **B1** | Data availability nói "code available" nhưng link 404 | **Đã tự kiểm tra: `https://github.com/edu-risk-lab/DH2A-KT` xác nhận trả về 404** (không public) | Cần bạn chọn: (a) công khai kho thật, hoặc (b) tạo kho ẩn danh cho phản biện + sửa câu thành "provided to reviewers at …; will be made public upon acceptance"; nên kèm DOI Zenodo |
| 4 | **B2** | 3 con số ở §4.4 không khớp Bảng 3 ("+0.047", so sánh test vs validation, "Tables 8–7" đảo thứ tự) | **Đã tìm ra khả năng cao nguồn gốc "+0.047":** trong `docs/kbs-gpu-runbook.md` có dòng "Manipulation folds 0–2: ΔAUC 0.038 / 0.043 / 0.047" — đúng như thầy Hậu nghi ngờ, "+0.047" nhiều khả năng bị lẫn từ dải ΔAUC của manipulation check (§7.3), không phải từ Bảng 3 (Bảng 3 cho 0,0512 và 0,0563) | Sửa cận dưới thành "+0.051 to +0.056"; đổi phép so sánh sang cùng loại đại lượng (test-vs-test: ×5,4 lần so với khoảng cách twin↔tracer =0,0094; hoặc val-vs-val ×8 lần so với gate 0,00612); sửa "Tables 8–7"→"Tables 7–8" |
| 5 | **B3** | "the mix cannot collapse into e_c" sai về đại số tuyến tính | — | Bỏ mệnh đề đó; nếu muốn giữ 2 đường biểu diễn riêng thì tách $W_q, W_c$ trong PT (1); nếu không thì thay bằng phép đo tỉ lệ chuẩn 2 số hạng |
| 6 | **B4** | Số 6.413.353 "interactions" của XES3G5M thực ra là số hàng đã mở rộng theo KC, gốc là 5.549.635 | — | Sửa câu ở §6.1 tách 2 con số: "5,549,635 canonical attempts … yield 6,413,353 KC-expanded rows"; thêm trích dẫn XES3G5M gốc cho con số 5.549.635 |
| 7 | **B5** | "Pre-registered" không có hiện vật đăng ký; ngưỡng +0,002 chưa biện minh | Có sẵn SD=0,0007 (5-seed simpleKT) trong `a2_multiseed_summary.csv` để biện minh: +0,002 ≈ 2,9×SD | Đổi "pre-registered"→"pre-specified" (trừ khi có OSF/AsPredicted/commit mốc thời gian — xem Câu hỏi #12); thêm câu quy ngưỡng ra bội số SD; áp Holm–Bonferroni cho các PASS; dán nhãn "exploratory" cho phân tầng hậu nghiệm ở §8.3; bỏ nhãn PASS/FAIL khỏi Hình A.2 |
| 8 | **B6** | AKT, simpleKT, DGEKT, SKT, BKT là baseline nhưng thiếu trong tài liệu tham khảo | — | Thêm trích dẫn: AKT (Ghosh et al., KDD 2020), simpleKT (Liu et al., ICLR 2023, arXiv:2302.06881), DGEKT, SKT, BKT; đối chiếu ngược toàn bộ 20 mục đã trích đều được dùng trong thân bài |
| 9 | **B7** | Abstract phát biểu FAIL mạnh hơn mức cổng cho phép ("do not" thay vì "did not meet gate") | — | Sửa 2 câu trong Abstract theo mẫu ở bản nhận xét; thêm câu chú thích dưới Bảng 7 rằng FAIL nghĩa là "không được ghi công trong cấu hình này", không phải "vô giá trị" |
| 10 | **B8** | 2 phát biểu phủ định toàn xưng ("None…", "no published…") + cụm "agentic closed loop" phóng đại | — | Thêm câu mô tả phạm vi tra cứu (database, từ khóa, thời điểm) làm chỗ dựa cho "to our knowledge…"; đổi "the agentic closed loop" → mô tả đúng: ghi session hyperedge trở lại đồ thị nhưng Tier 1 không đọc lại |
| 11 | **A5 (phần rẻ)** | Đổi tên "concept-prerequisite hyperedge" → "path-derived concept hyperedge" (phương án (b) an toàn, không cần chạy lại) | — | *Cần bạn quyết định trước*: chọn phương án (a) sửa biểu diễn giữ hướng (đắt, cần Gate 3) hay phương án (b) đổi tên (rẻ, Gate 0) — xem mục 4 bên dưới |
| 12 | **C1** | PDF đầy khung viền hyperlink màu | — | Thêm `hidelinks` (hoặc `colorlinks=false, allbordercolors=white`) vào `\usepackage[...]{hyperref}`; biên dịch lại và lướt toàn bộ PDF bằng mắt |
| 13 | **C2** | Metadata 3 tài liệu tham khảo [9],[13],[15] sai | **Đã tìm được:** [9] FoundationalASSIST — tác giả đúng là Worden, Heffernan, Heffernan, Sonkar (2026), arXiv:2602.00070, không phải "ASSISTments Foundation" | Sửa [9]; tra [13] trên ACL Anthology (số trang) và [15] trên IEEE Xplore (tập/số/trang/tác giả) — cần bạn tra vì tôi không truy cập được các cơ sở dữ liệu này từ đây |

**Việc cần bạn quyết định trước khi tôi sửa (Gate 0):**
- B1: công khai kho thật hay tạo kho ẩn danh cho phản biện?
- A5: đổi tên (rẻ) hay sửa biểu diễn giữ hướng (đắt, dời sang Gate 3)?
- B5: có hiện vật đăng ký (OSF/AsPredicted/commit) cho ngưỡng +0,002 không, hay đổi hẳn thành "pre-specified"?

---

## 3. GATE 1 — thí nghiệm protocol để khóa phát hiện mạnh nhất của bài (~1 tuần)

Đây là các thí nghiệm **phải chạy trước Gate 2**, vì chúng quyết định giao thức đánh giá chuẩn — nếu đợi tới Gate 2 mới phát hiện vấn đề thì phải chạy lại toàn bộ 5-seed.

| # | Mã | Việc | Ghi chú từ việc tra cứu của tôi |
|---|---|---|---|
| 1 | **A3** | Dư lượng +0,0101 ở Table 3 hàng 4 (p0_dt_on, tracer khuyến nghị) có thể do trùng lặp ở CHUỖI ĐẦU VÀO chứ không chỉ ở đích | **Tôi đã tìm thấy điều quan trọng trong `results/tables/xes3g5m_fold0_p0parity_protocol.csv` và `p0_dt_testonly_clean.csv`: hàng 3 (v4q_clean) và hàng 4 (p0_dt_on) không chỉ khác nhau ở nhánh Δt — chúng còn khác nhau ở `window_mode` (hàng 3 dùng `last`/`p0parity`, hàng 4 dùng `chunked`) và ở số vị trí được chấm (599.546 so với 1.093.755, chênh lệch rất lớn). Đây là câu trả lời một phần cho Câu hỏi #4 của thầy: đúng, có khác biệt khác ngoài Δt.** Cần làm rõ trong bài trước khi kết luận dư lượng +0,0101 là do gì. |
| 2 | | Kiểm tra rẻ trước (nửa buổi, không cần train lại): trích phân bố Δt tại các hàng lặp dưới giao thức raw, xác nhận có bằng 0 không | Có thể làm ngay bằng script đọc dữ liệu đã có, không cần GPU |
| 3 | | Chấm tracer khuyến nghị dưới raw nhưng loại các vị trí Δt=0, so dư lượng còn lại với +0,0101 | |
| 4 | | Xây bản dữ liệu gộp mức lượt thử (attempt-level collapse): mỗi (learner, item, timestamp) → 1 sự kiện | |
| 5 | | Chạy tracer khuyến nghị trên 3 giao thức (raw / mask-target / attempt-collapse) cùng 1 fold, lập bảng 3 dòng | |
| 6 | | Viết lại đoạn cuối §4.4 quy dư lượng +0,0101 đúng thành phần; nếu attempt-collapse khác biệt đáng kể, dùng nó làm giao thức chuẩn và CHẠY LẠI Bảng 5 theo giao thức đó | **Quan trọng: nếu kết quả buộc đổi giao thức chuẩn, phải làm việc này XONG trước khi chạy đợt 5-seed của Gate 2, nếu không sẽ lặp lại đúng lỗi "sửa rời rạc" đã sinh ra A1** |
| 7 | **A7** | Δt phía query (khoảng chờ tới lượt thử tiếp theo) — chứng minh thông tin này có tại thời điểm dự đoán, không chỉ "không nhìn thấy tương lai" | Trên FoundationalASSIST: nhánh query-only giữ +0,01471, nhánh LSTM-only chỉ +0,00280 — phần lớn giá trị nằm ở nhánh "nhìn về phía trước", đây là chỗ phản biện sẽ nhắm đầu tiên |
| 8 | | Thêm định nghĩa "attempt-onset next-response prediction" ở §4.3; chạy 4 nhánh trên XES3G5M (không thời gian / chỉ Δt lịch sử / chỉ Δt query / cả hai); thêm nhánh "deployment-realistic" (gap tính tới thời điểm dự đoán, không tới lượt thử thực tế) | |
| 9 | | Viết tiểu mục "Deployment availability of the query-side gap" ở §8.1 | Nếu nhánh deployment-realistic mất phần lớn lợi ích: đây vẫn là phát hiện có giá trị, không phải thất bại — nên giữ và trình bày trung thực |
| 10 | **B9** | Chấm lại checkpoint simpleKT L=200 của P0 dưới clean protocol (không train lại) để có điểm dữ liệu chéo-kiến-trúc đầu tiên cho mệnh đề "independent of architecture family" | Xem Câu hỏi #2 — nếu checkpoint L=200 còn lưu, đây chỉ là chấm điểm lại, rất rẻ |

---

## 4. GATE 2 — MỘT đợt chạy 5-seed thống nhất, xuất AUC+NLL+Brier+ECE cùng lúc (~1–2 tuần)

**Đây là điểm thầy Hậu nhấn mạnh nhất: gộp TẤT CẢ việc dưới đây vào một đợt chạy duy nhất, không chia nhỏ.** Chỉ bắt đầu Gate 2 sau khi Gate 1 (đặc biệt A3) đã chốt giao thức chuẩn.

Với mỗi cấu hình trong danh sách dưới, chạy đúng bộ 5 seed `{0, 17, 42, 1234, 2024}` — bộ hạt giống đã dùng cho simpleKT trong `a2_multiseed_matrix.csv`:

| # | Mã | Việc |
|---|---|---|
| 1 | **A2** | Chạy cả 6 cấu hình của Bảng 5 (tracer khuyến nghị, Q←KC, twin, GIKT, AKT, simpleKT — simpleKT đã có sẵn) trên cùng 5 seed; báo cáo mọi ô dạng mean±SD; bỏ khái niệm "single canonical run" |
| 2 | **A4** | Trong CÙNG đợt chạy: xuất thêm NLL, Brier, ECE (15 bin) cho cả 6 cấu hình; thêm 3 cột vào Bảng 5; thêm hình reliability diagram (4 đường: tracer, twin, GIKT, AKT); thêm câu ở đầu §5 nêu ECE của checkpoint đóng băng dùng cho Tier 2; nếu ECE > 0,05 thì cân nhắc temperature scaling hoặc ghi rõ vào Limitations |
| 3 | **B10** | Lấy giao (intersection) của tập ID đánh giá giữa DH²-KT (n=1.093.755) và các baseline (n=1.093.720, lệch 35 hàng — **tôi đã xác nhận trong `a1_clean_baselines_L400_fold0.csv`: đây là chênh lệch hệ thống giữa 2 pipeline (DH2-KT tracer riêng vs. pyKT baselines), không phải nhiễu ngẫu nhiên**); chấm lại tất cả trên đúng tập giao; tính lại mọi CI bootstrap trên tập đó |
| 4 | **B11** | Bổ sung phụ lục bảng siêu tham số cho MỌI mô hình (tên tham số / miền tìm / giá trị chọn / cách chọn); tinh chỉnh lại AKT ở L=400 (learning rate, dropout); nới trần epoch GIKT; chạy thêm simpleKT ở L=200 clean để tách phần do độ dài chuỗi |
| 5 | | Nếu quỹ máy cho phép: mở rộng tracer khuyến nghị + GIKT thành 3 fold × 3 seed |

---

## 5. GATE 3 — quyết định phạm vi khoa học, sau khi đã có số của Gate 1–2 (~3–4 ngày viết lại)

Đây là các quyết định **không nên viết lại trước khi có số liệu G1/G2**, để tránh viết lại hai lần.

| # | Mã | Quyết định cần chọn |
|---|---|---|
| 1 | **A5** | (a) sửa Algorithm 1 phát ra danh sách CÓ THỨ TỰ + chạy lại M5 trên toàn bộ Junyi (không chỉ mẫu 5.000), ≥3 seed; hay (b) chỉ đổi tên "concept-prerequisite"→"path-derived concept hyperedge" (đã liệt ở Gate 0 nếu chọn (b)) |
| 2 | **B12** | C1 (kiểm toán rò rỉ hyperedge): bổ sung đối chứng dương (dựng tập hyperedge nhiễm bẩn cố ý, chạy 4 chẩn đoán, kỳ vọng group-membership leak và \|ρ\| khác 0) — hoặc hạ tuyên bố C1 thành "taxonomy + instantiate trên 1 corpus" |
| 3 | **B13** | Manipulation check: bổ sung phép phá hủy giữ nguyên phân bố bậc (permutation cạnh) + phép hoán vị nhãn quan hệ; nói rõ phép hiện tại chỉ chứng minh "phụ thuộc đồ thị" chứ không phải "dùng đúng cấu trúc" |
| 4 | **B14** | Tier 2 (Critic-Gated Explanations): củng cố (thu lại đánh giá người trên chính tracer khuyến nghị, n=80–100, ≥3 người chấm độc lập bị làm mù, dùng Krippendorff's α thay Spearman, thêm đối chứng không tự-đúng-theo-định-nghĩa) — hoặc hạ cấp (bỏ khỏi tiêu đề và danh sách đóng góp, chuyển §5 thành mục mở rộng sơ bộ) |
| 5 | **B15** | Chốt MỘT câu luận đề duy nhất cho toàn bài (thầy đề xuất trục "quy công tri thức có kiểm soát" thay vì trục "giao thức đánh giá", vì pyKT đã sở hữu phần lớn trục thứ hai — xem A6); viết lại tiêu đề + Abstract theo câu đó; chuyển ATE (§4.5, §7.6) xuống Phụ lục |
| 6 | **C3** | Giải nghĩa DH²A ngay câu đầu; thống nhất cách gọi tên (bỏ v1/v2/v4 khỏi Bảng 8/10/Phụ lục A, theo cách Bảng 3 đã làm); đổi tên hệ thống nếu B15 đổi trục và "H2"(hypergraph) không còn khớp hệ thống khuyến nghị; cấu trúc lại Abstract (vấn đề→cách tiếp cận→2-3 phát hiện→giới hạn phạm vi ở cuối, không phải ở đầu) |

---

## 6. Trả lời sơ bộ 12 câu hỏi của thầy Hậu — dựa trên rà soát file đã có trên máy

Tôi đã tra trực tiếp trong kho `D:\0. NCS\CODE\DH2A_KT` để trả lời được phần nào không cần hỏi bạn hay chạy thí nghiệm mới:

| # | Câu hỏi | Kết quả tra cứu của tôi |
|---|---|---|
| 1 | Kho GitHub đã public chưa? | **Đã tự kiểm tra: `https://github.com/edu-risk-lab/DH2A-KT` trả về 404** — xác nhận đúng như 2 người đọc độc lập báo cáo. B1 phải xử lý trước khi nộp. |
| 2 | Checkpoint simpleKT L=200 của P0 (cho số 0,8747 ở Bảng A.13) còn lưu không? | Chưa xác nhận được — cần bạn kiểm tra thư mục `checkpoints/` hoặc nơi lưu trữ mô hình P0 gốc. Nếu còn, B9 chỉ cần chấm điểm lại, không cần train lại. |
| 3 | Đợt 5-seed simpleKT có đi đúng pipeline chính không? | Có bằng chứng gián tiếp ủng hộ: `a2_multiseed_matrix.csv` cho n=1.093.720 ở mọi seed simpleKT, khớp chính xác với n của AKT/GIKT trong `a1_clean_baselines_L400_fold0.csv` — cùng pipeline baseline. Nhưng cần bạn xác nhận về mặt code path vì tôi không chạy lại được. |
| 4 | Bảng 3 hàng 3/4 khác nhau ở điểm nào ngoài Δt? | **Tìm thấy: CÓ, khác nhiều hơn chỉ Δt.** Hàng 3 (v4q_clean, nguồn `xes3g5m_fold0_p0parity_protocol.csv`) dùng window "last"/"parity", n=599.546 vị trí clean; hàng 4 (p0_dt_on, nguồn `p0_dt_testonly_clean.csv`) dùng `window_mode=chunked`, n=1.093.755 vị trí — chênh lệch rất lớn về cách cắt cửa sổ đánh giá. Đây củng cố mạnh giả thuyết A3 của thầy rằng dư lượng +0,0101 có thể không thuần túy do Δt. |
| 5 | M5 (expert DAG Junyi) nạp dạng cạnh đôi có hướng hay chain hyperedge? | Chưa tra được từ dữ liệu — cần đọc code nạp dữ liệu M5 cụ thể (nằm ngoài phạm vi tra cứu nhanh này); nên làm cùng lúc quyết định A5. |
| 6 | HypergraphConv có hướng hay không hướng? | Chưa tra — cần đọc `dh2a_kt/models/` phần toán tử tích chập; có thể trả lời trong vài phút khi ngồi sửa A5. |
| 7 | Nguồn con số "+0.047" ở §4.4? | **Rất có khả năng tìm ra: `docs/kbs-gpu-runbook.md` ghi "Manipulation folds 0–2: ΔAUC 0.038 / 0.043 / 0.047"** — khớp đúng nghi ngờ của thầy rằng "+0.047" bị lẫn từ dải ΔAUC manipulation check (§7.3, 0,038–0,047), không phải từ Bảng 3. Cần xác nhận thêm khi sửa B2. |
| 8 | Chênh lệch 35 hàng (1.093.755 vs 1.093.720) đến từ đâu? | **Xác nhận: đây là chênh lệch hệ thống, nhất quán** — trong `a1_clean_baselines_L400_fold0.csv`, tracer DH²-KT (`p0_dt_on`) luôn có n=1.093.755, còn TẤT CẢ baseline (simpleKT, DKT, AKT, GIKT, DyGKT, DGEKT) đều có n=1.093.720 — tức khác biệt nằm ở pipeline riêng của DH²-KT so với pipeline pyKT dùng chung cho baseline, không phải nhiễu ngẫu nhiên giữa các lần chạy. Cần tìm đúng bước xử lý chuỗi gây khác biệt (có thể là cách cắt chuỗi cuối cùng ở L=400). |
| 9 | ℓmin/ℓmax của Algorithm 1 là bao nhiêu? | **Tìm thấy: `min_chain_len=3`, `max_chain_len=8`** — có trong `configs/xes3g5m.yaml`, `configs/junyi.yaml`, và code `dh2a_kt/hyperedge/construction.py`. Chỉ cần thêm vào bài, không phải thiếu sót về thực nghiệm — thiếu sót trình bày, sửa ở Gate 0. |
| 10 | 2 người chấm A1/B1 (Bảng 11) là ai, có bị làm mù không? | Cần bạn xác nhận — thông tin nhân sự không nằm trong file kho mã. |
| 11 | arXiv:2508.17092 có trùng phạm vi không? | Cần đọc trực tiếp bài đó — tôi có thể tra và đọc nếu bạn muốn, hiện chưa làm vì đây là việc thuộc Gate 0 mục B6 phần mở rộng Related Work. |
| 12 | Có hiện vật đăng ký (OSF/AsPredicted/commit) cho ngưỡng +0,002 không? | Cần bạn xác nhận — không tìm thấy trong kho mã đã rà (không có thư mục OSF hay ghi chú commit-timestamp riêng cho việc này). |

**Tóm lại: Câu #1, #4, #7, #8, #9 gần như đã có câu trả lời từ dữ liệu sẵn có — không cần hỏi thêm.** Câu #2, #3, #5, #6, #10, #11, #12 cần bạn xác nhận hoặc tôi cần đọc thêm code/tài liệu cụ thể khi bắt tay sửa từng điểm.

---

## 7. Những phần thầy Hậu nói đang tốt — GIỮ NGUYÊN khi sửa

- Bảng 5 + §6.2: việc thêm simpleKT dưới clean protocol là xử lý đúng, đã gỡ điểm chặn lớn nhất của ver2. Chỉ bổ sung 5-seed cho các dòng còn lại (Gate 2), không viết lại dòng simpleKT.
- §7.1 + Bảng 3: con số 0,8747 − 0,8214 = 0,0533 khớp gọn trong dải 0,0512–0,0563 của Bảng 3 (trên kiến trúc khác họ) — bằng chứng mạnh, bài chưa khai thác hết (xem B9). Đừng xóa đoạn này khi sửa A1, chỉ sửa đúng câu bị sai hướng.
- Bảng 3 + chú thích: cách đặt tên checkpoint theo mô tả kiến trúc (đã làm ở phiên trước) — áp dụng tương tự cho Bảng 8, 10, Phụ lục A (điểm C3).
- Bảng 7: giữ nguyên toàn bộ 6 dòng FAIL, kể cả HypergraphConv — đây là điểm mạnh hiếm gặp, không rút gọn.
- §8.1, §9: cách khai báo phạm vi trung thực (đã cập nhật ở phiên trước) — giữ nguyên tinh thần khi viết lại ở Gate 3.
- Chú thích Bảng 5, SD=0,0007: mỏ neo nhiễu seed đầu tiên — dùng ngay cho B5, đừng bỏ khi chuyển sang 5-seed đồng bộ ở Gate 2.
- Toàn bộ số liệu khác của ver3 đã được thầy kiểm chéo và khớp — A1 là ngoại lệ, không phải xu hướng, nghĩa là quy trình kiểm soát số liệu nhìn chung đang tốt.

---

## 8. Đề xuất trình tự thực hiện

1. **Ngay bây giờ (5 phút):** sửa câu A1 — chỉ cần xác nhận, tôi sửa trực tiếp vào `main.tex`.
2. **Gate 0 (2–3 ngày):** làm hết bảng ở mục 2, theo thứ tự A trước B trước C; 3 quyết định cần bạn chọn (B1, A5, B5) nên chốt sớm vì ảnh hưởng cách viết các phần khác.
3. **Gate 1 (~1 tuần):** ưu tiên A3 trước A7, vì A3 có thể buộc đổi giao thức chuẩn — nếu đổi, phải xong trước khi vào Gate 2.
4. **Gate 2 (~1–2 tuần):** một đợt chạy 5-seed duy nhất, xuất AUC+NLL+Brier+ECE cùng lúc cho toàn bộ Bảng 5, dùng đúng tập ID giao (B10) và ngân sách đã khớp (B11).
5. **Gate 3 (~3–4 ngày):** viết lại các phần phụ thuộc quyết định phạm vi (A5 phần đắt nếu chọn, B12–B15, C3) — chỉ sau khi có số của Gate 1–2, tránh viết lại hai lần.
6. **Trước khi nộp:** hidelinks (C1), metadata tham khảo (C2), rà toàn văn lần cuối đối chiếu mọi số với bảng nguồn (bài học từ chính lỗi A1).

Bạn muốn tôi bắt đầu từ đâu — sửa A1 ngay, hay đi hết Gate 0 trước?
