**BẢN NHÁP DO MÁY SOẠN — CHƯA ĐƯỢC NGƯỜI NHẬN XÉT DUYỆT**

# Nhận xét bản thảo (vòng 4): “DH²A-KT: Leakage-Audited Knowledge Attribution for Knowledge Tracing”

Bản thảo bài báo tạp chí (dự kiến nộp Knowledge-Based Systems, Elsevier) — bản attribution sau Gate 0 + một phần Gate 2 · Vòng 4 — đối chiếu với nhận xét ver3 ngày 05/09/2026

Tác giả bản thảo: NCS Đào Minh Tuấn — tác giả thứ nhất

Người nhận xét: *không in tên — bản máy soạn, chưa duyệt*

Ngày: 07/09/2026

Phạm vi đã đọc: Toàn văn `paper/main.tex` (aux ghi 49 trang), 20 bảng trong `paper/tables/`, `paper/references.bib` (33 mục), Abstract / C2 / Results / Limitations / Data availability / Appendix A–D. Đã đối chiếu từng số headline với `results/tables/a2_multiseed_matrix.csv`, `a2_multiseed_summary.csv`, `qkc_capacity_matched_matrix.csv`, `a7_query_dt_ablation.csv`, `a6_calibration_xes3g5m_fold0.csv`, `p0_dt_testonly_clean.csv`, và mã ghi nhận metric trong `scripts/27_a2_multiseed_matrix.py`, `scripts/31_a7_query_dt_ablation.py`, `scripts/34_qkc_capacity_matched.py`. Đã đối chiếu từng điểm A/B/C của bản 05/09. Đã thử mở `https://github.com/edu-risk-lab/DH2A-KT` (HTTP 404).

Phạm vi chưa xem: chưa train lại mô hình, chưa mở từng file `.npz` dự đoán, chưa đọc bản thảo đồng hành P0, chưa xem Zenodo (chưa có DOI), chưa chấm Likert thô. Những chỗ không kiểm được nằm ở mục 6, không nằm trong danh sách thiếu sót.

Đã tự tính lại: trung bình và SD 5 seed của `a2_multiseed_summary.csv` (QKC-T 0,82961955±0,000100; Q←KC 0,82697084±0,000193; GIKT 0,82584603±0,000100; AKT 0,82225073±0,000119; simpleKT 0,82141400±0,000658; pathway-off 0,82059691±0,000190); hiệu số thô QKC-T−GIKT +0,003774, QKC-T−AKT +0,007369, QKC-T−simpleKT +0,008206; trung bình Δval capacity-matched (−0,00056−0,00010−0,00021+0,00038−0,00003)/5 = −0,000104; trung bình Δval Zero+Δt (0,00242+0,00245+0,00243+0,00270+0,00272)/5 = 0,002544; trung bình test-only Zero+Δt từ cột `test_only_auc` = 0,829486±0,000148; paired *t* trên năm Δval `dt_both` (trung bình 0,002735, SD 0,000255) → *t*₄ = 23,99; trung bình 3 fold (0,8296+0,8298+0,8291)/3 = 0,8295, SD 0,00036; seed 42 test-only QKC-T 0,829593 (`a6`) trừ combined 0,829754 (`a2`) = 0,000161; *n* combined A2 = 1.640.242 = 1.093.755 + 546.487.

---

## 1. BẢNG ĐIỀU HƯỚNG

| Mức | Số điểm | Nghĩa là gì |
|---|---|---|
| **A — BUỘC PHẢI SỬA** | 3 | Không sửa thì một kết luận không còn được chống đỡ, hoặc bản thảo không tái lập / không qua được vòng phản biện. |
| **B — NÊN SỬA** | 8 | Bản thảo vẫn đứng được, nhưng yếu đi và người đọc phải tự đoán thay tác giả. |
| **C — TUỲ CHỌN** | 3 | Tuỳ chọn — gợi ý để bản thảo đẹp hơn, không sửa cũng không sao. |

▸ So với bản 05/09 (7 A / 15 B / 3 C trên ver3): vòng này **gỡ đúng cụm chữ** — A1 cũ (simpleKT thấp hơn twin) đã hết; A2 cũ (một lần chạy cạnh 5 seed) đã hết trên Bảng 5; A4 cũ (không có ECE) đã có bảng; A5 cũ chọn phương án đổi tên; A6 cũ (pyKT) đã có trích dẫn; A7 cũ được định nghĩa attempt-onset + bốn nhánh. Trục bài đổi thành attribution, Tier 2 hạ khỏi đóng góp, “pre-registered” thành “pre-specified”, 5.549.635 tách khỏi 6.413.353, đối chứng dương membership = 1,0. Đó là một vòng sửa có hướng, không phải sửa rời.

▸ Nhưng cụm gốc của **ba điểm A lần này** không phải ý tưởng khoa học mới. Nó là cùng một van đã rò ở ver3: **số trên bảng chính không phải số mà câu văn (và Abstract) đang bán**. Lần trước là một mệnh đề thứ tự sai 0,0012. Lần này nặng hơn: Bảng 5 gọi mọi dòng là “test-only / *n* = 1.093.720”, trong khi ba hàng DH² lấy `dh2_kt_auc` — mà chính `scripts/34_qkc_capacity_matched.py` đổi tên thành `combined_eval_auc` với *n* = 1.640.242 (val+test) — còn GIKT/AKT/simpleKT là AUC test-only *n* = 1.093.720. Cùng bảng, hai metric. Abstract đặt 0,8296 (combined) cạnh 0,8295 (test-only của Zero+Δt) cạnh 0,8258 (test-only của GIKT).

▸ Điểm A3 vòng 3 (dư lượng +0,0101 / hàng lặp phía đầu vào) **chưa được chẩn đoán**. Bảng protocol đã biến mất khỏi bản thảo; CSV nội bộ vẫn còn residual +0,01013. Đây không phải điểm mới — đây là điểm cũ bị **giấu bằng cách xoá bảng**.

▸ Kết luận: **chưa nộp được**. Gate 0 lần này chỉ là sửa chữ/số và *re-score CPU* — không cần GPU. Đừng chạy thêm thí nghiệm attribution nào trước khi Bảng 5 và Abstract nói cùng một metric trên cùng một *n*.

▸ Ước lượng: A1+A2 khoảng nửa buổi nếu các checkpoint/npz 5 seed còn (chỉ chấm lại test-only trên giao 1.093.720). A3 nửa buổi chẩn đoán rẻ + 2–3 ngày nếu phải dựng attempt-collapse. Tám điểm B khoảng 2–3 ngày chữ, trừ B1 (Zenodo/kho) phụ thuộc bên ngoài. Ba điểm C dưới một giờ.

---

## 2. THỨ TỰ VIỆC CẦN LÀM

Làm hết mức A trước, rồi mới sang mức B. Mức C để cuối, bỏ được nếu hết thời gian.

Ngoại lệ tài nguyên: A1/A2 là `may` nhỏ (chấm lại, không train) — chạy **trước** mọi câu chữ Abstract, vì A2 phụ thuộc số A1 sẽ đổi. A3 chẩn đoán rẻ (`doc`/`may` nửa buổi) trước khi quyết định có dựng attempt-collapse. B1 (`ngoai`) mở song song vì chờ URL/DOI.

| # | Mã | Việc | Công sức | Tài nguyên |
|--:|:---|:-----|:--------:|:----------:|
| 1 | **A1** | Tách AUC gộp val+test khỏi AUC test-only; dựng lại Bảng 5 (và A7 `both`) trên một metric / một *n* | vừa | máy (chấm lại, không train) |
| 2 | **A2** | Đưa Zero+Δt thành một hàng tuyệt đối trong Bảng 5; Abstract chỉ dẫn số có trong bảng | nhỏ | đọc/viết |
| 3 | **A3** | Chẩn đoán dư lượng +0,0101; đừng để giao thức headline đứng trên một bảng đã xoá | lớn | máy |
| 4 | **B1** | Kho GitHub vẫn 404; chốt URL thật hoặc câu “cung cấp cho phản biện” | nhỏ | ngoài |
| 5 | **B2** | 35 hàng lệch và bootstrap “ghép cặp” vẫn chưa ghép theo vị trí | nhỏ | máy |
| 6 | **B4** | History-only FAIL 2/5 nhưng hệ được bán vẫn là nhánh `both` | vừa | đọc/viết |
| 7 | **B3** | Ngân sách tinh chỉnh baseline vẫn không khớp — giữ giới hạn, siết câu Abstract | nhỏ | đọc/viết |
| 8 | **B5** | Holm–Bonferroni được tuyên bố nhưng họ kiểm không được viết ra | nhỏ | đọc/viết |
| 9 | **B6** | Bảng attribution đặt 4 hàng 1-seed cạnh hàng 5-seed như cùng một cổng | nhỏ | đọc/viết |
| 10 | **B7** | Chú thích bảng còn chữ “GS Hau A4” / “Hau B12” | nhỏ | đọc/viết |
| 11 | **B8** | M5 vẫn 5k-user + hyperedge không thứ tự — giữ FAIL, siết câu | nhỏ | đọc/viết |
| 12 | **C1** | Hiệu +0,0074 (từ số thô) trông thành +0,0073 nếu trừ hai ô đã làm tròn | nhỏ | đọc/viết |
| 13 | **C2** | Nhãn v2/v4 còn ở Bảng paired / Tier 2 | nhỏ | đọc/viết |
| 14 | **C3** | Pin commit `09e0b96` và xem PDF bằng mắt trước khi gửi | nhỏ | đọc/viết |

Câu chốt: làm hết A trước. A1 trước A2 vì hàng Zero+Δt phải cùng metric với các hàng còn lại. Toàn bộ mức B về diễn giải Δ*t* / GIKT chờ số A1. C bỏ được nếu hết thời gian — trừ C3 (năm phút, làm cùng Gate 0).

---

## 3. MỨC A — BUỘC PHẢI SỬA (3 điểm)

Không sửa thì một kết luận của bản thảo không còn được chống đỡ, hoặc bản thảo không tái lập / không qua được vòng phản biện.

### A1. Bảng 5 đặt AUC gộp val+test của DH² cạnh AUC test-only của baseline, rồi gọi cả bảng là test-only trên 1.093.720 vị trí

Vị trí: Bảng 5 (`table_auc_xes3g5m.tex`) và chú thích; Abstract câu về 0,8296 / 0,8258 / 0,8214; §7.1; Bảng A7 hàng `both` · Loại: `so-sanh-khong-cong-bang` · Công sức: vừa

Bản thảo viết: “Test-only AUC on XES3G5M fold 0 under the clean protocol … All models are scored on an identical set of 1,093,720 target positions. Every row is mean±SD over five training seeds.”

Lỗi là gì: Ba hàng DH² của Bảng 5 (QKC-T 0,8296±0,0001; history-only 0,8288±0,0002; Q←KC 0,8270±0,0002) lấy từ cột `auc`/`test_auc` của `a2_multiseed_matrix.csv`. Script A2 gán cả hai cột bằng `dh2_kt_auc` từ CSV so sánh (`a2_dh2_*_s*.csv`). Các file đó ghi `n_predictions = 1.640.242`. Chính script capacity-matched — viết sau, có kỷ luật hơn — đổi tên cùng một trường thành `combined_eval_auc` và tách riêng `test_only_auc` (*n* = 1.093.755). Seed 42 là bằng chứng số: combined QKC-T = 0,829754 (A2) còn test-only = 0,829593 (bảng calibration, cùng checkpoint). Hiệu 0,000161 **lớn hơn** SD 5 seed mà bài in cho QKC-T (0,0001). Các hàng GIKT / AKT / simpleKT / DKT / DGEKT đi đường pyKT, `n_scored = 1.093.720`, là test-only. Cột “*n* scored” của Bảng 5 in 1.093.720 cho *mọi* dòng DH², trong khi CSV A2 để trống `n_scored` ở đúng các dòng đó.

Vì sao là lỗi [Trong bản thảo]: Đây không phải sai số làm tròn. Đây là hai đại lượng khác nhau đứng trong một cột có một cái tên. Phản biện chỉ cần mở `a2_dh2_p0_dt_on_s42.csv` (một file bài tự khai là nguồn) và thấy 1.640.242. Từ đó trở đi, khoảng +0,0038 so với GIKT bị đọc như hiệu giữa (val+test của mình) và (test của họ). Độ lớn thật của seed 42 test-only vs GIKT seed 42 vẫn khoảng +0,0039 nên *hướng* kết luận có thể sống — nhưng bài đang tự phá cái van mà cả vòng 05/09 đã nhắc: caption phải mô tả đúng phép đo. Nặng hơn ver3 A1 vì nó không nằm ở một mệnh đề văn, mà nằm ở **bảng kết quả chính** và ở Abstract.

Cách khắc phục [Suy luận của tôi]

1. Với mỗi checkpoint A2 của DH² (5 seed × QKC-T / Q←KC / pathway-off / history-only), chấm lại **test-only** trên giao 1.093.720 — CPU, không train. Script 34 đã làm đúng việc này cho capacity-matched; sao chép `_test_only_score`.
2. Dựng lại Bảng 5 từ một cột duy nhất. Chú thích chỉ còn một câu: “mọi ô là test-only trên cùng *N* vị trí”.
3. Hàng `both` của Bảng A7 đang `imported_a2` — cùng `dh2_kt_auc` gộp. Thay bằng số test-only vừa chấm, để ba nhánh A7 cùng metric.
4. Viết lại Abstract, §7.1, C2: khoảng cách 5 seed tính từ **cùng** cột test-only. Không trừ combined cho test-only.
5. Thêm một dòng vào Limitations: bản trước (kể cả cover letter) đã in 0,8296 từ cột gộp; bản này thay bằng test-only.

Viết lại mẫu: “All rows in Table 5 are test-only AUC on the same 1,093,720 positions (pyKT-aligned intersection). DH² combined val+test scores (*n* = 1,640,242) are archived in the repository and are not reported as test-only.”

Tự kiểm: Mọi dòng Bảng 5 có cùng *n*; với seed 42, ô QKC-T trùng 0,829593 của Bảng calibration (hoặc ô giao 1.093.720 nếu khác 35 hàng, thì Bảng calibration cũng phải đổi). `grep` “1,640,242” không còn bị gọi là test-only. Hiệu QKC-T−GIKT in trong Abstract trừ được trực tiếp từ hai ô Bảng 5.

Lưu ý: Em có thể phản bác rằng 0,8296 làm tròn thì trùng test-only seed 42 (0,829593) lẫn trung bình combined 5 seed (0,829620). Đúng — đó là lý do lỗi sống sót qua Gate 0. Hai số khác nhau, một cách làm tròn. Phải tách chúng, không được dựa vào sự trùng ngẫu nhiên của bốn chữ số.

---

### A2. Zero+Δt là hệ được ghi công nhưng không có hàng tuyệt đối nào trong bảng kết quả chính

Vị trí: Abstract (“credited-minimal positive control (test AUC 0.8295±0.0001)”); C2 cùng số; Bảng 5 không có hàng này; Bảng Zero+Δt chỉ có Δval / Δtest · Loại: `so-khong-tai-lap` · Công sức: nhỏ

Bản thảo viết: “Zero+Δt is the credited-minimal positive control (test AUC 0.8295±0.0001).”

Lỗi là gì: 0,8295±0,0001 **đúng** nếu lấy trung bình cột `test_only_auc` của năm hàng `zero_time` trong `qkc_capacity_matched_matrix.csv` (0,829598 / 0,829533 / 0,829441 / 0,829251 / 0,829609 → 0,829486±0,000148). Nhưng không bảng nào đã in cho người đọc con số tuyệt đối đó. Bảng Zero+Δt chỉ in Δ. Bảng 5 có pathway-off (capacity-mismatched, 0,8206) chứ không có Zero+Δt. Người đọc trừ 0,8270 − 0,00004 + 0,00269 cũng không ra 0,8295, vì 0,8270 là combined còn hai Δ kia là test-only / val.

Vì sao là lỗi [Suy luận của tôi]: Bài vừa tuyên bố một quyết định khoa học mạnh — incidence không được ghi công, hệ tối thiểu là Zero+Δt — rồi để hệ đó **không có mặt** trên bảng so với GIKT. Abstract vẫn dẫn QKC-T 0,8296 vs GIKT 0,8258 như khoảng cách headline. Người đọc trung thành với C2 sẽ hỏi: vậy 0,8295 so GIKT là bao nhiêu, trên cùng *n*? Hiện không trả lời được từ PDF. Đó đúng là lỗi `so-khong-tai-lap` mà khuôn này xếp mức A khi số ấy chống đỡ đóng góp.

Cách khắc phục [Suy luận của tôi]

1. Sau A1, thêm một hàng “Zero+Δt (credited-minimal)” vào Bảng 5, cùng metric, cùng *n*, mean±SD 5 seed.
2. Abstract dẫn **cả hai** hàng đã in: QKC-T (evaluation arm) và Zero+Δt (credited), mỗi cái một hiệu số vs GIKT lấy từ bảng.
3. Bỏ hàng pathway-off khỏi Bảng 5 hoặc chuyển xuống chú thích / phụ lục — nó không còn là twin ghi công.

Viết lại mẫu: “Zero+Δt, the credited-minimal control, reaches test-only AUC 0.8295±0.0001 on the same *N* positions (Table 5); the five-seed mean gap versus compute-matched GIKT is +0.00XX.”

Tự kiểm: Số 0,8295±0,0001 xuất hiện nguyên văn trong một ô Bảng 5 (hoặc ô đã đổi sau A1). Abstract không còn số tuyệt đối nào của Zero+Δt mà không có trong bảng.

---

### A3. Clean protocol vẫn chỉ che đích; dư lượng +0,0101 còn trong CSV và bảng đã bị xoá khỏi bản thảo

Vị trí: §4.4 / clean protocol hiện tại (trang ~16); `results/tables/p0_dt_testonly_clean.csv` hàng `p0_dt_on`; bản 05/09 A3 · Loại: `ro-ri-du-lieu` · Công sức: lớn

Không tìm thấy trong bản thảo: Không còn bảng raw / mask-repeats / residual. Không còn câu “+0.0101 remains for the recommended tracer”.

Lỗi là gì: File `p0_dt_testonly_clean.csv` vẫn ghi residual = 0,0101279 giữa AUC gộp-cửa-sổ 0,83972 (*n* = 1.277.288) và test-only clean 0,829593 (*n* = 1.093.755) cho đúng tracer QKC-T. Bản 05/09 đã chỉ ra residual này lớn hơn cả thang ablation, và lớn hơn nhiều residual của hàng không có Δ*t*. Giả thuyết còn mở: hàng lặp có Δ*t* = 0 tương tác với nhánh thời gian. Gate 0 ngày 07/09 ghi rõ “Không đưa residual +0.0101 / attempt-collapse vào Abstract” và “A3 không chạy”. Kết quả: tín hiệu biến mất khỏi PDF, giao thức headline không đổi, thí nghiệm 5 seed (A2/A4/A7) đã chạy trên đúng giao thức chưa được chẩn đoán.

Vì sao là lỗi [Suy luận của tôi]: Twin Δval +0,00274 **cùng** giao thức nên khoảng cách ghi công có thể sống. Cái không sống được là câu “clean protocol” như thể đã xử lý xong hiện tượng row-repeat. pyKT khuyến nghị question-level chứ không chỉ mask target; bài đã trích dẫn pyKT rồi vẫn dừng ở mask-repeats. Việc xoá bảng protocol — thay vì thêm một hàng chẩn đoán — là đúng cơ chế vòng 9 Scientometrics: sửa một file, van rò ở file kia. Ở đây van rò là “bảng biến mất, CSV còn”. Với một bài lấy kiểm toán rò rỉ làm bản sắc, phản biện đọc được CSV (nếu kho mở) sẽ hỏi vì sao PDF im.

Cách khắc phục [Đề xuất mới]

1. Nửa buổi, không train: phân bố Δ*t* tại hàng lặp dưới raw; tỉ lệ vị trí Δ*t* = 0 trong tập chấm.
2. Chấm QKC-T raw nhưng loại vị trí Δ*t* = 0; so residual còn lại với +0,0101.
3. In lại một bảng ba dòng raw / mask-target / (nếu làm) attempt-collapse, kể cả khi chọn giữ mask-target — để residual không còn là kiến thức nội bộ.
4. Nếu (2) cho thấy Δ*t*×repeat chiếm phần lớn residual: viết một câu Limitations rằng credit timing là trên tập đã mask, và không ngoại suy ra raw KC-rows.
5. Chỉ dựng attempt-collapse và chạy lại Bảng 5 nếu (2)–(3) cho thấy mask-target không đủ. Đừng train 5 seed trước bước này — đó là đúng lời 05/09 mà vòng này đã không theo.

Tự kiểm: PDF có một bảng hoặc một câu nêu residual +0,0101 thuộc thành phần nào; § clean protocol không còn đọc như thể mask-repeats đã đóng hồ sơ.

Lưu ý: Nếu em chứng minh được residual chỉ là hiệu giữa chấm mọi KC-row và chấm hàng đã mask — tức đúng cái pyKT cảnh báo, không phải lặp phía lịch sử — thì hạ A3 xuống B và giữ mask-target. Phải có số, không được hạ bằng cách xoá bảng.

---

## 4. MỨC B — NÊN SỬA (8 điểm)

Bản thảo vẫn đứng được, nhưng yếu đi và người đọc phải tự đoán thay tác giả.

### B1. Data availability vẫn chỉ tới một kho trả về 404

Vị trí: Mục Data availability · Loại: `khong-tai-lap-duoc` · Công sức: nhỏ

Bản thảo viết: “Code matching this manuscript is at https://github.com/edu-risk-lab/DH2A-KT (default branch, commit 09e0b96; a Zenodo snapshot DOI will be minted at submission). The repository will be made public at submission.”

Lỗi là gì: Đường dẫn vừa được thử lại ngày 07/09 và vẫn HTTP 404. Câu đã mềm hơn ver3 (không còn khẳng định “available” như thể đang mở) — đúng hướng. Nhưng URL in trong PDF vẫn là URL chết, và chưa có DOI Zenodo, chưa có kho ẩn danh cho phản biện.

Vì sao là lỗi [Chuẩn mực ngành]: Ban biên tập Elsevier bấm liên kết ở khâu kỹ thuật. 404 ở mục bắt buộc làm mất lòng tin vào mọi câu “pinned commit” còn lại, kể cả pin P0 `5e3d6a0`.

Cách khắc phục [Chuẩn mực ngành]

1. Hoặc mở kho trước khi gửi, hoặc thay URL bằng “reviewer snapshot at … / Zenodo DOI”.
2. Sau commit sửa A1, cập nhật pin — `09e0b96` sẽ cũ ngay.
3. README ánh xạ Bảng → script đã có `docs/paper-artifacts.md`; giữ.

Tự kiểm: Mở URL trong cửa sổ ẩn danh: vào được, hoặc câu văn không còn in URL chết.

---

### B2. Lệch 35 hàng vẫn còn; bootstrap ghép cặp vẫn không giao theo vị trí

Vị trí: Chú thích Bảng 5; Appendix bootstrap; `table_bootstrap_qkct.tex` · Loại: `so-sanh-khong-cong-bang` · Công sức: nhỏ

Bản thảo viết: “Native DH² scoring … yields 1,093,755 positions (difference of 35). … Learner-level bootstrap … does not intersect the 35-row position gap.”

Lỗi là gì: Bài đã khai đúng sự lệch — tốt hơn ver3. Nhưng vẫn gọi bootstrap là “paired” trên 3614 học viên trong khi hai file khác 35 hàng, và Bảng 5 tuyên bố mọi mô hình chung 1.093.720 trong khi A1 cho thấy hàng DH² chưa chắc đã được chấm trên giao đó.

Vì sao là lỗi [Suy luận của tôi]: 35/1,09e6 không đổi dấu CI. Lỗi là tên gọi. Sau A1, nếu DH² đã chấm trên giao thì B2 chỉ còn sửa chú thích bootstrap.

Cách khắc phục [Suy luận của tôi]

1. Gói vào A1: chấm mọi mô hình trên giao ID vị trí, một *N*.
2. Tính lại bootstrap trên giao đó, hoặc đổi tên thành “learner-level bootstrap on overlapping learners, not position-aligned.”
3. Một *N* trong Bảng 5 *và* Bảng calibration.

Tự kiểm: Cột *n* đồng nhất; chú thích bootstrap không còn vừa nói “paired” vừa nói “does not intersect.”

---

### B3. “Compute-matched” không phải tinh chỉnh khớp, và Abstract vẫn có thể bị đọc như đã thắng GIKT đã tune

Vị trí: Abstract; chú thích Bảng 5; Appendix hparams · Loại: `so-sanh-khong-cong-bang` · Công sức: nhỏ

Bản thảo viết: “five-seed mean gaps +0.0038 and +0.0082; compute-matched, not a matched tuning search.”

Lỗi là gì: Phụ lục đã đủ thành thật: GIKT *n*=1 config, dừng epoch 8/patience 5; AKT/simpleKT thừa hưởng L=200. Abstract đã có cụm “not a matched tuning search”. Với khoảng +0,0038, một phản biện vẫn có thể viết một câu: họ tune thang ablation, mình một vector pyKT.

Vì sao là lỗi [Chuẩn mực ngành]: Pineau et al. (2021) — bài đã trích — phân biệt khớp compute và khớp search. Bài làm đúng việc khai, chưa làm đúng việc **đặt khoảng cách nhỏ sau caveat, không trước**.

Cách khắc phục [Chuẩn mực ngành]

1. Abstract: một mệnh đề caveat *trước* +0,0038, hoặc bỏ hiệu vs GIKT khỏi Abstract, để lại Bảng 5.
2. Không train lại AKT/GIKT ở vòng này (quỹ máy). Giữ Appendix.

Tự kiểm: Ba câu đầu Abstract không còn đọc như QKC-T thắng GIKT đã tune.

---

### B4. History-only FAIL 2/5; hệ được bán vẫn là nhánh `both` phụ thuộc khoảng chờ phía query

Vị trí: Abstract; §4.3; Bảng A7; tiểu mục Deployment · Loại: `pham-vi-han-che` · Công sức: vừa

Bản thảo viết: “History-only LSTM Δt also fails the gate (2/5 PASS). … The recommended tracer therefore assumes attempt-onset availability.”

Lỗi là gì: Số A7 nhất quán: `both` 5/5 mean Δval +0,00274; `lstm` 2/5 +0,00193; `query` 0/5 +0,00099. Limitations đã viết planned-gap không đánh giá. Vậy mà Abstract và C2 vẫn lấy QKC-T (`both`) làm cánh tay đánh giá, và 0,8296 là số bán hàng. Hệ triển khai được mà không nhìn trước — history-only — không vượt cổng.

Vì sao là lỗi [Suy luận của tôi]: Với bài kiểm toán rò rỉ, đây là chỗ phản biện nhắm đầu tiên. Bản 05/09 A7 đã nói vậy. Vòng này *đo* được rồi — tốt — nhưng chưa để số đó điều khiển câu bán hàng. Không cần thêm GPU.

Cách khắc phục [Đề xuất mới]

1. Abstract: một câu “timing credit is attempt-onset (`both`); history-only is 2/5 FAIL.”
2. Nếu sau A1 số `both` test-only vẫn là headline: đổi “recommended tracer” thành “evaluation arm under attempt-onset scoring.”
3. Planned-gap: giữ Limitations, đừng hứa sẽ có.

Tự kiểm: Abstract không còn đọc như Linear Δ*t* luôn sẵn khi triển khai.

---

### B5. Holm–Bonferroni được tuyên bố; họ kiểm không đọc được từ bài

Vị trí: § Credit gate · Loại: `thong-ke` · Công sức: nhỏ

Bản thảo viết: “Holm–Bonferroni at α=0.05 among the remaining XES credit arms leaves the QKC-backbone timing test significant.”

Lỗi là gì: *t*₄ = 23,99, *p* = 1,79×10⁻⁵ trên năm Δval `both` — thầy tính lại được, khớp. Nhưng “remaining XES credit arms” không liệt kê *m*. Attribution summary có bốn phép 5-seed trên XES (incidence, Δ*t* observed, Δ*t* history-only, Δ*t* zero-incidence) cộng P4. Holm trên họ nào? Chỉ các phép đã PASS? Mọi phép 5-seed?

Vì sao là lỗi [Chuẩn mực ngành]: Một hiệu chỉnh đa so sánh mà không nêu *m* thì không tái lập. Demšar (2006) — nguồn vòng trước — đòi hỏi họ được định nghĩa trước.

Cách khắc phục [Chuẩn mực ngành]

1. Một câu: *m* = ?, danh sách phép, thứ tự Holm, p đã chỉnh của phép timing.
2. Hàng FAIL (incidence, history-only, P4) không cần *p* nếu chỉ dùng cổng +0,002 — nói rõ cổng ≠ kiểm định đồng thời.

Tự kiểm: Một độc giả tính lại Holm từ các số đã in, ra cùng kết luận “vẫn significant.”

---

### B6. Bảng attribution đặt bốn hàng 1-seed cạnh hàng 5-seed như cùng một bản đồ ghi công

Vị trí: `table_attribution_summary.tex`; C2 · Loại: `trinh-bay-bang` · Công sức: nhỏ

Bản thảo viết: “HypergraphConv, hint-item HE, full Q-matrix, and the Junyi DAG remain single-seed diagnostics and are not significance tests.”

Lỗi là gì: Chú thích đúng; thân bảng vẫn một cột Credit (FAIL / 0/5 FAIL / 5/5 PASS). Người đọc lướt thấy tám FAIL. C2 đã tách câu. Bảng thì chưa.

Vì sao là lỗi [Suy luận của tôi]: Sau khi bài lấy 5-seed làm chuẩn ghi công, một hàng 1-seed in cùng cột là mời người đọc gộp.

Cách khắc phục [Suy luận của tôi]

1. Tách bảng: khối 5-seed / khối 1-seed. Hoặc cột “seeds” đủ lớn và hàng 1-seed không dùng chữ FAIL cùng kiểu.
2. C2 giữ như hiện tại.

Tự kiểm: Không còn một cột Credit nào gộp 5/5 với FAIL một seed.

---

### B7. Chú thích bảng còn nhãn nội bộ “GS Hau A4” và “Hau B12”

Vị trí: `table_leakage_audit.tex` (in ra PDF, aux trang 12); `table_manipulation.tex` (phụ lục) · Loại: `trinh-bay` · Công sức: nhỏ

Bản thảo viết: “with a full-log member map (GS Hau A4)” và “permutation (Hau B12).”

Lỗi là gì: Đây là mã điểm của bản nhận xét nội bộ, không phải thuật ngữ khoa học. Người phản biện KBS không có sổ cái A4/B12. Aux xác nhận chuỗi này **đi vào PDF**, không chỉ comment TeX (comment dòng 1 của leakage table là chuyện khác — caption thì không).

Vì sao là lỗi [Suy luận của tôi]: Cùng họ với nhãn v1/v2 mà ver3 đã gỡ ở Bảng 3. Rẻ, và để lộ quy trình sửa bài.

Cách khắc phục [Đề xuất mới]

1. Xoá “GS Hau A4” / “Hau B12”; thay bằng “positive control” / “structure checks.”
2. `grep -i hau paper/` phải sạch phần caption.

Tự kiểm: PDF trang 12 và phụ lục manipulation không còn chuỗi “Hau”.

---

### B8. M5 vẫn là mẫu 5.000 người dùng trên hyperedge không thứ tự — FAIL đứng, câu bán hàng phải hẹp

Vị trí: C2; Bảng negative-knowledge hàng M5; Algorithm 1 · Loại: `overclaim` · Công sức: nhỏ

Bản thảo viết: “a Junyi DAG subsample remain single-seed FAIL rows and are not five-seed credit tests.”

Lỗi là gì: Algorithm 1 đã thừa nhận tập không thứ tự — đúng phương án (b) của A5 vòng 3. M5 vẫn 5k, 1 seed. C2 không còn “do not” toàn xưng — tốt. Rủi ro còn lại: một độc giả đọc bảng attribution như bằng chứng “DAG vô dụng.”

Vì sao là lỗi [Suy luận của tôi]: Nếu M5 nạp chain không hướng thì FAIL không nói về tri thức tiên quyết (câu hỏi 5 vòng 3, vẫn mở).

Cách khắc phục [Suy luận của tôi]

1. Chú thích M5: “5k users; path-derived unordered groups; not a directed-DAG test.”
2. Không chạy lại Junyi đầy đủ ở vòng này trừ khi chọn phương án (a).

Tự kiểm: Hàng M5 không đọc được như kiểm định DAG có hướng trên toàn Junyi.

---

## 5. MỨC C — TUỲ CHỌN (3 điểm)

Tuỳ chọn — gợi ý để bản thảo đẹp hơn, không sửa cũng không sao.

### C1. +0,0074 so với AKT là hiệu của số thô; trừ hai ô đã làm tròn ra +0,0073

Vị trí: §7.2; chú thích Bảng 5 · Loại: `hieu-lam-tron` · Công sức: nhỏ

Bản thảo viết: “the five-seed mean gaps are +0.0074 versus AKT.”

Lỗi là gì: 0,82961955 − 0,82225073 = 0,007369 → +0,0074 đúng từ số thô. 0,8296 − 0,8223 = 0,0073. Cùng họ V9 C2.

Cách khắc phục [Suy luận của tôi]: In thêm một chữ số, hoặc viết “+0.0074 from unrounded five-seed means (printed cells subtract to +0.0073).”

Tự kiểm: Độc giả trừ hai ô Bảng 5 ra đúng hiệu mà câu văn nêu, hoặc câu văn nói rõ đang trừ số thô.

---

### C2. Nhãn v2/v4 còn ở Bảng paired và Tier 2

Vị trí: `table_paired_ablation.tex`, `table_tier2.tex`, `table_kc_jaccard_ig.tex` · Loại: `thuat-ngu-khong-nhat-quan` · Công sức: nhỏ

Bản thảo viết: “Grounded + Critic (v2 ckpt)”, “v4 Q←KC+Δt”.

Lỗi là gì: Bảng aliases đã map. Phần exploratory còn song ngữ. Ver3 đã gỡ ở Bảng protocol; làm nốt ở đây.

Cách khắc phục [Đề xuất mới]: Dùng “graph-only diagnostic” / “QKC-T”; giữ v2/v4 trong ngoặc một lần.

Tự kiểm: Lần xuất hiện đầu của v2/v4 đi kèm mô tả kiến trúc.

---

### C3. Pin commit và xem PDF bằng mắt

Vị trí: Data availability; preamble `hidelinks` đã có · Loại: `bo-cuc-lech-chuan` · Công sức: nhỏ

Không tìm thấy trong bản thảo: `hidelinks` đã có trong `main.tex` — điểm C1 vòng 3 về khung hyperlink đã xử lý ở nguồn. Vẫn phải mở PDF thật.

Lỗi là gì: Pin `09e0b96` sẽ sai sau commit Gate 0. Aux hiện 49 trang (ver3: 38) — cần lướt toàn trang.

Cách khắc phục [Đề xuất mới]: Cập nhật pin ở commit khoá; mở trang 1, Bảng 5, References.

Tự kiểm: URL/pin khớp HEAD; PDF không còn khung liên kết.

---

## 6. CÂU HỎI CẦN LÀM RÕ

Những chỗ không kết luận được chỉ từ bản thảo và CSV đã đọc. Trả lời được thì một số điểm trên có thể đổi mức.

1. Các checkpoint A2 của QKC-T (seed 0, 17, 1234, 2024) còn file `.pt` / `.npz` không? Nếu còn, A1 chỉ là chấm lại. Nếu mất, phải train lại — A1 nhảy từ nửa buổi lên 1–2 tuần.
2. Cột `auc` của pyKT trong A2 có chắc là test-only (không lẫn valid) không? *n* = 1.093.720 gợi ý là có; em xác nhận giúp từ script 24.
3. Bảng 3 hàng 3/4 của ver3 — v4q_clean vs p0_dt_on — còn khác `window_mode` (last vs chunked) như đã tìm ngày 05/09 không? Nếu còn, giả thuyết A3 không chỉ là Δ*t*×repeat.
4. M5 nạp DAG Junyi như cạnh đôi có hướng hay như chain Algorithm 1? (câu 5 vòng 3, chưa đóng.)
5. Kho `edu-risk-lab/DH2A-KT` có bản ẩn danh cho phản biện chưa, hay chỉ chưa push?
6. `09e0b96` có phải HEAD tại thời điểm khoá PDF không, hay là hash tại một lần sửa Data availability?
7. Likert *n* = 100, 3 rater: có phải người ngoài nhóm và có làm mù nguồn không? Nếu có, một câu là đủ — α = 0,52 sẽ đọc khác.
8. arXiv:2606.21832 (AgentCAT) — thầy không tra Crossref trong lượt này. Em xác nhận DOI trước khi nộp.

---

## 7. NHỮNG PHẦN ĐANG LÀM TỐT — GIỮ NGUYÊN

Ghi ra đây để lúc sửa A1–A3 không viết lại rồi mất.

✓ Đổi trục sang attribution, hạ Tier 2 khỏi C3/tiêu đề — đúng hướng B15/B14 vòng 3. Câu “which knowledge survives” đọc được trong một hơi.

✓ Bảng 5 đưa mọi đối thủ lên 5 seed, bỏ “single canonical run” — điểm A2 vòng 3 đã được xử lý *về ý*. Giữ khung này; chỉ thay cột số (A1).

✓ Capacity-matched 0/5 và Zero+Δt 5/5 là cặp kết quả mạnh và hiếm. Script 34 *đã* tách `combined_eval_auc` / `test_only_auc` — đó là đúng kỷ luật; Bảng 5 phải học đúng script này, không phải ngược lại.

✓ Calibration ECE/Brier/NLL + reliability; không ECE nào > 0,05. A4 vòng 3 đóng. Giữ; chỉ đồng bộ *n* với A1.

✓ “pre-specified” + neo +0,002 bằng SD 5 seed (0,00015 / 0,00023 → khoảng 9–13×). B5 vòng 3 đóng phần thuật ngữ.

✓ pyKT có trong bibliography và § clean protocol; 5.549.635 tách khỏi 6.413.353; “cannot collapse” đã bỏ, thừa nhận *W* ràng buộc.

✓ Đối chứng dương membership = 1,0 (Bảng leakage). `|ρ|` = 0 ở cả hai cột được khai đúng là không validate pairwise. Rewiring 3 fold trên encoder chẩn đoán, không gán cho QKC-T.

✓ Bảng negative-knowledge giữ FAIL, kể cả HypergraphConv. FAIL = không ghi công, không = vô giá trị — câu này giữ.

✓ History-only 2/5 và query-only 0/5 được in, không bị rút. Đó là phần trung thực nhất của RQ2 — hãy để nó chỉ huy Abstract (B4), đừng làm nhẹ.

✓ hidelinks đã có; aliases đã có; ethics + generative-AI statements đã có.

✓ Toàn bộ *t*₄ = 23,99, trung bình Δval, PASS 5/5 và 0/5, trung bình 3 fold — thầy tính lại, khớp CSV. A1 là van metric, không phải bài đếm sai hàng loạt. Giống ver3: một chỗ hỏng hệ thống, không phải xu hướng số liệu bừa.

---

## 8. NGUỒN ĐÃ VIỆN DẪN

Các chuẩn mực và file được nhắc ở trên đều tra được tại đây.

Liu, Z., et al. (2022). pyKT. NeurIPS 2022. https://arxiv.org/abs/2206.11460 — repeat format; nguồn siêu tham số simpleKT.

Liu, Z., et al. (2023). XES3G5M. NeurIPS 2023. 5.549.635 lượt thử gốc.

Bouthillier, X., et al. (2021). Accounting for Variance in Machine Learning Benchmarks. MLSys. https://arxiv.org/abs/2103.03098

Pineau, J., et al. (2021). Improving Reproducibility in Machine Learning Research. JMLR 22(164). https://www.jmlr.org/papers/v22/20-303.html

Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7:1–30.

Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML. https://arxiv.org/abs/1706.04599

Elsevier — Guide for Authors, Knowledge-Based Systems. https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors

Nội bộ, đã mở khi soạn bản này:

- `docs/NhanXet_DH2A-KT_ver3_Tuan_05Sep2026.pdf` — sổ cái vòng 3
- `scripts/34_qkc_capacity_matched.py` dòng 289–292 — `dh2_kt_auc` ≡ `combined_eval_auc`
- `scripts/27_a2_multiseed_matrix.py` dòng 199–210 — gán `test_auc = dh2_kt_auc`
- `results/tables/a2_dh2_p0_dt_on_s42.csv` — *n* = 1.640.242
- `results/tables/a6_calibration_xes3g5m_fold0.csv` — test-only seed 42 = 0,829593, *n* = 1.093.755
- `results/tables/p0_dt_testonly_clean.csv` — residual +0,010128
- `https://github.com/edu-risk-lab/DH2A-KT` — HTTP 404 ngày 07/09/2026
