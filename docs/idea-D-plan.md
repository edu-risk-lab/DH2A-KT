# Ý tưởng bản thảo D: DH²A-KT+P0 — Hybrid Two-Tier Hypergraph KT xây trên nền Leakage-Controlled Audit Protocol (P0)

> D = **Phiên bản C giữ nguyên kiến trúc hai tầng** (Tier 1 DH²-KT định lượng + Tier 2 AgentDG-KT pilot diễn giải) **+ tái sử dụng có hệ thống artefact, protocol và baseline đã kiểm chứng của bài P0** (*Leakage-Controlled Concept Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing*, Dao Minh Tuan et al., APIN). Không sửa C — D là một nhánh riêng, khai thác P0 làm nền móng leakage-control cho Tier 1 thay vì tự xây từ đầu.

## 1. Vì sao dùng P0 làm nền

P0 không phải một kiến trúc KT mới — nó là một **audit protocol** cho việc xây concept graph không rò rỉ (train-only construction, DAG audit, 4 chỉ số leakage, cold-start stratification, DDR, ground-truth cross-validation), đã kiểm chứng trên XES3G5M, ASSISTments 2012, Junyi Academy với baseline đầy đủ (BKT, DKT, AKT, simpleKT, GKT, GIKT, SKT, DyGKT, DGEKT). Đây đúng là phần hạ tầng mà Tier 1 của C cần nhưng chưa có lời giải cụ thể (leakage theo thời gian, cold-start, đối chiếu chuyên gia). Tái sử dụng nó giúp D: (a) tiết kiệm compute lớn nhất trên 1x RTX 3090 — không phải train lại baseline; (b) có sẵn bằng chứng "leakage control không tự sinh ra gain lớn" (≤0.003 AUC trên 3 benchmark công khai) để phòng thủ trước phản biện; (c) có sẵn split protocol, cold-start strata, và ground-truth cross-validation đã kiểm chứng.

## 2. Đóng góp mới của D so với P0 — Hyperedge-level Leakage Taxonomy

P0 chỉ audit quan hệ **cặp đôi KC–KC** (prerequisite, similarity), một loại node duy nhất (Table 1 của P0: 5 lớp leakage — Structural, Edge, Target, Temporal, Cold-start neighbourhood). D mở rộng taxonomy này sang **hyperedge dị thể nhiều loại node** — đây là đóng góp phương pháp luận độc lập của D, không chỉ là trích dẫn P0:

| Lớp leakage mới (D bổ sung) | Ví dụ cụ thể | Vì sao P0 chưa bao phủ |
|---|---|---|
| **Group-membership leakage** | Session hyperedge {Student, Exercise, Hint, Video} chứa một thành viên thuộc tương tác held-out, dù các thành viên khác thuộc train | P0 chỉ kiểm tra overlap ở cấp cạnh đôi, không có khái niệm "nhóm" nhiều thành viên |
| **Cross-modal leakage** | Embedding forum-thread được tính trên toàn bộ thread (gồm cả bài đăng tương lai) rồi mới cắt theo thời điểm dự đoán | P0 không có node Forum/Video nên không cần xử lý rò rỉ liên phương thức |
| **Intervention-timing leakage** | Teacher-intervention hyperedge gán nội dung hint dựa trên kết quả đúng/sai *sau đó* của học sinh (rò rỉ nhãn tương lai vào đặc trưng can thiệp) | P0 không mô hình hoá hành động giáo viên/hint như một thực thể có thời điểm riêng |

=> D áp dụng lại đúng 4 chỉ số của P0 (ECR^flag, ECR^overlap, \|ρ\|, TBMR — Thuật toán 1) nhưng tính trên **từng loại hyperedge** thay vì chỉ trên `E_pre`, và bổ sung 3 lớp leakage trên vào bảng audit.

## 3. Câu hỏi nghiên cứu (kế thừa C, bổ sung RQ5)

- RQ1–RQ4: giữ nguyên như Phiên bản C (Tier 1 định lượng: hypergraph dị thể vs baseline; tầng nhân quả; Tier 2 định tính: faithfulness + đánh giá con người).
- **RQ5 (mới)**: Bốn chỉ số leakage của P0 — vốn thiết kế cho graph cặp đôi — có mở rộng được sang hyperedge dị thể mà vẫn giữ tính chặt chẽ (rigor) không, và những lớp leakage nào chỉ xuất hiện khi thêm node Teacher/Forum/Video (Group-membership, Cross-modal, Intervention-timing)?

## 4. Kiến trúc hai tầng (không đổi so với C, chú thích rõ phần kế thừa P0)

```
        TIER 1 -- DH2-KT (loi, huan luyen day du)
        - Concept-prerequisite hyperedge: GOM NHOM tu E_pre da audit cua P0
          (Thuat toan 2 cua P0 tai su dung nguyen, khong viet lai)
        - Session / Discussion-thread / Teacher-intervention hyperedge:
          MOI, audit bang 4 chi so P0 + 3 lop leakage moi (muc 2)
        - Causal-effect layer: moi (P0 khong co)
                    |
                    v (dong bang sau khi train xong)
        TIER 2 -- Lop dien giai Agentic (pilot, giu nguyen nhu C)
```

## 5. Phạm vi dữ liệu — cập nhật theo P0

| Nguồn | Vai trò trong D | Ghi chú |
|---|---|---|
| **XES3G5M** | **Primary** (đổi từ ASSISTments2012/EdNet ở bản C) | Vai trò primary giống hệt P0: KC-repeat thấp (~21%), AUC không bão hòa, phân tách backbone rõ nhất (simpleKT≈0.875 vs GKT≈0.834) — điều kiện tốt nhất để chứng minh hypergraph có tác dụng thật |
| ASSISTments 2012 | Secondary | Q-matrix single-skill → `E_sim` rỗng (đã biết từ P0), dùng để test leakage ablation |
| Junyi Academy | Ground-truth cross-validation + cold-start | Có expert DAG (Chang et al.) — dùng lại thủ tục đối chiếu edge/direction/reachability/node-Jaccard của P0 (Section 3.8) cho concept-prerequisite hyperedge của D |
| ES-KT-24 | Node Video (P0 không có) | Vẫn phải tự xử lý — P0 không cung cấp |
| Teacher/Forum/Discussion | Không có trong P0 lẫn benchmark công khai | Giữ nguyên hướng xử lý của C: giới hạn Tier 1 quantitative core, đưa vào Tier 2 case study |

## 6. Bảng tái sử dụng artefact P0 cụ thể

| Artefact P0 | Vị trí trong P0 | Cách D dùng lại |
|---|---|---|
| Code + graph export | `data/graphs/<dataset>/fold_<f>/`, repo `github.com/edu-risk-lab/leakage-controlled-kt-audit` | Lấy trực tiếp `e_pre_train_only.csv`, `e_sim_train_only.csv` đã audit sạch làm input cho bước gom hyperedge |
| Thuật toán 2 (xây `E_pre` + DAG audit) | Section 3.4–3.5 | Dùng nguyên, không viết lại |
| Thuật toán 1 (4 chỉ số leakage) | Section 3.2 | Mở rộng sang hyperedge (mục 2) |
| Split protocol | Section 4.1 — learner-based 0.7/0.1/0.2, seed 42/43/44 | Dùng nguyên làm protocol chuẩn của D |
| Baseline AUC (Table 8, 9) | BKT/DKT/AKT/simpleKT/GKT/GIKT/SKT/DyGKT/DGEKT trên XES3G5M/ASSIST2012/Junyi | Dùng làm cột so sánh cho RQ1 — **không train lại trên RTX 3090** |
| Cold-start KC strata | Section 3.7 — S1<20, S2 20–100, S3 100–500, S4≥500 | Dùng nguyên cho ablation cold-start |
| DDR + manipulation-check | Section 3.6, 4.7 | Làm khuôn cho phép thử "model có thực sự đọc hyperedge không" ở Pha 3 |
| Ground-truth cross-validation | Section 3.8, 4.9 | Mở rộng cho concept-prerequisite hyperedge trên Junyi |
| Phát hiện train-only vs full-log ΔAUC≤0.003 | Section 4.5 | Trích dẫn làm bằng chứng: gain của D (nếu có) không phải do rò rỉ |

## 7. Ngăn xếp công cụ (bổ sung so với C)

Giữ nguyên stack của C (PyG HypergraphConv/HeteroConv, sentence-embedding nhỏ, CLIP offline, LLM local Qwen3 8B quant cho Tier 2), **thêm**:

- Clone repo P0 (`leakage-controlled-kt-audit`) làm điểm khởi đầu thay vì viết pipeline tiền xử lý từ số 0.
- Viết lớp "hyperedge audit wrapper" gọi lại 4 hàm chẩn đoán của P0 (ECR^flag, ECR^overlap, \|ρ\|, TBMR) trên từng tập hyperedge mới — đây là code mới, không có sẵn trong P0.

## 8. Kế hoạch thực hiện theo pha (khung ~15 tuần, rút ngắn so với C nhờ tái sử dụng)

| Pha | Tuần | Nội dung chính | Deliverable |
|---|---|---|---|
| 0 — Kế thừa & thiết kế audit mở rộng | 1–2 | Lấy code/artefact P0; verify preprocessing khớp trên XES3G5M/ASSISTments2012/Junyi; thiết kế 3 lớp leakage mới (mục 2) và viết wrapper audit hyperedge | Bộ audit hyperedge mở rộng + báo cáo verify khớp P0 |
| 1 — Xây hyperedge trên nền P0 | 3–5 | Gom `E_pre` đã audit thành concept-prerequisite hyperedge; xây session hyperedge từ log gốc; chạy audit mở rộng trên từng loại hyperedge | Graph dị thể Student-Exercise-Concept-Hint(-Video) đã audit sạch |
| 2 — Encoder + causal layer | 5–7 | Text/video embedding offline; cài HypergraphConv/HeteroConv; causal-effect layer (propensity/2-stage) | DH²-KT chạy end-to-end |
| 3 — Thực nghiệm định lượng đối chiếu P0 | 8–10 | Train DH²-KT trên XES3G5M dưới ngân sách epoch/batch khớp P0 (Table S15); so trực tiếp với Table 8/9 của P0 (không train lại baseline); ablation node/hyperedge/dynamic/causal; manipulation-check kiểu DDR; ground-truth cross-validation mở rộng trên Junyi | Bảng kết quả đối chiếu trực tiếp P0 + kết quả manipulation-check |
| 4 — Xây Tier 2 (pilot) | 11–12 | Giống C: Diagnostician-wrapper, Tutor/Hint, Critic agent đọc output Tier 1 đóng băng | Agent pipeline chạy trên tập mẫu |
| 5 — Đánh giá Tier 2 | 12–13 | Giống C: hallucination rate, Critic override rate, đường cong quên, đánh giá con người (nếu có điều kiện) | Bảng metric Tier 2 + case study |
| 6 — Tổng hợp & viết | 13–15 | Viết rõ phần "kế thừa P0" như nền tảng phương pháp luận; Scope & Limitations; rà soát trích dẫn/số liệu | Bản thảo hoàn chỉnh |

So với C (16 tuần): Pha 0 ngắn hơn (kế thừa thay vì xây từ đầu) và Pha 3 ngắn hơn (không train lại baseline), bù lại có thêm việc mở rộng audit — tổng thể tiết kiệm khoảng 1–2 tuần, nhưng lợi ích chính không phải ở số tuần mà ở **độ tin cậy đã kiểm chứng** của phần leakage-control.

## 9. Ngân sách compute & điểm kiểm tra

Giữ nguyên nguyên tắc của C (đo epoch time sau Pha 1, đo throughput LLM trước Pha 4, ghi log GPU mỗi thí nghiệm), **bổ sung**: trước Pha 3, đối chiếu ngân sách epoch/batch dự kiến của D với ngân sách P0 đã công bố (GKT: max 10 epoch, batch 4; sequence checkpoint: max 30 epoch, batch 64) — nếu không khớp được đúng ngân sách, phải nêu rõ đây là so sánh "quan sát" (observational) chứ không phải so sánh khớp compute, đúng cách P0 tự giới hạn phát biểu của họ.

## 10. Bảng rủi ro & giảm thiểu (bổ sung so với C)

| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| Preprocessing của D không khớp chính xác với P0 (khác hashed `kc_id`, khác Q-matrix) → baseline reuse không hợp lệ | Cao | Bắt buộc lấy code từ repo P0, không tự viết lại "một bản XES3G5M khác"; verify ở Pha 0 |
| Mở rộng audit sang hyperedge phức tạp hơn dự kiến, làm chậm Pha 0–1 | Trung bình | Ưu tiên concept-prerequisite hyperedge (kế thừa nhiều nhất từ P0) trước, session/discussion-thread hyperedge có thể lùi sang Tier 2 case study nếu hết ngân sách |
| So sánh ngân sách epoch/batch với P0 không khớp được | Trung bình | Nêu rõ là quan sát (observational), không phải causal — đúng cách P0 đã tự giới hạn |
| Các rủi ro còn lại (thiếu Teacher/Forum, causal claim trên dữ liệu quan sát, LLM local chậm, thiếu metric hallucination chuẩn) | — | Giữ nguyên như bảng rủi ro của C |

## 11. Kế hoạch đánh giá

Giữ nguyên khung 2 tầng của C (Tier 1 định lượng bắt buộc / Tier 2 định tính phạm vi pilot), **bổ sung cho Tier 1**: ground-truth cross-validation (edge/direction/reachability/node-Jaccard F1) trên Junyi cho concept-prerequisite hyperedge, tái dùng thủ tục Section 3.8/4.9 của P0; và một manipulation-check kiểu DDR (phá hủy gần như toàn bộ hyperedge ở mức nhiễu cao) để chứng minh DH²-KT thực sự dựa vào hypergraph chứ không graph-inert.

## 12. Venue & mốc nộp bài

Có thể định vị D như một **companion paper** của P0: P0 (APIN) là protocol/audit layer; D (AIED/EDM/LAK) là kiến trúc KT mới được xây và kiểm chứng leakage bằng chính protocol đó — một mô hình trích dẫn tự nhiên (D cite P0), giúp cả hai bài củng cố lẫn nhau thay vì cạnh tranh phạm vi.

## Nguồn tham khảo

- Dao Minh Tuan et al., *Leakage-Controlled Concept Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing* (P0, APIN submission) — nguồn chính cho toàn bộ mục 6.
- Toàn bộ nguồn tham khảo của Phiên bản A, B, C (xem `dynamic-graph-kt-ideas-A-B.docx` và `dynamic-graph-kt-idea-C-hybrid-plan.docx`) vẫn áp dụng cho phần kiến trúc không đổi.
