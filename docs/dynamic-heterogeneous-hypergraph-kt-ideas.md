# Ý tưởng bản thảo: Dynamic Heterogeneous Hypergraph Knowledge Tracing (DH²-KT)

## 1. Bối cảnh và khoảng trống nghiên cứu

Dòng nghiên cứu Knowledge Tracing (KT) đã đi qua ba làn sóng: (1) mô hình xác suất/chuỗi (BKT, DKT, SAKT), (2) mô hình đồ thị tĩnh (GKT, SGKT dùng graph Student–Skill–Exercise), và hiện nay (3) mô hình nhận thức sâu kết hợp LLM, Agentic AI, GNN, đa phương thức, và Causal AI. Khảo sát cho thấy các khoảng trống cụ thể mà hướng "Dynamic Graph KT" của bạn có thể khai thác:

- **Đồng nhất hoá quan hệ bậc cao bị bỏ sót**: các mô hình graph KT phổ biến (SGKT, HHSKT, Psy-KT) chỉ dùng đồ thị nhị phân (pairwise) Student–Exercise–Concept, không biểu diễn được quan hệ nhóm (một buổi học, một chủ đề thảo luận, một video bài giảng liên quan nhiều concept cùng lúc). Các công trình hypergraph gần đây (THMN – AIED 2025, HGKT – *Applied Sciences* 2025, HTKT – IEEE 2025) mới dừng ở hyperedge Exercise–Concept, **chưa đưa Teacher, Forum, Discussion, Video, Hint vào làm node**.
- **Tĩnh về topology**: phần lớn graph KT coi cấu trúc quan hệ là cố định theo thời gian. DyGKT (arXiv 2407.20824, 2024) là công trình hiếm hoi mô hình hoá KT như continuous-time dynamic graph, nhưng vẫn giới hạn ở cặp Student–Exercise, chưa mở rộng sang đồ thị dị thể (heterogeneous) đa loại node/quan hệ.
- **Thiếu tác nhân xã hội/giảng dạy**: Teacher, Forum/Discussion là nguồn tín hiệu can thiệp sư phạm (feedback, gợi ý, thảo luận bạn học) gần như vắng mặt trong graph KT hiện tại — trong khi các mô hình "student cognitive modeling" gần đây (Psy-KT) đã chỉ ra yếu tố tâm lý/tương tác xã hội cải thiện đáng kể độ chính xác.
- **Đa phương thức chưa gắn với graph**: benchmark ES-KT-24 (2024) và các nghiên cứu dùng LMM trích knowledge components từ video/hình ảnh (EDM 2025) cho thấy dữ liệu đa phương thức (video, hint text) đã sẵn sàng, nhưng chưa có mô hình hypergraph nào dùng chúng làm node biểu diễn trực tiếp.
- **Thiếu suy luận nhân quả**: các framework causal-KG hiện nay (C-KHG) tách biệt khỏi giáo dục; graph KT gần như không phân biệt được "tương quan" (student làm đúng vì đã xem video) với "nhân quả" (xem video → cải thiện knowledge state), dẫn đến rủi ro can thiệp sư phạm sai (gợi ý hint không thực sự giúp ích nhưng model vẫn học được tương quan giả).

=> Đây là **khoảng trống hội tụ**: chưa có mô hình nào kết hợp đồng thời (a) hypergraph dị thể đa loại node giáo dục, (b) tính động theo thời gian thực (dynamic/temporal), (c) tác nhân LLM/Agentic để sinh và cập nhật hyperedge, và (d) suy luận nhân quả cho can thiệp sư phạm.

## 2. Câu hỏi nghiên cứu đề xuất

- RQ1: Biểu diễn hypergraph dị thể (Student, Exercise, Concept, Teacher, Hint, Forum, Discussion, Video là node) cải thiện độ chính xác dự đoán knowledge state so với graph nhị phân/đồng nhất như thế nào, và cải thiện đến từ loại quan hệ bậc cao nào?
- RQ2: Cấu trúc đồ thị nên tiến hoá theo thời gian ra sao (continuous-time vs. discrete snapshot) để phản ánh đúng nhịp học (một buổi học, một thread thảo luận, một lần xem lại video) mà không gây "quên" thông tin quá nhanh hoặc quá chậm?
- RQ3: Tác nhân LLM (agentic) đóng vai trò gì khi được đưa vào graph — như một node sinh hyperedge mới (ví dụ: sinh gợi ý/hint), hay như một cơ chế cập nhật trọng số cạnh dựa trên ngữ nghĩa nội dung forum/video?
- RQ4: Có thể tách bạch quan hệ nhân quả (giáo viên can thiệp → cải thiện knowledge state) khỏi quan hệ tương quan giả trong đồ thị động hay không, và điều này ảnh hưởng thế nào đến tính khả diễn giải (interpretability) của mô hình?

## 3. Kiến trúc đề xuất: DH²-KT

### 3.1 Tầng node (Heterogeneous nodes)

| Loại node | Đặc trưng đầu vào | Vai trò |
|---|---|---|
| Student | lịch sử tương tác, đặc trưng hành vi | chủ thể cần theo dõi trạng thái tri thức |
| Exercise | text câu hỏi, độ khó | đơn vị đánh giá |
| Concept | quan hệ tiên quyết (prerequisite graph) | đơn vị tri thức |
| Teacher | hành động can thiệp (feedback, chấm điểm, gợi ý) | tác nhân sư phạm |
| Hint | nội dung gợi ý, thời điểm cấp | tín hiệu hỗ trợ có kiểm soát |
| Forum/Discussion | nội dung bài đăng, thread, người tham gia | tương tác xã hội/bạn học |
| Video | embedding đa phương thức (visual + transcript) | tài nguyên học liệu |

### 3.2 Tầng hyperedge (quan hệ bậc cao, biến đổi theo thời gian)

- **Session hyperedge**: nhóm {Student, Exercise, Hint, Video} xuất hiện cùng một phiên học — nắm bắt ngữ cảnh học tập đồng thời thay vì chuỗi cặp rời rạc.
- **Concept-prerequisite hyperedge**: nhóm nhiều Concept có quan hệ tiên quyết, cho phép lan truyền tín hiệu "quên/thành thạo" qua toàn bộ chuỗi concept liên quan một lần thay vì từng cặp.
- **Discussion-thread hyperedge**: nhóm {Student, Forum post, Concept, đôi khi Teacher} — mô hình hoá học tập xã hội (peer learning).
- **Teacher-intervention hyperedge**: nhóm {Teacher, Student, Exercise/Hint} tại thời điểm can thiệp — tách riêng để phục vụ tầng nhân quả (mục 3.4).

Hyperedge được **sinh động** (không cố định): dùng continuous-time dynamic hypergraph (lấy cảm hứng từ DyGKT, mở rộng sang hypergraph) — mỗi sự kiện tương tác tạo/kích hoạt lại hyperedge với timestamp, dùng temporal point process hoặc memory-based update (như THMN) để lan truyền tín hiệu.

### 3.3 Tầng mã hoá (Encoding layer)

- Text (câu hỏi, forum, hint): LLM embedding (frozen hoặc fine-tune nhẹ).
- Video: vision-language embedding (theo hướng ES-KT-24/EDM 2025 dùng LMM trích knowledge component từ multimedia).
- Lan truyền: Heterogeneous Hypergraph Attention (mở rộng HTKT sang dị thể) + cơ chế cổng thời gian (dual-gated, theo HGKT) để cân bằng tín hiệu cũ/mới.

### 3.4 Tầng nhân quả (Causal layer) — điểm khác biệt cốt lõi

- Xây **causal subgraph** riêng cho Teacher-intervention và Hint hyperedge: dùng kỹ thuật ước lượng hiệu ứng can thiệp (ví dụ: causal effect estimation trên đồ thị động, hoặc LLM-assisted causal discovery như hướng C-KHG) để phân biệt "hint có tác dụng thật" khỏi "hint chỉ tương quan với học sinh giỏi hơn".
- Đầu ra phụ: điểm số nhân quả cho mỗi loại can thiệp — hỗ trợ diễn giải & gợi ý sư phạm actionable, không chỉ dự đoán đúng/sai.

### 3.5 Tầng tác nhân (Agentic loop, tuỳ chọn mở rộng)

- LLM agent quan sát trạng thái tri thức hiện tại → đề xuất hint/exercise tiếp theo → hành động này tạo hyperedge mới → vòng lặp đóng (closed-loop) giữa dự đoán và can thiệp, biến DH²-KT từ mô hình dự đoán thuần tuý thành hệ thống hỗ trợ quyết định sư phạm.

## 4. So sánh với các công trình liên quan (định vị đóng góp)

| Công trình | Hypergraph | Dị thể (Teacher/Forum/Video) | Động theo thời gian | Nhân quả |
|---|---|---|---|---|
| SGKT / HHSKT / Psy-KT | Không | Một phần (Student-Skill-Exercise) | Không | Không |
| THMN (AIED 2025) | Có | Không | Có (temporal memory) | Không |
| HGKT (2025) | Có | Không | Có (dual-gated) | Không |
| HTKT (IEEE 2025) | Có | Không | Hạn chế | Không |
| DyGKT (2024) | Không | Không | Có (continuous-time) | Không |
| **DH²-KT (đề xuất)** | **Có** | **Có (Teacher, Hint, Forum, Video)** | **Có** | **Có** |

=> Đóng góp chính không nằm ở việc thêm một kỹ thuật đơn lẻ, mà ở **sự hội tụ của 4 trục** mà từng công trình hiện tại chỉ giải quyết riêng lẻ.

## 5. Phương án thực nghiệm

- **Dữ liệu**: ASSISTments, EdNet, Junyi (chuẩn KT); bổ sung ES-KT-24 (đa phương thức có video) để kiểm chứng node Video; cần thu thập/ghép thêm dữ liệu forum (ví dụ MOOC forum log công khai) nếu muốn kiểm chứng đầy đủ node Forum/Discussion — đây là điểm cần làm rõ ngay từ đầu vì phần lớn benchmark KT public **không có** dữ liệu Teacher/Forum đồng bộ với log bài tập.
- **Baseline**: DKT, SAKT, GKT/SGKT, DyGKT, THMN, HGKT, HTKT.
- **Metric**: AUC/ACC cho dự đoán chuẩn; thêm metric diễn giải (attention/causal attribution) và metric ổn định theo thời gian (dùng temporal-aware evaluation, theo hướng arXiv 2412.07273) để tránh leakage thời gian thường gặp khi đánh giá dynamic graph.
- **Ablation bắt buộc**: (a) bỏ từng loại node để đo đóng góp riêng; (b) hypergraph vs. graph nhị phân cùng dữ liệu; (c) static vs. dynamic topology; (d) có/không tầng nhân quả.

## 6. Rủi ro/hạn chế cần lường trước

- Thiếu benchmark công khai có đủ Teacher + Forum + Video đồng bộ — có thể phải giới hạn phạm vi thực nghiệm ban đầu (ví dụ chỉ Student-Exercise-Concept-Hint-Video) rồi mở rộng Teacher/Forum ở phần discussion là hướng tương lai, thay vì claim đã kiểm chứng đầy đủ 8 loại node.
- Causal claim trên dữ liệu quan sát (observational) dễ bị phản biện — cần nêu rõ giả định nhận diện (identification assumptions) hoặc dùng thiết kế bán thực nghiệm nếu có.
- Độ phức tạp tính toán của hypergraph động dị thể lớn — nên có mục phân tích scalability/độ trễ.

## 7. Venue tiềm năng

AIED, EDM, LAK (giáo dục + AI), hoặc track ứng dụng của KDD/WWW/IJCAI nếu nhấn mạnh phần dynamic hypergraph + causal như đóng góp kỹ thuật tổng quát.

## Nguồn tham khảo đã kiểm tra khi xây ý tưởng

- [Knowledge Tracing with A Temporal Hypergraph Memory Network (AIED 2025)](https://link.springer.com/content/pdf/10.1007/978-3-031-99267-4_10.pdf)
- [Hypergraph-Driven High-Order Knowledge Tracing with a Dual-Gated Dynamic Mechanism](https://doi.org/10.3390/app15158617)
- [HTKT: Knowledge Tracing Based on Hypergraph Transformer (IEEE)](https://ieeexplore.ieee.org/document/10885089/)
- [DyGKT: Dynamic Graph Learning for Knowledge Tracing (arXiv 2407.20824)](https://arxiv.org/pdf/2407.20824)
- [Psychological factors enhanced heterogeneous learning interactive graph knowledge tracing (Frontiers in Psychology, 2024)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1359199/full)
- [HHSKT: A learner–question interactions based heterogeneous graph neural network model for KT](https://www.sciencedirect.com/science/article/abs/pii/S0957417422023521)
- [Knowledge relation rank enhanced heterogeneous learning interaction modeling for neural graph forgetting KT](https://arxiv.org/pdf/2304.03945)
- [A survey of dynamic graph neural networks (Frontiers of Computer Science, 2025)](https://journal.hep.com.cn/fcs/EN/PDF/10.1007/s11704-024-3853-2)
- [ES-KT-24: A Multimodal Knowledge Tracing Benchmark Dataset](https://arxiv.org/pdf/2409.10244)
- [Using Large Multimodal Models to Extract Knowledge Components for KT from Multimedia (EDM 2025)](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.170/)
- [LLM-augmented causal-knowledge heterogeneous graph framework (C-KHG)](https://www.sciencedirect.com/science/article/abs/pii/S0957417426002563)
- [Temporal-Aware Evaluation and Learning for Temporal Graph Neural Networks](https://arxiv.org/pdf/2412.07273)
