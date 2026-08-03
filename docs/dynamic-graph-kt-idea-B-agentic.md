# Phiên bản B: AgentDG-KT — Agentic Dynamic Graph Knowledge Tracing

> Khác biệt cốt lõi so với **Phiên bản A (DH²-KT)**: A coi đồ thị dị thể là *input biểu diễn* cho một mạng GNN end-to-end (học biểu diễn → dự đoán, có thêm tầng nhân quả). B coi đồ thị dị thể là **bộ nhớ chung sống động (shared external memory)** mà một đội **LLM agent** đọc/ghi/suy luận trên đó — chuyển KT từ "bài toán dự đoán có một mô hình khả vi" sang **"hệ thống mô hình nhận thức đa tác nhân"**, bám sát đúng xu hướng bạn nêu ban đầu (cognitive modeling + LLM + Agentic AI).

## 1. Căn cứ từ tài liệu gần đây

- **LOOM** (Cui et al., 2025): learner memory graph ánh xạ concept + quan hệ tiên quyết để sinh lộ trình học cá nhân hoá — xác nhận hướng "graph-as-memory" cho agent giáo dục đã khả thi.
- **Agent4Edu** (Gao et al., 2025): agent mô phỏng đường cong quên Ebbinghaus để huấn luyện giáo viên — gợi ý cơ chế cập nhật trạng thái tri thức theo thời gian có thể do agent điều khiển thay vì thuần công thức decay cố định.
- Multi-agent KT framework với 3 vai trò **administrator – judger – critic**: judger suy luận trạng thái nhận thức qua thảo luận, critic phản biện — mẫu hình trực tiếp cho tầng kiểm soát chất lượng của B.
- **AgentCAT**: mô phỏng Computerized Adaptive Testing bằng multi-agent LLM — cơ sở cho "Tutor/Hint Agent" chọn câu hỏi/gợi ý kế tiếp.
- **Graph-Augmented LLM Agents** (arXiv 2507.21407) và **GraphRAG-Induced Dual Knowledge Structure Graphs cho Personalized Learning Path** (arXiv 2506.22303), **Generative GraphRAG / Auto-HKG + CG-RAG**: xác nhận cộng đồng nghiên cứu đang tích cực ghép GraphRAG với đồ thị tri thức giáo dục — nhưng chưa ai ghép với **hypergraph dị thể động có Teacher/Forum/Video** như node.

=> B không phải suy diễn chủ quan mà là tổng hợp có cơ sở của 3 dòng đang tách rời: (i) graph memory cho LLM agent, (ii) multi-agent cognitive diagnosis, (iii) GraphRAG cho giáo dục.

## 2. Kiến trúc đề xuất

### 2.1 Đồ thị vẫn giữ 8 loại node như bản A

Student, Exercise, Concept, Teacher, Hint, Forum, Discussion, Video — nhưng ở B, mỗi node là một **memory object** có thể được agent đọc/ghi (không chỉ là vector đặc trưng tĩnh cho GNN).

### 2.2 Đội agent (mỗi agent thao tác trên một lát cắt của đồ thị)

| Agent | Đọc/ghi trên | Vai trò |
|---|---|---|
| **Diagnostician Agent** | subgraph Student–Concept–Exercise | Cập nhật trạng thái tri thức sau mỗi tương tác; mô phỏng quên (kiểu Agent4Edu) bằng cách suy luận + cập nhật trọng số cạnh, không chỉ công thức cố định |
| **Tutor/Hint Agent** | Hint, Video, Exercise | Chọn/sinh gợi ý hoặc câu hỏi kế tiếp (kiểu AgentCAT); **ghi hyperedge mới** vào đồ thị — đây là vòng lặp agentic đóng |
| **Forum/Social Agent** | Forum, Discussion, Student | Trích tín hiệu ảnh hưởng bạn học (ai giúp ai hiểu concept nào), ghi social-influence edge động |
| **Critic/Judger Agent** | toàn bộ output của Diagnostician | Phản biện, phát hiện trạng thái tri thức bị "trôi" hoặc ảo giác (hallucination) trước khi ghi chính thức vào graph — cơ chế kiểm soát chất lượng bắt buộc, không tuỳ chọn |
| **Curriculum Planner Agent** | toàn bộ đồ thị quần thể (cross-student) | Retrieval kiểu GraphRAG/community-aware (như EDU-GraphRAG) để gợi ý lộ trình học, không chỉ cho 1 học sinh mà học chéo từ pattern quần thể |

### 2.3 Cơ chế "động" (dynamic) khác với bản A

Ở A, tính động đến từ temporal point process / memory network cập nhật embedding liên tục theo thời gian (kiểu DyGKT/THMN). Ở B, tính động đến từ **hành động của agent**: mỗi lần Tutor Agent cấp hint, hoặc Forum Agent phát hiện một thread mới, đồ thị được cập nhật cấu trúc (thêm/xoá node-edge) một cách tường minh, có thể diễn giải bằng ngôn ngữ tự nhiên (agent ghi lý do cập nhật) — đánh đổi giữa khả năng diễn giải cao hơn và chi phí suy luận (nhiều lệnh gọi LLM) cao hơn so với A.

### 2.4 Suy luận nhân quả (tuỳ chọn, do Critic Agent thực hiện)

Thay vì một tầng nhân quả cố định trong pipeline (như A), ở B đây là một **năng lực agent**: Critic Agent có thể chủ động đặt câu hỏi phản thực ("nếu không có hint này, trạng thái có tiến bộ không?") bằng cách truy vấn lịch sử đồ thị, tận dụng hướng LLM-assisted causal discovery — linh hoạt hơn nhưng khó kiểm định độ tin cậy hơn một mô-đun causal huấn luyện chuyên biệt.

## 3. So sánh nhanh Phiên bản A vs Phiên bản B

| Tiêu chí | A — DH²-KT (GNN-centric) | B — AgentDG-KT (Agent-centric) |
|---|---|---|
| Vai trò của đồ thị | Input biểu diễn cho GNN | Bộ nhớ chung, agent đọc/ghi trực tiếp |
| Cơ chế học | End-to-end differentiable | Suy luận LLM + cập nhật có kiểm soát (Critic) |
| Tính động | Temporal point process / memory network | Agent hành động → cập nhật cấu trúc tường minh |
| Khả năng diễn giải | Qua attention/causal score | Qua ngôn ngữ tự nhiên (agent tự giải thích) |
| Chi phí suy luận | Thấp hơn, 1 forward pass | Cao hơn, nhiều lệnh gọi LLM/agent mỗi sự kiện |
| Rủi ro chính | Thiếu dữ liệu Teacher/Forum đồng bộ | Hallucination/drift của agent, cần Critic bắt buộc |
| Đánh giá | AUC/ACC + ablation node/hyperedge | AUC/ACC + tỉ lệ hallucination, độ đồng thuận Critic, khớp đường cong quên, đánh giá con người với gợi ý của Tutor Agent |
| Bám xu hướng nào rõ nhất | GNN + Causal AI | LLM + Agentic AI (đúng tinh thần "cognitive modeling" bạn nêu ban đầu) |

**Gợi ý thực dụng**: A và B không loại trừ nhau — có thể viết A như baseline kỹ thuật (dễ huấn luyện, dễ benchmark chuẩn AUC) và B như hướng mở rộng/thảo luận (discussion) cho khả năng diễn giải và tích hợp LLM, hoặc chọn một trong hai làm trọng tâm chính tuỳ venue: A hợp AIED/EDM/LAK theo lối truyền thống GNN-KT; B hợp thêm cả các venue/workshop về LLM agent (đồng thời vẫn nộp được AIED/EDM nếu nhấn về giáo dục).

## 4. Phương án thực nghiệm cho B

- **Baseline agent**: multi-agent KT (administrator–judger–critic), AgentCAT, LOOM, Agent4Edu.
- **Dữ liệu**: do log Teacher/Forum đồng bộ hiếm, có thể **mô phỏng tương tác agent** trên benchmark chuẩn (ASSISTments, EdNet) bằng cách để Tutor Agent đóng vai giáo viên/hint-giver — theo đúng cách AgentCAT mô phỏng CAT.
- **Metric bổ sung so với A**: tỉ lệ hallucination của Diagnostician (đối chiếu ground-truth performance), tỉ lệ Critic phải sửa/từ chối cập nhật, độ khớp với đường cong quên Ebbinghaus (như Agent4Edu), đánh giá con người (giáo viên thật chấm chất lượng gợi ý của Tutor Agent).
- **Ablation bắt buộc**: bỏ Critic Agent (đo mức độ trôi trạng thái tăng bao nhiêu — chứng minh Critic không phải phần trang trí); bỏ Forum/Social Agent; agent-driven update vs. temporal point process cố định (so trực tiếp với cơ chế của A trên cùng dữ liệu).

## 5. Hạn chế cần nêu rõ trong bài

- Chi phí suy luận (nhiều lệnh gọi LLM mỗi sự kiện học tập) có thể không khả thi ở quy mô lớp học thật-time lớn — cần bàn hướng distill agent policy thành mô hình nhỏ sau giai đoạn khám phá (tương tự hướng trainable graph memory ICLR 2026).
- Đánh giá "hallucination/drift" chưa có metric chuẩn hoá trong KT — cần tự định nghĩa và biện minh rõ trong bài, tránh bị phản biện là thiếu ground truth.
- Multi-agent tăng độ phức tạp hệ thống — cần phần reproducibility/latency analysis kỹ hơn A.

## Nguồn tham khảo bổ sung cho phiên bản B

- LOOM (learner memory graph) và Agent4Edu (mô phỏng đường cong quên Ebbinghaus): được nhắc tới qua bài tổng hợp [2026 Memory Literature Scan — LLM Agent Research](https://lin-guanguo.github.io/llm-memory-research/memory.literature-scan/); **chưa xác minh được link bản gốc trực tiếp của 2 công trình này** — cần tự tra lại trước khi trích dẫn chính thức trong bài báo.
- [AgentCAT: Simulating Computerized Adaptive Testing via Multi-Agent LLMs](https://arxiv.org/pdf/2606.21832)
- [LLM Agents for Education: Advances and Applications](https://arxiv.org/pdf/2503.11733)
- [Graph-Augmented Large Language Model Agents: Current Progress and Future Prospects](https://arxiv.org/pdf/2507.21407)
- [GraphRAG-Induced Dual Knowledge Structure Graphs for Personalized Learning Path Recommendation](https://arxiv.org/abs/2506.22303)
- [PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM](https://arxiv.org/abs/2511.17467)
- [Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction](https://arxiv.org/pdf/2602.00020)
