# Ý tưởng bản thảo C: DH²A-KT — Hybrid Two-Tier Dynamic Heterogeneous Hypergraph + Agentic Knowledge Tracing

*(DH²A-KT = Dynamic Heterogeneous Hypergraph + Agentic Knowledge Tracing)*

> Nguyên tắc thiết kế: **Tier 1 (lõi kỹ thuật) gánh toàn bộ tuyên bố định lượng** — huấn luyện end-to-end, benchmark AUC/ACC, ablation đầy đủ, chạy được trọn vẹn trên 1x RTX 3090 24GB. **Tier 2 (agentic) chỉ gánh tuyên bố định tính/diễn giải** — pilot có kiểm soát quy mô, không cố benchmark ngang hàng Tier 1. Đây chính là cách giới hạn phạm vi để tránh bẫy "B đơn lẻ khả thi thấp" đã đánh giá ở bước trước.

## 1. Tóm tắt kiến trúc hai tầng

```
                ┌───────────────────────────────────────────┐
                │   TIER 1 — DH²-KT (lõi, huấn luyện đầy đủ)  │
                │   Heterogeneous hypergraph + dual-gated     │
                │   temporal encoder + causal-effect layer    │
                │   Output: knowledge-state embedding,        │
                │           dự đoán đúng/sai, causal score     │
                └───────────────────┬───────────────────────┘
                                    │ (đóng băng sau khi train xong)
                                    ▼
                ┌───────────────────────────────────────────┐
                │   TIER 2 — Lớp diễn giải Agentic (pilot)    │
                │   Diagnostician / Tutor–Hint / Critic agent │
                │   đọc output Tier 1, KHÔNG tự suy luận      │
                │   trạng thái tri thức từ đầu                │
                │   Output: giải thích ngôn ngữ tự nhiên,      │
                │           gợi ý hint, case study             │
                └───────────────────────────────────────────┘
```

Điểm mấu chốt: Tier 2 **không phải một pipeline suy luận song song độc lập** (như bản B thuần tuý) mà là **agent bọc quanh output đã có sẵn của Tier 1** — giảm mạnh số lệnh gọi LLM cần thiết (mỗi sự kiện chỉ cần 1 lệnh diễn giải thay vì 3-5 lệnh suy luận từ đầu), nên khả thi trên tập mẫu lớn hơn nhiều so với B nguyên bản mà vẫn giữ trần rủi ro compute thấp.

## 2. Câu hỏi nghiên cứu (điều chỉnh từ A + B)

- RQ1 (Tier 1, định lượng): Hypergraph dị thể động cải thiện AUC/ACC dự đoán knowledge state ra sao so với baseline (DKT/SAKT/AKT/DyGKT), và đóng góp đến từ node/hyperedge nào?
- RQ2 (Tier 1, định lượng): Tầng nhân quả có tách được hiệu ứng can thiệp thật khỏi tương quan giả trên dữ liệu quan sát không (đo qua ablation + độ nhạy với giả định nhận diện)?
- RQ3 (Tier 2, định tính): Một lớp agent diễn giải đọc trực tiếp output Tier 1 có tạo ra giải thích ngôn ngữ tự nhiên **nhất quán** với điểm số/trạng thái mà Tier 1 đã tính không, và tỉ lệ "trôi"/ảo giác là bao nhiêu trên tập mẫu pilot?
- RQ4 (Tier 2, định tính): Giải thích/gợi ý do agent sinh ra có được giáo viên thật đánh giá là hữu ích hơn so với chỉ hiển thị điểm số thô của Tier 1 không?

=> RQ1–RQ2 là xương sống bài báo (có thể trả lời chắc chắn bằng thực nghiệm trên 1 GPU). RQ3–RQ4 là phần mở rộng/thảo luận, được phép có kết luận sơ bộ ("promising nhưng cần khảo sát quy mô lớn hơn ở tương lai") mà không làm yếu bài.

## 3. Phạm vi dữ liệu — chốt thực tế cho 1x RTX 3090 24GB

| Nguồn | Vai trò | Ghi chú khả thi |
|---|---|---|
| ASSISTments 2012 (27.145 học sinh, 2,63M interaction, 245 KC) | Benchmark chính Tier 1 | Full training feasible trong ~1 ngày |
| Junyi Academy (mẫu phổ biến ~500k interaction, 705 câu hỏi, 39 KC) | Benchmark phụ + kiểm tra prerequisite hyperedge | Nhẹ, dùng để test tầng concept-prerequisite |
| EdNet (subsample 5–15k học sinh từ tập gốc 780k học sinh / 88M interaction) | Kiểm tra khả năng mở rộng (scalability) | Không dùng full 88M — chỉ subsample, đúng thông lệ literature |
| ES-KT-24 (đa phương thức, có video) | Kiểm chứng node Video | Dùng embedding CLIP trích sẵn (offline), không cần train lại encoder video |
| Node Teacher/Forum | **Không có benchmark public đồng bộ** | Xử lý bằng 1 trong 2 cách: (a) giới hạn Tier 1 quantitative ở Student-Exercise-Concept-Hint-Video, đưa Teacher/Forum vào **Tier 2 case study** với tập nhỏ tự thu thập/gán nhãn thủ công (vài trăm bản ghi); (b) nêu rõ đây là **acknowledged limitation**, không claim đã kiểm chứng đầy đủ 8 loại node ở Tier 1 |

## 4. Ngăn xếp công cụ (tech stack) đề xuất

| Lớp | Công cụ | Lý do |
|---|---|---|
| GNN framework | PyTorch Geometric — `HypergraphConv` (Bai et al.) kết hợp `HeteroConv`/`to_hetero()` để ghép dị thể | PyG đã hỗ trợ sẵn cả hypergraph convolution và heterogeneous graph, không cần viết lại từ đầu |
| Text embedding | Sentence-embedding nhỏ (MiniLM/BERT-base cỡ), trích 1 lần offline | Không cần fine-tune, gần như không tốn GPU |
| Video embedding | CLIP/ViT trích frame embedding offline | Nghẽn ở I/O chứ không phải VRAM |
| LLM cho Tier 2 (local) | 7B–8B instruct quantized 4-bit (ví dụ nhóm Qwen3 8B, ~5GB VRAM) là lựa chọn cân bằng tốc độ/chất lượng; có thể nâng lên lớp 13–14B (~8–10GB 4-bit) nếu chất lượng giải thích chưa đạt | Vẫn còn dư >10GB VRAM để chạy song song Tier 1 khi cần kiểm tra tích hợp |
| LLM cho Tier 2 (phương án dự phòng) | Gọi API (nếu ngân sách cho phép) cho riêng phần đánh giá cuối cùng (case study chốt) | Tránh rủi ro throughput nếu local model không đủ nhanh cho deadline |
| Theo dõi thực nghiệm | Weights & Biases / MLflow local | Ghi lại toàn bộ ablation, tránh mất kết quả khi so sánh nhiều biến thể |

## 5. Kế hoạch thực hiện theo pha (khung 16 tuần, có thể co giãn)

| Pha | Tuần | Nội dung chính | Deliverable |
|---|---|---|---|
| 0 — Chuẩn bị | 1–2 | Set up môi trường (PyTorch + PyG), tải/tiền xử lý ASSISTments2012, Junyi; audit dữ liệu Teacher/Forum để chốt phạm vi Tier 1 vs Tier 2 (mục 3) | Pipeline tiền xử lý + báo cáo audit dữ liệu |
| 1 — Xây Tier 1 (core) | 3–6 | Xây graph dị thể + hyperedge (session, concept-prerequisite); cài `HypergraphConv`/`HeteroConv`; encoder text/video offline; cài baseline DKT/SAKT/AKT/DyGKT để so sánh | Code Tier 1 chạy được end-to-end trên 1 dataset nhỏ |
| 2 — Tầng nhân quả Tier 1 | 6–7 | Thêm causal-effect layer (bắt đầu bằng ước lượng đơn giản: propensity-score/2-stage regression trước khi thử kiến trúc phức tạp hơn) | Module causal + test đơn vị |
| 3 — Thực nghiệm định lượng Tier 1 | 8–10 | Train full trên ASSISTments2012 + Junyi + EdNet-subsample; chạy toàn bộ ablation (node, hypergraph vs pairwise, static vs dynamic, có/không causal); kiểm định ý nghĩa thống kê | Bảng kết quả AUC/ACC + ablation, biểu đồ |
| 4 — Xây Tier 2 (pilot) | 11–12 | Cài 3 agent lõi (Diagnostician-wrapper, Tutor/Hint, Critic) đọc output Tier 1 đã đóng băng; chọn tập mẫu pilot (vài trăm–vài nghìn interaction); dùng LLM local 7–8B quant | Agent pipeline chạy được trên tập mẫu |
| 5 — Đánh giá Tier 2 | 12–13 | Đo tỉ lệ hallucination/trôi so với ground truth Tier 1, tỉ lệ Critic phải sửa, (nếu có điều kiện) khảo sát giáo viên thật đánh giá chất lượng gợi ý | Bảng metric Tier 2 + case study minh hoạ |
| 6 — Tổng hợp & viết | 14–16 | Gộp kết quả Tier 1 + Tier 2, viết rõ ràng phần "Scope & Limitations" tách bạch cái gì đã benchmark định lượng và cái gì chỉ là pilot; nội bộ rà soát trích dẫn/số liệu trước khi nộp | Bản thảo hoàn chỉnh |

## 6. Ngân sách compute & điểm kiểm tra (checkpoint)

- Sau Pha 1: đo thời gian 1 epoch trên tập nhỏ để ước lượng tổng thời gian Pha 3 — nếu vượt ngân sách, cắt bớt số lượng ablation thay vì cắt dataset.
- Trước khi vào Pha 4: đo tốc độ suy luận thực tế của LLM local trên máy (token/giây) — nếu quá chậm để xử lý cả tập pilot dự kiến, giảm cỡ mẫu pilot hoặc chuyển sang API cho riêng bước đánh giá cuối, **không** mở rộng ngược lại thành benchmark toàn tập (giữ đúng cam kết scope Tier 2 đã chốt ở đầu).
- Ghi log GPU memory/thời gian mỗi thí nghiệm để đưa vào phần "reproducibility" của bài báo.

## 7. Bảng rủi ro & giảm thiểu

| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| Thiếu dữ liệu Teacher/Forum đồng bộ | Cao | Giới hạn rõ phạm vi Tier 1; đưa vào Tier 2 case study nhỏ hoặc "future work" |
| Causal claim trên dữ liệu quan sát bị phản biện | Trung bình | Nêu rõ giả định nhận diện; bắt đầu bằng ước lượng đơn giản, dễ biện minh trước khi làm phức tạp |
| LLM local không đủ nhanh cho tập pilot dự kiến | Trung bình | Có phương án dự phòng dùng API cho bước đánh giá cuối; giảm cỡ mẫu pilot nếu cần |
| Thiếu metric chuẩn hoá cho "hallucination/drift" trong KT | Trung bình | Tự định nghĩa rõ + biện minh, đối chiếu với ground truth Tier 1 làm proxy |
| Ablation Tier 1 quá nhiều tổ hợp, vượt thời gian | Thấp–Trung bình | Ưu tiên ablation theo RQ, cắt bớt tổ hợp phụ nếu cần |

## 8. Kế hoạch đánh giá (tách bạch rõ 2 tầng)

- **Tier 1 (định lượng, bắt buộc)**: AUC, ACC trên từng dataset; so sánh baseline; ablation node/hyperedge/dynamic/causal; kiểm định thống kê (ví dụ paired t-test giữa các mô hình).
- **Tier 2 (định tính, phạm vi pilot, không thay thế Tier 1)**: tỉ lệ giải thích của agent khớp với output Tier 1 (proxy cho "faithfulness"); tỉ lệ Critic can thiệp/sửa; (tuỳ điều kiện) đánh giá con người trên một tập nhỏ case study.

## 9. Venue & mốc nộp bài

Giữ nguyên định hướng AIED/EDM/LAK cho phần Tier 1 (đúng chuẩn cộng đồng GNN-KT); có thể thêm 1 phần thảo luận/case-study Tier 2 làm điểm nhấn khác biệt so với các submission GNN-KT thuần tuý khác, mà không cần đạt chuẩn benchmark multi-agent quy mô lớn.

## Nguồn tham khảo bổ sung cho kế hoạch C

- [HypergraphConv — PyG Guide (Kumo.ai)](https://kumo.ai/pyg/layers/hypergraph-conv/)
- [Heterogeneous Graph Learning — PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)
- [PyTorch Geometric High Order: A Unified Library for High Order GNN](https://arxiv.org/pdf/2311.16670)
- Ghi chú VRAM/throughput LLM local (Qwen3 8B ~5GB ở 4-bit, lớp 13–14B ~8–10GB ở 4-bit): tổng hợp từ các bài hướng dẫn phần cứng LLM 2026 — cần tự đo lại tốc độ token/giây thực tế trên máy trước khi chốt Pha 4.
