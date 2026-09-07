# Checklist Gate 0 — sau review kiểu GS Hậu (07/09/2026)

Nguồn: nhận xét vòng 1 (29/08) + vòng 2 ver3 (05/09), đối chiếu bản attribution hiện tại.
**Không chạy GPU** cho các mục dưới. Đợt máy (M4/P4 5-seed, ECE pyKT, bootstrap trên giao) chỉ khi Gate 0 khóa và còn quỹ.

Quy ước: `[x]` = đã sửa trong bản thảo lần này. `[ ]` = cố ý chưa làm (cần máy / Zenodo / quyết định nộp).

## A — buộc phải sửa (chữ)

- [x] **A1 Abstract / C2 / Kết luận:** chỉ nêu hàng 5-seed (Δt observed + zero-incidence PASS; incidence FAIL 0/5; history-only 2/5). Bỏ `as do` / `do not` cho sáu FAIL một seed. Sáu hàng đó xuống bảng, dán nhãn single-seed.
- [x] **A2 Abstract:** bỏ bootstrap khỏi Abstract. Thân bài + chú thích bảng: CI học viên = nhiễu lấy mẫu học viên trên *một* checkpoint, không phải biến thiên 5-seed. Tách *n*=1.093.720 (bảng AUC) vs 1.093.755 (native / twin dung lượng / calibration).

## B — nên sửa (chữ)

- [x] **B5:** QKC-T = cánh tay so sánh cùng họ GIKT; Zero+Δt = hệ tối thiểu được ghi công.
- [x] **B1 pin:** Data availability + cover letter: `ca8bc2b` (HEAD lúc sửa). *Sau khi commit Gate 0, cập nhật pin sang hash mới.*
- [ ] **B1 Zenodo / URL ẩn danh:** mint DOI hoặc mở kho; kiểm tra URL ở chế độ ẩn danh trước khi nộp.
- [x] **B8 Related Work:** arXiv:2508.17092 (Badran & Preisach) — trùng *nhãn* KC-expansion, không trùng audit hyperedge / attribution twin. C1: đối chứng dương bắt leak membership; `|ρ|` vẫn 0 ở cột bẩn.
- [x] **B3 planned-gap:** một bullet Limitations (không đánh giá proxy khoảng chờ kế hoạch).
- [x] **B6 / B9:** chú thích bảng v2: 0,8747 là protocol P0 cũ, không so với simpleKT clean 0,8214.
- [x] **B7 tune:** Abstract/Kết luận không đọc như đã thắng GIKT đã tune (đã có compute-matched; giữ câu Appendix hparams).
- [x] **B4 Abstract:** sáu FAIL một seed không còn trong câu bán hàng (xem A1). Caption bảng negative-knowledge: M4/P4 5-seed $0/5$; K/L/M1/M5 còn 1 seed.

## C — tuỳ chọn / không đụng máy

- [x] hidelinks đã có — *vẫn phải mở PDF bằng mắt trước khi nộp.*
- [ ] Rà DOI toàn bộ `references.bib` (Gate 0 còn lại, không chặn luận đề).
- [ ] Nhãn v2/v4: giữ bảng alias; không viết lại phụ lục.

## Cấm khi sửa

- Không đưa residual `+0.0101` / attempt-collapse vào Abstract.
- Không chạy lại capacity-matched, Zero+Δt, A2 sáu hàng cạnh tranh.
- Không đưa Critic / ATE lên trụ đóng góp.

## Việc máy — server chạy theo hướng dẫn

Chi tiết copy-paste: [`docs/HuongDan_Checklist_GPU_2026-09-07.md`](HuongDan_Checklist_GPU_2026-09-07.md). Kết quả trên `main` (fast-forward từ `gpu-checklist-2026-09-07`, `d613fd6`).

- [x] **S1** ECE/Brier/NLL pyKT — 3 hàng, `n=1,093,720`. ECE GIKT 0.0073 / AKT 0.0047 / simpleKT 0.0100 (cả ba < 0.05; không temperature scaling)
- [x] **S2** Bootstrap learner-level — 3614 học viên; CI hiệu AUC không chứa 0. Khoảng 35 hàng còn (1,093,755 vs 1,093,720)
- [x] **S3** P4 5-seed `--concept-forget` — **0/5 PASS**, mean Δval ≈ −0.00252
- [x] **S4** M4 5-seed `--group-embed` — **0/5 PASS**, mean Δval ≈ +0.00025 (seed 42 +0.00142 gần cửa)
- [x] **S5** B9 rescore simpleKT P0 L=200 — **SKIP** (không checkpoint)
