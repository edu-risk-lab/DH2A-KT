# Kế hoạch bổ sung bản thảo IEEE TLT — theo review `paper/review_DH2A-KT_IEEE-TLT.docx`

> Khuyến nghị review: **MAJOR REVISION**.  
> Mục tiêu: đưa `paper/main.tex` từ “skeleton có số liệu” → **sẵn sàng nộp Short Paper (≤8 trang)** hoặc Regular (≤14 trang) mà không còn lỗi liêm chính trích dẫn / placeholder.

Nguồn: `paper/review_DH2A-KT_IEEE-TLT.docx` (rà soát mô phỏng phản biện, 6 Aug 2026).  
Bản thảo hiện tại: ~7 trang, 2 cột.

---

## 0. Nguyên tắc ưu tiên

| Mức | Ý nghĩa | Khi nào xong mới nộp |
|-----|---------|----------------------|
| **P0 — BẮT BUỘC** | Desk-reject / liêm chính | Trước mọi vòng nộp |
| **P1 — NÊN LÀM** | Reviewer TLT sẽ hỏi mạnh | Trước nộp khuyến nghị |
| **P2 — CÂN NHẮC** | Tăng sức thuyết phục; tốn compute/effort | Nếu còn thời gian |

**Giữ nguyên:** tinh thần tự phê phán, manipulation check, Limitations trung thực, Critic gate Tier 2.

---

## Phase A — P0 Liêm chính & nộp được (ước lượng 1–2 ngày)

### A1. Xóa placeholder `[TODO]`
- [x] Footnote trang 1: `[TODO: Manuscript received/revised dates…]` → đã xóa; bỏ macro `\todo`.
- [x] Grep `paper/main.tex` + `references.bib`: không còn TODO/verify.

**File:** `paper/main.tex`

### A2. Sửa 100% bibliography “verify / TBD / Author list to verify”
| Key / ~ref | Việc cần làm | Trạng thái |
|------------|--------------|------------|
| `thmn2025` | Mohammadi et al.; CCIS 2592, pp. 77–85 | **Done** |
| `hgkt2025` | Ma, Zhu, Lei; Appl. Sci. 15(15):8617 | **Done** |
| `htkt2025` | Guo et al.; ISPA 2024, pp. 1472–1477 | **Done** |
| `psykt2024` | Wang et al.; Front. Psychol. 15:1359199 | **Done** |
| `hhskt2023` | Ni et al.; ESWA 215:119334 | **Done** |
| `minn2022interpretable` | AAAI 36(11):12810–12818, DOI 10.1609/aaai.v36i11.21560 | **Done** |
| `agentcat2026`, `llmedu2025`, `graphaugmentedllm2025`, `graphrag2025`, `esk2024` | Bổ sung author | **Done** |
| `dygkt2024` | Cheng et al.; KDD 2024, pp. 409–420 | **Done** |

**Gate:** `grep -i "verify\|TBD\|Author list" paper/references.bib` → **0 hits** (đã đạt, 6 Aug 2026).

### A3. Phụ thuộc P0 / [8] chưa peer-review
- [x] **Option (a)** — thêm §\ref{sec:p0-selfcontained} *Self-contained summary of the reused P0 protocol* trong Preliminaries + note supplementary PDF trong Baselines.

---

## Phase B — P1 Khoảng cách claim ↔ bằng chứng (ước lượng 2–3 ngày prose)

### B1. Title / Abstract / Contributions — hạ giọng & khớp phạm vi
- [x] Abstract viết lại (Phase B): quantitative core = chain concept-prerequisite; HE dị thể / causal / human lúc đó còn đóng.
- [x] **Cập nhật sau Phase C (7 Aug 2026):** Abstract/C1/C3/C5 phản ánh FoundationalASSIST session+Hint + limited ATE + dual-rater human eval — vẫn giữ primary = XES3G5M; không claim SOTA / RCT.
- [x] Title giữ nguyên; Motivation gap điều chỉnh.
- [x] Intro + Related Work không overclaim “we fill heterogeneous empirically” trên 3 benchmark P0.

### B2. Định nghĩa hyperedge 4 loại (§III)
- [x] Scope note ngay sau định nghĩa $h$: chỉ concept-prerequisite vào Tier~1 train/eval **trên XES3G5M**; session+Hint trên FoundationalASSIST (secondary).

### B3. Thảo luận giá trị thực tiễn AUC↓ / graph-reliance↑
- [x] §\ref{sec:auc-tradeoff} *When is lower AUC with graph reliance preferable?*

### B4. Junyi & ASSIST2012
- [x] Datasets: XES3G5M = sole *primary* predictive; Junyi structural only.
- [x] **Cập nhật Phase C:** secondary fold trên **FoundationalASSIST** (không tái dùng P0 ASSIST2012 GKT AUC).
- [x] Results: giải thích F1 chain < E_pre (path collapse / projection).

### B5–B6. Fig. 2 + presentation
- [x] Ngưỡng 0.003 = illustrative floor (~10× inert noise) trong Results + caption + figure legend.
- [x] Leakage table: cap $100{,}000$ + số cặp projected.
- [x] Thuật ngữ DH²A-KT (hệ) vs DH²-KT (Tier 1) giữ trong Conclusion.

---

## Phase C — P2 Tăng sức mạnh thực nghiệm (optional)

| Track | Việc | Trạng thái (7 Aug 2026) |
|-------|------|-------------------------|
| **C-ES** | Session HE dị thể + audit + ablation | **Xong** qua FoundationalASSIST (thay ES-KT-24): 308k session HE; +20k train → AUC 0.722 (Δ+0.002 vs 0.720) |
| **C-Human** | Likert faithfulness/usefulness, 2–3 raters | **Xong** 2 raters (A1/B1), n=40: F 4.01±1.07, U 3.53±0.71; Spearman 0.71/0.58 |
| **C-ASSIST** | Secondary predictive fold | **Xong** FoundationalASSIST fold0 AUC 0.720 (+ session 0.722); **không** claim = ASSIST2012 P0 |
| **C-ATE** | (bổ sung) hint ATE khi có log | **Xong** limited IPW −0.207; trimmed −0.211; CI95 [−0.230,−0.187]; weak overlap |
| **C-Junyi-train** | Full Junyi train | **Không làm** (đúng khuyến nghị) |

---

## Phase D / Review-2 follow-ups (7 Aug 2026)

Verification review (`paper/review2_DH2A-KT_IEEE-TLT.docx`): **MINOR REVISION**.

| Mục | Trạng thái |
|-----|------------|
| Manipulation check cho GKT (reuse P0 DDR) + simpleKT N/A | **Xong** — `table_manipulation.tex`; thảo luận reframed |
| Mean±SD / paired Δ cho AUC chính | **Xong** — `table_auc_xes3g5m.tex`; `scripts/17_export_review2_stats.py` |
| Cover letter Short Paper + FoundationalASSIST access note | **Xong** — `paper/cover_letter_tlt_short.md`; Acknowledgment |
| Push bản review-2 | ⬜ khi tác giả yêu cầu |

---

## Checklist nộp (gate cuối)

- [x] Không còn `[TODO]` / note “verify before submission” trong PDF
- [x] References: không còn verify/TBD trong `references.bib`
- [x] Abstract khớp phạm vi **hiện tại**: primary XES3G5M; Junyi GT-only; FoundationalASSIST secondary; Tier2 pilot + dual-rater; GKT cũng pass destruction
- [x] Manipulation check + Limitations giữ tinh thần tự phê phán (GKT ΔAUC lớn hơn; ATE overlap yếu; human n=40/2 raters)
- [x] Fig. 2 threshold được định nghĩa trong text
- [x] `main.pdf` biên dịch; **≤8 trang** Short Paper
- [x] Cover letter draft (`paper/cover_letter_tlt_short.md`)
- [ ] Push GitHub bản nộp (review-2 fixes)
---

## Mapping file chính

| Việc | File |
|------|------|
| Prose / claim | `paper/main.tex` |
| Bib | `paper/references.bib` |
| Tables / figs | `paper/tables/` (gồm `table_foundational_assist.tex`, `table_human_eval.tex`), `paper/figures/` |
| Số liệu nguồn | `results/tables/` (local), `docs/M9-results-summary.md` |
| Theo dõi tiến độ dự án | `docs/execution-plan.md` |
| Kế hoạch này | `docs/paper-revision-plan-tlt.md` |

---

## Không làm (tránh scope creep trước nộp Short Paper)

- Không claim SOTA AUC trên XES3G5M.
- Không train lại 9 baseline P0.
- Không biến Tier 2 thành full multi-agent re-inference trên toàn log.
- Không đánh đồng FoundationalASSIST với ASSIST2012 / tái dùng GKT P0.
- Không diễn giải ATE âm như “cấm hint” (weak overlap).
- Không phóng đại ΔAUC session (+0.002) thành win lớn.