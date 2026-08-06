# Bản thảo LaTeX — DH²A-KT

## Venue đề xuất: Q1

Đã kiểm tra trực tiếp trên Scimago (không suy đoán từ trí nhớ, vì quartile thay đổi theo năm):

| Journal | SJR 2025 | Quartile xác nhận | Vì sao phù hợp |
|---|---|---|---|
| **IEEE Transactions on Learning Technologies** (khuyến nghị chính) | 1.893 | **Q1** trong Computer Science Applications, Education, E-learning, Engineering (misc) | Scope nêu đích danh "intelligent tutors," "personalized and adaptive learning systems," "learning analytics and educational data mining" — khớp gần như 1-1 với DH²A-KT |
| **Knowledge-Based Systems** (phương án 2) | 1.753 | **Q1** trong Artificial Intelligence, Information Systems and Management, Software (Q1 liên tục từ ~2010-2011) | H-index 206, phạm vi rộng hơn (knowledge engineering, ML-based systems) — hợp nếu muốn nhấn khung "hypergraph + causal AI" tổng quát hơn là riêng giáo dục |

**Lưu ý quan trọng**: Applied Intelligence (APIN) — nơi P0 đang nộp — hiện là **Q2** (SJR 2025 = 0.889, Q2 trong Artificial Intelligence, đã kiểm tra trực tiếp trên Scimago), không phải Q1. Nếu mục tiêu là Q1, DH²A-KT nên nhắm venue khác P0, không nộp cùng chỗ.

Nguồn: [Scimago — Applied Intelligence](https://www.scimagojr.com/journalsearch.php?q=23674&tip=sid), [Scimago — IEEE Transactions on Learning Technologies](https://www.scimagojr.com/journalsearch.php?q=19700167026&tip=sid), [Scimago — Knowledge-Based Systems](https://www.scimagojr.com/journalsearch.php?q=24772&tip=sid).

## Cấu trúc file

- `main.tex` — bản thảo IEEE Transactions; Results/Discussion/Conclusion + Fig. 1–2 đã điền (6 Aug 2026). Còn `\todo` duy nhất: ngày nộp bài.
- `tables/` — 5 bảng LaTeX, regenerate bằng `python scripts/08_export_paper_tables.py`.
- `figures/` — Fig. 1 TikZ architecture; Fig. 2 M6 sweep (`python scripts/09_export_paper_figures.py`).
- `main.pdf` — biên dịch local ~7 trang.
- `references.bib` — 18 mục, gồm P0 và các công trình đã tổng hợp trong `docs/idea-D-plan.md`. 4 mục (`thmn2025`, `hgkt2025`, `htkt2025`, `hhskt2023`, `esk2024`) có ghi chú "verify author list before submission" — tôi chỉ xác nhận được title/DOI/venue qua tìm kiếm, chưa xác nhận được đầy đủ danh sách tác giả, nên không tự điền để tránh trích dẫn sai.

## Đã kiểm tra biên dịch

Cập nhật: đã cài được `IEEEtran.cls`, `algorithmic.sty`/`algorithm.sty`, và `IEEEtran.bst` thật (không phải bản thay thế) vào sandbox — CTAN bị chặn mạng nhưng GitHub thì không, nên lấy 3 file này từ các repo GitHub công khai vendor lại đúng nguyên bản (`IEEEtran.cls` từ `iagoac/IEEE`, `algorithm.sty`/`algorithmic.sty` build từ `.dtx` gốc của `rbrito/algorithms`, `IEEEtran.bst` v1.12 của Michael Shell từ `suixiaodan/IEEE-BibTeX-style`), cài vào `TEXMFHOME` rồi `mktexlsr`. Từ đó biên dịch **trực tiếp** `main.tex` không sửa gì — `pdflatex → bibtex → pdflatex → pdflatex` — kết quả **0 lỗi, 0 citation/reference undefined**, ra đúng PDF 2 cột chuẩn IEEE Transactions, 4 trang (đã xem trực quan trang 1: tiêu đề, abstract, mục Introduction/Contributions với trích dẫn [4]-[8] lên số đúng).

## Cách biên dịch trên máy bạn

`IEEEtran.cls` và `algorithmic` có sẵn mặc định trên **Overleaf** hoặc bất kỳ bản TeX Live "full" nào. Trên máy Windows:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Hoặc đơn giản nhất: tải `main.tex` + `references.bib` lên Overleaf, chọn compiler pdfLaTeX, Overleaf tự có `IEEEtran.cls`.
