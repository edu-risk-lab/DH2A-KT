# KBS manuscript source

Venue: *Knowledge-Based Systems* (Elsevier). Source of record: `main.tex`.

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplement.tex   # after main.aux exists (xr)
```

Do **not** regenerate the headline AUC table with
`scripts/08_export_paper_tables.py`. Map tables to scripts in
[`../docs/paper-artifacts.md`](../docs/paper-artifacts.md).
