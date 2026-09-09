# DH²A-KT

Leakage-audited **knowledge attribution** for knowledge tracing (KT).
The manuscript for *Knowledge-Based Systems* lives in [`paper/`](paper/).
This is not a new hypergraph predictor: the pipeline is construct → audit
→ matched-twin ablation → credit only messages that meet a pre-specified
gate.

Companion protocol paper (pairwise train-only graphs, P0):
[`edu-risk-lab/leakage-controlled-kt-audit`](https://github.com/edu-risk-lab/leakage-controlled-kt-audit),
pinned at commit `5e3d6a0` under `external/p0_leakage_audit`.
P0 is a separate submission (EAAI); this repository restates the reused
protocol and does not redistribute P0's datasets.

## Reproduce the paper numbers

| What | Where |
|---|---|
| Table → script map | [`docs/paper-artifacts.md`](docs/paper-artifacts.md) |
| P0 overlap / evaluator split | [`docs/p0_overlap_disclosure.md`](docs/p0_overlap_disclosure.md) |
| Environment and data (not in Git) | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Headline tables (JSON/CSV) | `results/tables/` (allowlisted artefacts only) |

Do **not** run `scripts/08_export_paper_tables.py` to rebuild the
headline XES3G5M AUC table: that exporter still writes an older
graph-only format.

## Setup

Python ≥ 3.10. Clone with the vendored P0 tree (already in this repo):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
pip install -r external/p0_leakage_audit/requirements.txt
```

Optional GPU stack (RTX 3090 / CUDA): install a matching `torch` wheel,
then uncomment the Tier-1 lines in `requirements.txt`.
`scripts/00_setup_p0_submodule.sh` documents pyKT.

Raw XES3G5M / ASSISTments / Junyi files are **not** in Git. Follow
`external/p0_leakage_audit/README.md` (data download). FoundationalASSIST
must be obtained from Hugging Face under its Responsible Use Guidelines
and must not be redistributed.

## Tests

```bash
# Python 3.12+: disable auto-loaded pytest plugins if they conflict
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # PowerShell: $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -q
```

## Manuscript

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# Supplementary Material (Critic pilot + IPW): compile after main.aux exists
pdflatex supplement.tex
```

Source of record is `paper/main.tex`. Compiled PDFs and advisor-review
files are gitignored.

## Layout

```
dh2a_kt/          library (data, models, hyperedge audit, eval)
scripts/          numbered experiment drivers
tests/            unit tests (no GPU required for the core suite)
paper/            KBS LaTeX + tables + figures
results/tables/   reported numbers (not checkpoints)
external/p0_leakage_audit/   pinned P0 snapshot
```

## Licence

MIT (`LICENSE`). Datasets remain under their providers' terms.
P0 and pyKT are MIT; see `LICENSE` for the third-party notice.
