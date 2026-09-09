# Reproducibility checklist

English overview: [`README.md`](../README.md). Table-to-script map:
[`paper-artifacts.md`](paper-artifacts.md). P0 overlap:
[`p0_overlap_disclosure.md`](p0_overlap_disclosure.md).

This file lists environment pins, data that is **not** in Git, and what
not to push. Checkpoints and raw logs are gitignored; reported numbers
live under `results/tables/` (allowlisted).

## 1. Environment

| Component | Requirement | Notes |
|---|---|---|
| Python | `>=3.10` | Matches `pyproject.toml`; core tests run without GPU |
| DH2A-KT deps | `requirements.txt` | `pip install -r requirements.txt` |
| P0 deps | `external/p0_leakage_audit/requirements.txt` | scipy / tqdm / matplotlib / pyarrow / torch / hypothesis |
| pykt-toolkit | `github.com/pykt-team/pykt-toolkit` | Not vendored; see `scripts/00_setup_p0_submodule.sh` |
| GPU | 1× RTX 3090-class (24 GB) for training | Inference of stored tables does not need GPU |
| torch / torch_geometric | Not pinned here | Install a CUDA-matched wheel, then uncomment Tier-1 lines in `requirements.txt` |

Do not bump versions outside the ranges in `requirements.txt` without
updating that file.

On Python 3.12+, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` if third-party
pytest plugins conflict.

## 2. Frozen provenance (do not change)

| What | Value |
|---|---|
| P0 pin | `5e3d6a01c32a649f1655ac8712f03359bddfe478` under `external/p0_leakage_audit` |
| Split protocol | learner-based, 0.7/0.1/0.2, 3 folds, seeds `42`/`43`/`44` |
| Manipulation-check | `p=0.90`, default operator `node_drop`, seed `42` |
| Config checksum (xes3g5m) | sha256 `4f9f35d9…d980b2` |
| Config checksum (assist2012) | sha256 `a98dd247…4e6c8f9` |
| Config checksum (junyi) | sha256 `24aad059…fd6bc8` |

Verify checksums with:

```bash
python scripts/01_verify_p0_preprocessing_match.py configs/xes3g5m.yaml
```

If a checksum differs after clone, stop: the P0 pin or a config drifted
and baseline comparisons are no longer valid.

## 3. Data — not in Git

Raw and processed XES3G5M / ASSISTments 2012–2013 / Junyi files are
**not** committed. `.gitignore` blocks
`external/p0_leakage_audit/data/{raw,processed}/*` and `/data/`.
FoundationalASSIST must be obtained from Hugging Face under its
Responsible Use Guidelines and must not be redistributed.

On a GPU machine, after clone:

1. Follow `external/p0_leakage_audit/README.md` (data download).
2. Place files in `external/p0_leakage_audit/data/raw/<dataset>/`.
3. Preprocess with P0 code only (`python -m src.preprocess --config ...`).
4. Run `python -m src.split_checker --config ...` and confirm no learner leakage.

## 4. Directory skeleton (empty content, tracked `.gitkeep`)

```
DH2A-KT/
├── results/tables/     # allowlisted paper numbers only
├── results/figures/
├── checkpoints/        # gitignored weights
├── logs/
└── external/p0_leakage_audit/data/{raw,processed}/
```

## 5. Do not push

- `external/p0_leakage_audit/data/**` (raw + processed)
- `checkpoints/*.pt` / `*.pth` / `*.ckpt`
- `.venv/`
- advisor reviews, `reviewer/`, compiled `paper/main.pdf`
- unpublished lab diaries and idea plans (gitignored; keep local)

Headline XES3G5M AUC: do **not** rebuild with
`scripts/08_export_paper_tables.py` (legacy graph-only exporter).

## 6. Clone and test

Snapshot matching the manuscript: tag `kbs-submit-2026-09-09`,
DOI [10.5281/zenodo.22676635](https://doi.org/10.5281/zenodo.22676635).

```bash
git clone https://github.com/edu-risk-lab/DH2A-KT.git
cd DH2A-KT
git checkout kbs-submit-2026-09-09   # optional: exact snapshot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r external/p0_leakage_audit/requirements.txt
python -m pytest tests/ -q
```

Training and table rebuilds are listed in
[`paper-artifacts.md`](paper-artifacts.md). P0 is a pinned snapshot in
this tree (commit `5e3d6a0`); do not retarget it without updating the
manuscript Data availability statement.
