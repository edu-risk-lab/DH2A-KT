# Leakage-Controlled KC Graph Protocol for Knowledge Tracing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CITATION.cff](https://img.shields.io/badge/citation-CITATION.cff-9cf)](CITATION.cff)

Companion **research software** for the manuscript *Leakage-Controlled Concept
Graph Construction and Cold-Start Diagnostic Protocol for Knowledge Tracing*
(Applied Intelligence / APIN, Springer Nature). This repository supports
reproducible audit experiments and paper artefact generation; it is **not** a
new SOTA knowledge-tracing backbone.

**Canonical submission-ready manuscript package:** `paper/submission_APIN/` (`main_APIN.tex`,
Springer Nature `sn-jnl`, `sn-mathphys-num`, `refs_APIN.bib`).

**How to cite:** see [§10](#10-citation-licence-and-contact) and `CITATION.cff`.

> **What this repo is.** A protocol and audit pipeline that treats
> graph-mediated leakage as a *conditional* risk and turns the audit into a
> **decision-support and leakage risk-scoring layer** for graph-augmented
> knowledge tracing (KT). It provides: train-only, fold-aware multi-relational
> concept-graph construction with per-edge provenance; a DAG audit; leakage
> diagnostics (code columns `ECR_flag`, `ECR_overlap`, `EOC`, `TBVR`; the paper
> also reports throughput as builder mass / TBMR); cold-start KC stratification; a
> DAG Disruption Rate (**DDR**) probe over five augmentation operators (four
> label-agnostic plus the prerequisite-preserving `prereq_preserve`); a
> **controlled leak-injection** experiment that raises contamination throughput
> on purpose; an **anchored DDR→downstream** retraining test with a
> manipulation-check (positive control) that separates graph-reliant from
> graph-inert backbones; a **reachability-disruption** variant; a sequence
> **autocorrelation** diagnostic; **bootstrap / paired-\(t\) ΔAUC intervals**,
> paired **significance tests**, and **exploratory ANOVA**; and optional
> **ground-truth cross-validation** on Junyi (expert prerequisite DAG vs
> train-only inferred edges).
>
> **Central claim (two-factor / conditional harm).** Graph-mediated leakage
> shifts headline AUC only when contamination **throughput** and backbone
> **reliance** on the graph are simultaneously high; because AUC stays silent in
> the other regimes, the audit measures throughput directly instead of relying on
> accuracy.
>
> **What this repo is NOT.** A new KT baseline aimed at SOTA. No claim that the
> audit raises accuracy, nor about real-world learning outcomes or joint
> self-supervised graph pretraining.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Environment setup](#2-environment-setup)
3. [Data download and preparation](#3-data-download-and-preparation)
4. [Running experiments](#4-running-experiments) ([step-by-step](#40-step-by-step-experiment-guide))
   - [4.6 DDR→downstream, anchored + multi-seed (GPU)](#46-ddrdownstream-anchored--multi-seed-gpu)
   - [4.7 Sequence autocorrelation](#47-sequence-autocorrelation-diagnostic)
   - [4.8 Significance testing](#48-significance-testing)
   - [4.9 Controlled leak injection (two-factor "high-throughput" cell)](#49-controlled-leak-injection-two-factor-high-throughput-cell)
   - [4.10 Inferential summaries (ΔAUC CIs, ANOVA, epoch/parity)](#410-inferential-summaries-auc-cis-anova-epochparity)
5. [Per-stage commands](#5-per-stage-commands) ([graph_builder API](#51-graph_builder-python-api))
6. [Outputs and where they live](#6-outputs-and-where-they-live)
7. [Paper artefacts, reproduction map, and LaTeX build](#7-paper-artefacts-reproduction-map-and-latex-build)
8. [Troubleshooting](#8-troubleshooting)
9. [Project structure](#9-project-structure)
10. [Citation, licence, and contact](#10-citation-licence-and-contact)

---

## 1. Quick start

After [§2 Environment setup](#2-environment-setup) and
[§3 Data](#3-data-download-and-preparation), run a **single-dataset** smoke
pipeline (Junyi):

```bash
bash scripts/run_junyi_minimal.sh
```

On **Windows** (PowerShell, repo root):

```powershell
$env:PYTHON = "python"   # optional
bash scripts/run_junyi_minimal.sh
```

If you do not have Git Bash/WSL, run the same stages by hand (see
[§5](#5-per-stage-commands)); or use `scripts/run_all_datasets_full.ps1` for
everything including paper tables (see [§4](#4-running-experiments)). For a
**numbered walkthrough** of setup and experiment tracks, start at
[§4.0](#40-step-by-step-experiment-guide).

**Runtime.** Roughly tens of minutes on a laptop CPU for Junyi end-to-end if
deep baselines (`torch`) are enabled; longer for full three-dataset runs and
multi-fold baselines. Junyi preprocess + graph stages are memory-heavy; prefer
`--server` / `-ServerProfile` on ~32 GB RAM hosts (see full pipeline scripts).

**Sanity.** `split_checker` should report no learner leakage and temporal
ordering OK. `dag_audit` may report `cycles_before` hitting the **representative
cycle cap (100)** on dense graphs; the pruning loop still runs until the graph
is acyclic (see `paper/submission_APIN/main_APIN.tex` / `src/dag_audit.py`).

**Reports.** `results/reports/p0_diagnostic_report.md` aggregates available
CSVs and markdown reports.

---

## 2. Environment setup

Tested with **Python 3.10 / 3.11** (3.12+ often works). Linux, macOS, and
**Windows** (PowerShell + native Python) are supported; WSL2 remains optional.

### 2.1 Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2.2 Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

**Optional pyKT backend.** If you plan to run `baseline_runner` with real PyTorch / `pykt-toolkit` training (`evaluation.baseline_backend: pykt` in YAML or `--baseline-backend pykt`), clone **with submodules** or run:

```bash
git submodule update --init --recursive
```

Then install PyTorch/SciPy extras and the pinned submodule editable package:

```bash
pip install -e ".[pykt]"
pip install -e third_party/pykt-toolkit
```

(`pykt-toolkit` is pinned as a git submodule under `third_party/pykt-toolkit`;
see `third_party/README.md`. Do not rely on machine-specific `file://` paths.)

### 2.3 Tests

From the repository root:

```bash
pytest -q
```

### 2.4 GPU (optional)

Structural stages (`preprocess` → `graph_builder` → `dag_*`) use NumPy/pandas only.

PyTorch / CUDA matters only when you install the optional pyKT stack
(`pip install -e ".[pykt]"` and `pip install -e third_party/pykt-toolkit`) and run
`baseline_runner` with **`evaluation.baseline_backend: pykt`** (or `--baseline-backend pykt`).
Install a CUDA build that matches your driver when appropriate:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Data download and preparation

Public KT benchmarks only; respect each dataset licence.

**Optional bundle.** A project-maintained Google Drive folder with the expected
subfolders (`raw/`, `processed/`, `graphs/`, plus a short `README.md` inside the
drive) is here: [P0 data (Google Drive)](https://drive.google.com/drive/folders/1eQNSTV0pVDeB79Mx--vnPml_YPYwVzDP?usp=sharing).
Download what you need and place files under your local `data/` tree as in
[§3.1](#31-layout); licences of the underlying benchmarks still apply.

| Dataset | Role in P0 | External prerequisite DAG | Notes |
|--------|------------|----------------------------|--------|
| Junyi Academy | Core | Yes (`junyi_dag.csv`) | Expert DAG used only in optional GT CV; train-only prerequisite edges `E_pre` are inferred like other benchmarks. |
| ASSISTments 2012 | Core | No | Q-matrix only; `E_pre` inferred from train transitions. |
| XES3G5M | Core | Metadata only | KC IDs from hierarchy labels; `E_pre` inferred from train transitions. |

### 3.1 Layout

Place raw files under `data/raw/<dataset>/` (directories are gitignored).
Example for Junyi (filenames must match `configs/junyi.yaml`):

```
data/raw/junyi/
├── junyi_ProblemLog_original.csv   # interaction log (Chang et al. style)
├── junyi_dag.csv                   # expert prerequisite annotation (GT CV)
└── junyi_Exercise_table.csv       # used when building KC name→id mapping for GT CV
```

ASSISTments and XES3G5M paths are defined in `configs/assist2012.yaml` and
`configs/xes3g5m.yaml`.

### 3.2 Preprocess

Output path defaults from `processed_path` in each YAML. Override optional:

```bash
python -m src.preprocess --config configs/junyi.yaml
# Optional explicit output:
python -m src.preprocess --config configs/junyi.yaml --out data/processed/junyi.parquet
```

Canonical columns:
`user_id, item_id, kc_id, timestamp, correct` → parquet under `data/processed/`.

### 3.3 Split check

```bash
python -m src.split_checker --config configs/junyi.yaml
```

Stop and fix any learner overlap or temporal violations before continuing.

---

## 4. Running experiments

Use this section to reproduce paper-facing numbers and optional ablations.
Always run commands from the **repository root** with the venv that has
`pip install -e .` applied.

### 4.0 Step-by-step experiment guide

Follow **A → B** once per machine; then choose **one track** under **C**. Commands below use `configs/junyi.yaml` as an example; swap for `configs/assist2012.yaml` or `configs/xes3g5m.yaml` when working on other benchmarks.

#### A. First-time setup

1. **Clone** this repository. For the optional **pyKT** neural backend, fetch the pinned submodule:
   ```bash
   git submodule update --init --recursive
   ```
2. **Create and activate** a virtual environment ([§2.1](#21-virtual-environment)).
3. **Install core dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```
4. **(Optional)** Install PyTorch + submodule-backed **`pykt-toolkit`** for `--baseline-backend pykt` / `evaluation.baseline_backend: pykt`:
   ```bash
   pip install -e ".[pykt]"
   pip install -e third_party/pykt-toolkit
   ```
   Default YAML in this repo keeps **`diagnostic`** ensembles unless you change `evaluation.baseline_backend` or pass `--baseline-backend pykt` to `baseline_runner`.
5. **Sanity check:** `pytest -q` ([§2.3](#23-tests)).

#### B. Data preparation (per benchmark)

6. **Place raw files** under `data/raw/<dataset>/` as required by each YAML ([§3.1](#31-layout)).
7. **Preprocess** to canonical parquet (repeat for each dataset you need):
   ```bash
   python -m src.preprocess --config configs/junyi.yaml
   ```
8. **Split audit** — fix any failures before graphs or baselines:
   ```bash
   python -m src.split_checker --config configs/junyi.yaml
   ```

#### C. Choose an experiment track

**Track 1 — Smoke / one dataset (Junyi)**  
9. After **A** and **B** for Junyi, run the bundled minimal pipeline:
   ```bash
   bash scripts/run_junyi_minimal.sh
   ```
   On **Windows** without Git Bash:
   ```powershell
   $env:PYTHON = "python"   # optional
   bash scripts/run_junyi_minimal.sh
   ```
   If Bash is unavailable, run the same stages manually in [§5](#5-per-stage-commands).

**Track 2 — Full protocol (all three benchmarks, paper-scale)**  
10. Ensure raw inputs exist for **Junyi, ASSISTments 2012, and XES3G5M**.  
11. Run the orchestrator (RAM/thread notes in [§4.2](#42-full-pipeline-all-benchmarks)):
    ```bash
    chmod +x scripts/run_all_datasets_full.sh
    ./scripts/run_all_datasets_full.sh --server
    ```
    ```powershell
    .\scripts\run_all_datasets_full.ps1 -ServerProfile
    ```
12. **Post-process artefacts:** tables TeX + aggregated report:
    ```bash
    python scripts/generate_paper_artifacts.py
    python -m src.report_generator --out results/reports/
    ```

**Track 3 — Graph ablation H1 (train-only vs full-log diagnostics)**  
13. Requires parquet for each dataset you include (preprocess or Track 2 preprocess stage).  
14. Run (details and flags: [`scripts/GRAPH_ABLATION_EXPERIMENT.md`](scripts/GRAPH_ABLATION_EXPERIMENT.md)):
    ```bash
    chmod +x scripts/run_graph_ablation_experiment.sh
    SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh
    ```
    ```powershell
    .\scripts\run_graph_ablation_experiment.ps1 -ServerProfile
    ```
15. Refresh paper snippets: `python scripts/generate_paper_artifacts.py`.

**Track 4 — Junyi ground-truth cross-validation**  
16. Requires preprocess + graph stages so `data/processed/junyi/kc_name_to_id.json` exists.  
17. Run:
    ```bash
    python scripts/run_gt_cross_validation_junyi.py
    ```

**Track 5 — Regenerate figures / TeX only**  
18. DDR line figures (after `dag_disruption` CSVs exist): `bash scripts/make_all_figures.sh`.  
19. Paper `\input{...}` tables from existing CSVs: `python scripts/generate_paper_artifacts.py`.

**Track 6 — DDR→downstream, anchored + multi-seed (GPU, paper §4.7)**  
20. Requires parquet + fold graphs (`e_pre_train_only.csv`) for the chosen dataset, plus the pyKT backend ([§2.2](#22-dependencies)). Perturbs `E_pre` per operator/strength, retrains a graph-consuming KT model (`--model`, default `gkt`; `gikt` is rejected as it ignores `E_pre`), and links DDR to test AUC. The paper runs three seeds **plus** a near-empty-graph anchor (`edge_drop`/`node_drop` at `p=0.90`) as a manipulation check / positive control ([§4.6](#46-ddrdownstream-anchored--multi-seed-gpu)):
    ```bash
    # Reproduce the paper protocol (seeds 42/17/1234 + p=0.90 anchors, per-seed shards):
    bash scripts/run_ddr_downstream_gkt_multiseed.sh                    # all datasets
    DATASETS="configs/xes3g5m.yaml" bash scripts/run_ddr_downstream_gkt_multiseed.sh  # primary only
    # Windows: .\scripts\run_ddr_downstream_gkt_multiseed.ps1
    python -m scripts.merge_ddr_downstream \
      --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed42.csv \
      --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed17.csv \
      --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed1234.csv
    python -m scripts.plot_ddr_downstream         # tab:ddr-downstream(-gkt) + figure
    ```
    See the GPU playbooks [`docs/DDR_DOWNSTREAM_GKT.md`](docs/DDR_DOWNSTREAM_GKT.md) and [`docs/Q1_GPU_EXPERIMENTS.md`](docs/Q1_GPU_EXPERIMENTS.md).

**Track 7 — Sequence autocorrelation**  
21. Quantifies the "copy-the-previous-answer" shortcut on each benchmark ([§4.7](#47-sequence-autocorrelation-diagnostic)):
    ```bash
    python -m scripts.compute_autocorrelation
    python -m scripts.plot_autocorrelation
    ```

**Track 8 — Significance testing**  
22. Paired \(t\)-test / Wilcoxon over fold-level baseline AUCs ([§4.8](#48-significance-testing)):
    ```bash
    python -m scripts.run_significance_testing
    ```

**Track 9 — Controlled leak injection (two-factor "high-throughput" cell, paper §4.3)**  
23. Deliberately raises contamination throughput on XES3G5M fold 0 and records both the structural audit indicators (builder mass, TBMR) and downstream AUC per backbone ([§4.9](#49-controlled-leak-injection-two-factor-high-throughput-cell)):
    ```bash
    python -m scripts.run_leak_injection     # CPU: structural indicators -> leak_injection.{csv,tex} (Table S17)
    python -m scripts.run_injection_auc      # GPU: downstream AUC on injected graphs -> downstream_auc_injection.tex (S18)
    ```

**Track 10 — Reachability disruption + inferential tables**  
24. A6 reachability-disruption variant (offline, after Track 6 shards exist) and the paper's inferential/appendix tables ([§4.10](#410-inferential-summaries-auc-cis-anova-epochparity)):
    ```bash
    python -m scripts.reachability_disruption \
      --results results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed42.csv --perturb-seed 42
    python -m scripts.bootstrap_auc_ci          # Delta-AUC intervals -> bootstrap_auc_ci.tex + macros (Table S16)
    python scripts/generate_phase_c_tables.py   # significance + ANOVA tables (S11, S19-S20)
    ```

For **manual stage-by-stage** control on a single config (debugging), use the ordered CLI list in [§5](#5-per-stage-commands).

### 4.1 Experiment map

| Goal | Command | Main artefacts |
|------|---------|------------------|
| **Full protocol** (preprocess → graphs → DDR → baselines → cold-start → TeX) | `scripts/run_all_datasets_full.sh` or `scripts/run_all_datasets_full.ps1` | `results/tables/*.csv`/`.tex`, `results/figures/fig_ddr_*.pdf`, `results/reports/p0_diagnostic_report.md` |
| **Smoke / one dataset** | `scripts/run_*_minimal.sh` | Same layout under `data/processed/<dataset>/` and `results/` for that config |
| **Graph ablation** (train-only vs full-log graphs, graph-augmented diagnostics) | `scripts/run_graph_ablation_experiment.sh` or `scripts/run_graph_ablation_experiment.ps1` | `results/tables/graph_ablation_summary.csv`, `graph_ablation.tex`; see `scripts/GRAPH_ABLATION_EXPERIMENT.md` |
| **Junyi GT vs expert DAG** | `python scripts/run_gt_cross_validation_junyi.py` | `results/gt_validation/junyi/*` |
| **DDR→downstream** (GPU; perturb `E_pre`, retrain a graph-KT model via `--model`) | `python -m scripts.ddr_downstream --config configs/<ds>.yaml` then `python -m scripts.plot_ddr_downstream` | `results/tables/ddr_downstream*.csv`/`.tex`, `results/figures/fig_ddr_downstream.pdf` |
| **Sequence autocorrelation** | `python -m scripts.compute_autocorrelation` then `python -m scripts.plot_autocorrelation` | `results/tables/autocorrelation_stats.*`, `results/figures/fig_autocorr_vs_auc.pdf` |
| **Significance testing** | `python -m scripts.run_significance_testing` | `results/tables/significance_tests.csv`, `baseline_cv_template.tex` |
| **Paper tables only** (CSVs already produced) | `python scripts/generate_paper_artifacts.py` | Regenerates `\input{results/tables/...}` snippets |

**Leakage diagnostics** (`ECR_flag`, `ECR_overlap`, `EOC`, `TBVR`) are written to
`results/tables/leakage_metrics.csv` when you run **`python -m src.graph_builder`**
(or console **`p0-graph-build`** after editable install). Fold means are typeset
via `generate_paper_artifacts.py` → `results/tables/leakage_metrics.tex`.

### 4.2 Full pipeline (all benchmarks)

End-to-end on **Junyi, ASSISTments 2012, and XES3G5M**, **per dataset**, in order:

1. `preprocess` (skipped if the dataset parquet exists unless forced)
2. `split_checker`
3. `graph_builder` (fold-wise train-only graphs + leakage row merge)
4. `export_full_log_graph` (full-log prerequisite/similarity exports for ablations / `graph_construction` contrasts)
5. `dag_audit`
6. `dag_disruption`
7. `baseline_runner` — on **Junyi**, `run_all_datasets_full.ps1` appends **`--skip-cold-start`** to limit RAM (cold-start strata skipped; AUC/ACC/NLL still run on val+test). The Bash script calls the baseline **without** that flag; if Junyi exhausts memory on Linux/macOS, rerun step 7 manually with `--skip-cold-start` for `configs/junyi.yaml` only.
8. `cold_start_report`

Then globally: **`scripts/generate_paper_artifacts.py`** and
**`python -m src.report_generator --out results/reports/`**.

**Optional analyses** (run after the per-dataset stages above; see
[§4.6](#46-ddrdownstream-anchored--multi-seed-gpu)–[§4.8](#48-significance-testing)):
sequence autocorrelation (`scripts/compute_autocorrelation.py`), paired
significance (`scripts/run_significance_testing.py`), and the GPU-only
DDR→downstream study (`scripts/ddr_downstream.py`). These are not part of the
default CPU orchestrator.

**Linux / macOS (Git Bash):**

```bash
chmod +x scripts/run_all_datasets_full.sh
./scripts/run_all_datasets_full.sh              # skip preprocess if parquet exists
./scripts/run_all_datasets_full.sh --force-full # rebuild all parquet from raw
./scripts/run_all_datasets_full.sh --server     # BLAS/thread caps for ~32 GB RAM
```

**Windows (PowerShell, repo root):**

```powershell
.\scripts\run_all_datasets_full.ps1
.\scripts\run_all_datasets_full.ps1 -ForceFull      # set FORCE_PREPROCESS=1
.\scripts\run_all_datasets_full.ps1 -ServerProfile   # thread caps + PYTHONHASHSEED
```

Minimal per-dataset scripts (no cross-dataset paper aggregation):

- `scripts/run_junyi_minimal.sh`
- `scripts/run_assist_minimal.sh`
- `scripts/run_xes3g5m_minimal.sh`

### 4.3 Graph ablation (train-only vs full-log, H1)

After parquet exists (from preprocess or the full pipeline), run:

```bash
chmod +x scripts/run_graph_ablation_experiment.sh
SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh
```

```powershell
.\scripts\run_graph_ablation_experiment.ps1 -ServerProfile
```

Requires `graph_ablation.enabled: true` and a `models` list (e.g.\ GKT, GIKT, SKT, DyGKT, DGEKT) in each
`configs/*.yaml`. Full prerequisites, flags (`-SkipGraphBuild`, Junyi RAM notes),
and output filenames are documented in **`scripts/GRAPH_ABLATION_EXPERIMENT.md`**.

### 4.4 DDR sweep and figures

`dag_disruption` sweeps **five augmentation operators** declared in each config
under `augmentation.methods`: four label-agnostic families
(`attr_mask`, `subgraph`, `edge_drop`, `node_drop`) plus the
prerequisite-preserving operator `prereq_preserve`, which removes only
transitively redundant edges (protecting the transitive-reduction backbone) at
the same per-edge budget as `edge_drop`. Each operator is swept over
`augmentation.ps` (`0.05/0.10/0.20/0.30`) with `augmentation.seeds`
(three seeds) on each of three train folds.

```bash
python -m src.dag_disruption --config configs/junyi.yaml
python -m src.dag_disruption --config configs/assist2012.yaml
python -m src.dag_disruption --config configs/xes3g5m.yaml
```

Regenerate the three DDR line PDFs under `results/figures/` (after the CSVs exist):

```bash
bash scripts/make_all_figures.sh
```

### 4.5 Junyi ground-truth cross-validation

Train-only inferred `e_pre_train_only.csv` vs expert DAG; requires preprocess/graph so that
`data/processed/junyi/kc_name_to_id.json` exists:

```bash
python scripts/run_gt_cross_validation_junyi.py
```

For lightweight **directed / undirected overlap at @K** on your own edge
`DataFrame`s (with a `support` column on the inferred side), use
`evaluate_inferred_against_ground_truth` in `src/graph_builder.py` — see
[§5.1](#51-graph_builder-python-api).

Outputs: `results/gt_validation/junyi/` (`overlap_metrics_at_K.csv`,
`fig_pr_curve.pdf`, `gt_validation_table.tex`, etc.).

### 4.6 DDR→downstream, anchored + multi-seed (GPU)

Links the structural **DDR** diagnostic to **downstream KT accuracy**: for each
fold, the train-only prerequisite graph `E_pre` is perturbed by each operator at
strength `p`, the KC–KC adjacency is rebuilt (perturbed `E_pre` ∪ unchanged
`E_sim`), a **graph-consuming KT model is retrained**, and the test AUC is
recorded alongside the DDR of that perturbation. A positive DDR↔(AUC drop)
correlation shows DDR is predictive; `prereq_preserve` (low DDR at matched
budget) should degrade accuracy least.

**Anchored protocol (paper §4.7).** A null DDR↔AUC relationship is ambiguous — it
can mean "DDR does not predict AUC" *or* "this backbone ignores the graph". The
paper therefore gates every interpretation on a **manipulation check / positive
control**: a near-empty-graph **anchor** (`edge_drop` and `node_drop` at
`p=0.90`, DDR ≈ 0.9–1.0). A backbone whose AUC barely moves even under total
destruction is graph-inert (low-reliance anchor; e.g. DGEKT everywhere, GKT on
ASSISTments) and its correlation carries no downstream meaning. GKT on XES3G5M
**passes** the check (near-total destruction costs 0.07–0.09 AUC), so there
DDR↔AUC is interpretable (Pearson r ≈ 0.97). The paper runs **three seeds
(42, 17, 1234)** for CI / noise-floor / ANOVA power. Because
`ddr_downstream.py`/`merge_ddr_downstream.py` dedupe by
`(dataset, model, fold, operator, p)` **without** seed, each seed is written to a
**separate shard**; use the bundled runner rather than a single `--out`:

```bash
# Paper protocol: seeds + p=0.90 anchors, one CSV shard per seed:
bash scripts/run_ddr_downstream_gkt_multiseed.sh                 # all datasets
DATASETS="configs/xes3g5m.yaml" bash scripts/run_ddr_downstream_gkt_multiseed.sh
SEEDS="42 17" MAX_FOLDS=1 bash scripts/run_ddr_downstream_gkt_multiseed.sh  # calibrate
# Windows: .\scripts\run_ddr_downstream_gkt_multiseed.ps1
# Merge the per-seed shards into the paper CSV (one --append per shard):
python -m scripts.merge_ddr_downstream \
  --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed42.csv \
  --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed17.csv \
  --append results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed1234.csv
python -m scripts.reachability_disruption \     # A6 variant (offline, no GPU)
  --results results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed42.csv --perturb-seed 42
```

GPU playbooks with wall-clock estimates: [`docs/DDR_DOWNSTREAM_GKT.md`](docs/DDR_DOWNSTREAM_GKT.md),
[`docs/Q1_GPU_EXPERIMENTS.md`](docs/Q1_GPU_EXPERIMENTS.md).

For a single-config manual run (debugging or a lighter sweep):

**Model choice (`--model`).** Only models that ingest the KC–KC prerequisite
adjacency are valid here, because DDR perturbs `E_pre`:
> - `gkt` (default): canonical graph-KT model; most graph-dependent (strongest
>   expected signal) but the **slowest** to train (small batch, per-step graph
>   propagation).
> - `skt` / `dygkt` / `dgekt`: native lightweight models that consume the **same**
>   adjacency (`graph_npz` → `adj_matrix`) and train **much faster** — use these
>   for a cheaper sweep or to show the DDR↔accuracy link is robust across
>   architectures.
> - `gikt` is **rejected** by design: it consumes the question–concept bipartite
>   graph and ignores `E_pre`, so perturbing `E_pre` would have no effect.

> **Requires** the pyKT backend ([§2.2](#22-dependencies)) and a GPU for
> realistic runtime. GKT is the slowest; an RTX 3090 handles ASSISTments and
> XES3G5M (GKT on Junyi is impractical and is omitted, matching the paper). For a
> faster turnaround, run `--model dgekt` (or `skt`/`dygkt`) instead.

```bash
# Calibrate one fold first to measure per-run time, then scale up:
python -m scripts.ddr_downstream --config configs/assist2012.yaml --max-folds 1

# Full GKT runs (resumable: re-running skips rows already in the output CSV):
python -m scripts.ddr_downstream --config configs/assist2012.yaml
python -m scripts.ddr_downstream --config configs/xes3g5m.yaml

# Faster alternative / cross-architecture check (native lightweight graph model):
python -m scripts.ddr_downstream --config configs/assist2012.yaml --model dgekt

# CPU smoke test of the data path only (no torch / no training):
python -m scripts.ddr_downstream --config configs/assist2012.yaml --dry-run --max-folds 1

# Summary table + correlation + scatter figure:
python -m scripts.plot_ddr_downstream
```

Useful flags: `--model {gkt,skt,dygkt,dgekt}` (default `gkt`),
`--operators edge_drop node_drop prereq_preserve` (default),
`--ps 0.10 0.20 0.30` (default), `--max-folds N`, `--perturb-seed`,
`--experiment-seed`, `--out`. Within a fold the pyKT sequence files are written
once and reused; only the graph `.npz` changes, so each variant costs one model
training run. The output CSV carries a `model` column, so multiple models can
share one file and the plot script summarises each `(dataset, model)` separately.

Outputs: per-seed shards `results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed*.csv`;
merged `results/tables/ddr_downstream.csv` (raw, appended/resumable),
`results/tables/ddr_downstream_summary.csv`, `results/tables/ddr_downstream.tex`
(DGEKT low-reliance anchor, `tab:ddr-downstream`),
`results/tables/ddr_downstream_gkt.tex` (GKT anchored sweep, `tab:ddr-downstream-gkt`),
and `results/figures/fig_ddr_downstream.pdf`/`.png`.

### 4.7 Sequence autocorrelation diagnostic

Measures how predictable each interaction is from the immediately preceding one
(the "copy-the-previous-answer" shortcut that inflates deep-KT AUC on
repetition-heavy logs). Reads `data/processed/<ds>.parquet`:

```bash
python -m scripts.compute_autocorrelation        # KC/item repeat, persistence, lag-1 ρ, prev-answer AUC
python -m scripts.plot_autocorrelation           # KC-repeat rate vs deep-KT / BKT AUC bar chart
```

Outputs: `results/tables/autocorrelation_stats.csv` (+ `.tex`,
table `tab:autocorrelation`) and `results/figures/fig_autocorr_vs_auc.pdf`/`.png`.

### 4.8 Significance testing

Paired \(t\)-test and Wilcoxon signed-rank over fold-level baseline AUCs
(`results/tables/baseline_fold_results.csv`), comparing graph-augmented models
against sequence baselines:

```bash
python -m scripts.run_significance_testing
```

Outputs: `results/tables/significance_tests.csv` and
`results/tables/baseline_cv_template.tex` (table `tab:baseline-cv`); the
public-benchmark paired table `results/tables/significance_tests_public.tex`
(`tab:significance-public`, Table S11) is emitted by
`scripts/generate_phase_c_tables.py`. With three folds the Wilcoxon two-sided
\(p\) cannot fall below `0.25`, so significance claims rest on the paired
\(t\)-test and on the ΔAUC intervals ([§4.10](#410-inferential-summaries-auc-cis-anova-epochparity);
see `paper/submission_APIN/main_APIN.tex`).

### 4.9 Controlled leak injection (two-factor "high-throughput" cell)

Realises the dangerous cell of the two-factor model (paper §4.3): on XES3G5M
fold 0 it deliberately injects test-fold transitions to raise contamination
**throughput**, then shows that the structural audit fires first (builder mass /
TBMR rise monotonically while the learner-disjoint flag stays 0) while downstream
AUC stays near flat for sequence-only backbones and can dip slightly for GKT
(verified Server A GPU run, 2026-07-23).

```bash
python -m scripts.run_leak_injection   # CPU: |E_pre|, builder mass, TBMR vs injection rate
python -m scripts.run_injection_auc    # GPU (pyKT): downstream AUC per backbone on injected graphs
python -m scripts.collect_injection_auc  # regenerate Table S18 from cache
```

Outputs: `results/tables/leak_injection.{csv,tex}` (structural indicators,
Table S17) and `results/tables/downstream_auc_injection.tex`
(Table S18 verified @20%: \textit{simpleKT} ${+}0.0004$, GKT ${-}0.010$,
GIKT ${+}0.0002$ AUC vs clean fold~0).

### 4.10 Inferential summaries (ΔAUC CIs, ANOVA, epoch/parity)

The paper's **primary** inferential evidence is the ΔAUC interval, not a binary
\(p\)-value. Regenerate the appendix inferential/robustness tables:

```bash
python -m scripts.bootstrap_auc_ci            # paired-t 95% (or learner-bootstrap) Delta-AUC
python scripts/generate_phase_c_tables.py     # significance + exploratory ANOVA tables
python scripts/generate_gkt_epoch_ablation.py # GKT 10- vs 30-epoch budget ablation
python scripts/generate_training_parity.py    # compute/epoch parity note
```

| Command | Output(s) | Paper table |
|---|---|---|
| `scripts/bootstrap_auc_ci.py` (`--learner-bootstrap` optional) | `bootstrap_auc_ci.tex`, `bootstrap_ci_macros.tex`, `bootstrap_method_note.tex` | S16 `tab:bootstrap-auc-ci` |
| `scripts/generate_phase_c_tables.py` | `anova_baseline.tex`, `anova_ddr_downstream.tex`, `significance_tests_public.tex` | S19–S20 `tab:anova-*`, S11 `tab:significance-public` |
| `scripts/generate_gkt_epoch_ablation.py` | `gkt_epoch_ablation*.tex` | S21–S22 |
| `scripts/generate_training_parity.py` | `training_parity.tex` | S15 |

> Both ANOVA tables are **exploratory** (n = 3 folds/cell → underpowered); the
> paper reads their \(p\)-values as descriptive and defers a powered ANOVA to the
> completed multi-seed GKT sweep ([§4.6](#46-ddrdownstream-anchored--multi-seed-gpu)).
> `bootstrap_auc_ci.py` prefers pooled `results/predictions/<ds>/fold_*/<model>.parquet`
> (export via `scripts/export_predictions.py`) and falls back to paired-\(t\) over
> `baseline_fold_results.csv`.

---

## 5. Per-stage commands

Typical order for one config (e.g. `configs/junyi.yaml`):

```bash
python -m src.preprocess          --config configs/junyi.yaml
python -m src.split_checker       --config configs/junyi.yaml
python -m src.graph_builder       --config configs/junyi.yaml
python -m src.export_full_log_graph --config configs/junyi.yaml
python -m src.dag_audit           --config configs/junyi.yaml
python -m src.dag_disruption      --config configs/junyi.yaml
python -m src.baseline_runner     --config configs/junyi.yaml   # add --skip-cold-start on Junyi if RAM-limited
python -m src.cold_start_report   --config configs/junyi.yaml
```

Paper-facing tables from already-produced CSVs (dataset stats, baseline TeX,
cold-start summary TeX, artefact index):

```bash
python scripts/generate_paper_artifacts.py
python -m src.report_generator --out results/reports/
```

Most modules accept `--seed` (default `42`) and `--log-level` (`INFO` default).
After `pip install -e .`, optional CLI aliases such as `p0-graph-build` and
`p0-baseline` are defined in `pyproject.toml` and mirror the same flags as
`python -m src.graph_builder` / `src.baseline_runner`.

Fold-specific graph exports live under
`data/processed/<dataset>/fold_<k>/` (e.g. `e_pre_train_only.csv`,
`e_sim_train_only.csv`). Full-log ablation graphs (when exported) live under
`data/processed/<dataset>/full_log/`.

### 5.1 `graph_builder` Python API

Besides the CLI (`python -m src.graph_builder`), `src/graph_builder.py` exposes
train-only helpers for scripts and experiments:

| Function | Role |
|----------|------|
| `build_q_matrix_from_train` | Item–KC table from train interactions only (enforces train-only / single-fold discipline). |
| `infer_prerequisites_from_train` | Directed prerequisite candidates from temporal KC transitions (`support`, `weight`, …). |
| `infer_similarity_edges_from_train` | KC–KC similarity edges (Jaccard or PMI). Fails if `fold` column mixes multiple fold ids. |
| `load_ground_truth_dag` / `dataset_has_independent_prerequisite_graph` | Optional expert DAG loading and dataset capability checks (Junyi / XES-style layouts). |

**Ground-truth overlap at @K** — `evaluate_inferred_against_ground_truth(inferred, expert, top_k_list)` compares a directed inferred edge table to an expert edge list (columns `src_kc`, `dst_kc`). Requirements and behaviour:

- **`inferred`** must include a numeric **`support`** column (same role as in prerequisite outputs). For each `K` in `top_k_list`, the top-*K* rows by descending `support` are evaluated.
- **Directed hit:** same ordered pair as an expert row.
- **Undirected hit:** the unordered pair matches an expert edge in either direction (so an inferred `(a,b)` can match an expert `(b,a)`).
- **Returns** a `DataFrame` with one row per `K`: `k`, `directed_hits`, `undirected_hits`, `directed_precision` (= directed / undirected hits in that top-*K* set, or `0.0` if none), and `directed_recall` (= directed hits / number of expert rows).

See `tests/test_graph_builder_train_only.py` for examples.

---

## 6. Outputs and where they live

| Location | Produced by | Role |
|----------|-------------|------|
| `data/processed/*.parquet` | `preprocess` | Canonical interaction tables |
| `data/processed/<ds>/fold_*/e_pre_train_only.csv` | `graph_builder` | Train-only prerequisite candidates |
| `data/processed/<ds>/fold_*/e_sim_train_only.csv` | `graph_builder` | Train-only similarity edges |
| `results/tables/dataset_stats.csv` (+ `.tex`) | `generate_paper_artifacts.py` | Dataset scale summary |
| `results/tables/graph_stats.csv` | `graph_builder` | Per-fold edge / KC counts |
| `results/tables/leakage_metrics.csv` (+ `.tex`) | `graph_builder` + `generate_paper_artifacts.py` | Fold-wise leakage diagnostics (`ECR_flag`, `ECR_overlap`, …); TeX is fold-mean summary |
| `results/tables/graph_ablation_summary.csv` (+ `.tex`) | `baseline_runner` with `graph_ablation` + artefacts script | Train-only vs full-log graph-augmented diagnostics (models from YAML) |
| `results/tables/dag_audit_summary.csv` (+ `.tex`) | `dag_audit` + `generate_paper_artifacts.py` | Fold-wise DAG audit (`dag_audit` writes CSV; artefacts backfill `n_edges_raw` / `n_edges_pruned` from pruning logs and emit booktabs TeX) |
| `results/reports/<dataset>_dag_report.md` | `dag_audit` | Human-readable audit |
| `results/reports/<dataset>_dag_pruning_log.csv` | `dag_audit` | Pruned edges trail |
| `results/tables/dag_disruption.csv` | `dag_disruption` | Raw DDR rows (fold × aug × p × seed; five operators incl. `prereq_preserve`) |
| `results/tables/dag_disruption_summary.csv` | `dag_disruption` | Means/CIs used in paper DDR table |
| `results/figures/fig_ddr_<dataset>.pdf` | `dag_disruption` | DDR vs `p` line chart per dataset |
| `results/q1/ddr_downstream_gkt/*seed*.csv` | `scripts/run_ddr_downstream_gkt_multiseed.*` | Per-seed GKT DDR→downstream shards (grid + `p=0.90` anchors) |
| `results/tables/ddr_downstream.csv` | `scripts/ddr_downstream.py` + `merge_ddr_downstream.py` | Raw DDR→downstream rows (model × operator × p × fold, test AUC) |
| `results/tables/ddr_downstream_summary.csv` (+ `.tex`) | `scripts/plot_ddr_downstream.py` | Per (dataset, model, operator, p) mean DDR / AUC / AUC drop; table `tab:ddr-downstream` |
| `results/tables/ddr_downstream_gkt.tex` | `scripts/plot_ddr_downstream.py` | GKT anchored sweep (manipulation check); table `tab:ddr-downstream-gkt` |
| `results/figures/fig_ddr_downstream.pdf` (+ `.png`) | `scripts/plot_ddr_downstream.py` | DDR vs downstream AUC drop scatter + correlation |
| `results/tables/leak_injection.{csv,tex}` | `scripts/run_leak_injection.py` | Injection structural indicators (builder mass, TBMR); Table S17 |
| `results/tables/downstream_auc_injection.tex` | `scripts/run_injection_auc.py` | Downstream AUC on injected graphs; Table S18 |
| `results/tables/bootstrap_auc_ci.tex` (+ `bootstrap_ci_macros.tex`) | `scripts/bootstrap_auc_ci.py` | ΔAUC intervals; Table S16 `tab:bootstrap-auc-ci` |
| `results/tables/anova_baseline.tex`, `anova_ddr_downstream.tex`, `significance_tests_public.tex` | `scripts/generate_phase_c_tables.py` | ANOVA (S19–S20) + public significance (S11) |
| `results/tables/autocorrelation_stats.csv` (+ `.tex`) | `scripts/compute_autocorrelation.py` | Sequence-autocorrelation diagnostics; table `tab:autocorrelation` |
| `results/figures/fig_autocorr_vs_auc.pdf` (+ `.png`) | `scripts/plot_autocorrelation.py` | KC-repeat rate vs deep-KT / BKT AUC |
| `results/tables/significance_tests.csv` | `scripts/run_significance_testing.py` | Paired \(t\)-test / Wilcoxon over fold AUCs |
| `results/tables/baseline_cv_template.tex` | `scripts/run_significance_testing.py` | Paired-significance table `tab:baseline-cv` |
| `results/tables/baseline_results.csv` (+ `.tex`) | `baseline_runner` + `generate_paper_artifacts` | Multi-fold means + bootstrap CIs when enabled |
| `results/tables/cold_start_metrics.csv` (+ `.tex`) | `cold_start_report` + artefacts script | Stratum summaries |
| `results/tables/cold_start_by_stratum.tex` | `generate_paper_artifacts.py` | Paper table from `cold_start_metrics.csv` (default: fold~0 \textit{simpleKT}) |
| `results/reports/cold_start_report.md` | `cold_start_report` | Narrative cold-start report |
| `results/provenance/*inject*_result.json` | Verified GPU runs | Curated fold-0 injection evidence used by the cross-reference audit |
| `results/reports/paper_artifact_index.md` | `generate_paper_artifacts.py` | Index of tables/figures/reports |
| `results/reports/p0_diagnostic_report.md` | `report_generator` | Aggregated diagnostic markdown |
| `results/gt_validation/junyi/*` | `run_gt_cross_validation_junyi.py` | GT overlap metrics, PR curve, TeX snippet |
| `logs/leakage_audit_log.csv` | Multiple stages | Edge provenance audit trail |
| `logs/experiment_log.csv` | Run hooks | Run ledger (if configured) |

---

## 7. Paper artefacts, reproduction map, and LaTeX build

- **Manuscript (canonical / submission-ready):** `paper/submission_APIN/main_APIN.tex`
  with bibliography `paper/submission_APIN/refs_APIN.bib`. Class: Springer Nature `sn-jnl.cls` with
  option `sn-mathphys-num` only (**no** `referee`); the `.cls`/`.bst` files
  live beside the manuscript in `paper/submission_APIN/`.
- **Locked scope (2026-07-18):** Table **S22 / pooled nine-fold GKT30 removed**
  (S21 exploratory only; CI includes zero); B03 wording uses *associated with*
  (not causal *inflates*). See `paper/submission_APIN/BUILD_INSTRUCTIONS.txt`.
- **Do not maintain parallel flat trees** (e.g. `Leakage_Controlled_*` was
  merged into `submission_APIN/` and removed).

### 7.1 Paper table/figure → how to reproduce

Run the pipeline (§4) first; then each artefact is regenerated by the script
below. Table numbers `Sxx` are the appendix labels used in the manuscript.

| Paper table / figure (label) | Regenerated by | Source CSV / stage |
|---|---|---|
| Dataset stats (`tab:dataset-stats`) | `generate_paper_artifacts.py` | `preprocess` |
| Baselines, per dataset (S1–S5, `tab:baseline-*`) | `baseline_runner` → `generate_paper_artifacts.py` | `baseline_results.csv` |
| Leakage metrics (`tab:leakage-metrics`) | `graph_builder` → `generate_paper_artifacts.py` | `leakage_metrics.csv` |
| DAG audit (`tab:dag-audit`) | `dag_audit` → `generate_paper_artifacts.py` | `dag_audit_summary.csv` |
| Graph ablation, train-only vs full-log (`tab:graph-ablation`, S6–S10) | `run_graph_ablation_experiment.*` → `generate_paper_artifacts.py` | `graph_ablation_summary.csv` |
| DDR sweep (`tab:ddr-sweep`, `tab:ddr-raw`) | `dag_disruption` → `generate_paper_artifacts.py` | `dag_disruption*.csv` |
| **Controlled injection (S17–S18)** | `run_leak_injection.py` (+ `run_injection_auc.py`) | fold-0 injected graphs |
| **DDR→downstream, DGEKT anchor (`tab:ddr-downstream`)** | `ddr_downstream.py` → `plot_ddr_downstream.py` | `ddr_downstream.csv` |
| **DDR→downstream, GKT anchored+multiseed (`tab:ddr-downstream-gkt`)** | `run_ddr_downstream_gkt_multiseed.*` → `merge_ddr_downstream.py` → `plot_ddr_downstream.py` | `results/q1/ddr_downstream_gkt/*seed*.csv` |
| **ΔAUC intervals (S16, `tab:bootstrap-auc-ci`)** | `bootstrap_auc_ci.py` | predictions parquet or `baseline_fold_results.csv` |
| Paired significance, public (S11, `tab:significance-public`) | `generate_phase_c_tables.py` | `baseline_fold_results.csv` |
| Exploratory ANOVA (S19–S20, `tab:anova-*`) | `generate_phase_c_tables.py` | fold-level AUC / DDR-downstream |
| GKT epoch ablation (S21–S22) | `generate_gkt_epoch_ablation.py` | epoch-extended runs |
| Training parity (S15) | `generate_training_parity.py` | parity runs |
| Autocorrelation (`tab:autocorrelation`) | `compute_autocorrelation.py` → `plot_autocorrelation.py` | `data/processed/<ds>.parquet` |
| Cold-start strata (`tab:cold-start-*`, S12–S13) | `cold_start_report` → `generate_paper_artifacts.py` | `cold_start_metrics.csv` |
| Ground-truth CV (`tab:gt-validation*`, S14) | `run_gt_cross_validation_junyi.py` | `results/gt_validation/junyi/` |
| KC-graph figures | `plot_kt_graph_figures.py` | fold-0 exports |
| DDR curves `fig_ddr_*` | `dag_disruption` / `make_all_figures.sh` | `dag_disruption_summary.csv` |

> Metric naming: the CSV columns are `ecr_flag`, `ecr_overlap`, `eoc`, `tbvr`
> (`src/leakage_metrics.py`). The paper uses `ECR_flag` as the structural gate and
> additionally reports **throughput** as builder mass / TBMR in the injection
> experiment ([§4.9](#49-controlled-leak-injection-two-factor-high-throughput-cell)).

### 7.2 Build the PDF

Compile the canonical package only:

```bash
cd paper/submission_APIN
pdflatex -interaction=nonstopmode main_APIN.tex && bibtex main_APIN && \
  pdflatex main_APIN.tex && pdflatex main_APIN.tex
```

```powershell
cd paper\submission_APIN
pdflatex -interaction=nonstopmode main_APIN.tex
bibtex main_APIN
pdflatex -interaction=nonstopmode main_APIN.tex
pdflatex -interaction=nonstopmode main_APIN.tex
```

`sn-jnl.cls` and the Springer `.bst` files ship in `paper/submission_APIN/`;
if missing, fetch the
[Springer Nature LaTeX template](https://www.springernature.com/gp/authors/campaigns/latex-author-support).
After regenerating tables/figures under `results/`, copy the needed snippets
into `paper/submission_APIN/` (or re-run your flat-package sync script) before
rebuilding.

### 7.3 Full regeneration recipe

1. CPU protocol: `./scripts/run_all_datasets_full.sh --server` (or `.ps1`).
2. H1 graph ablation: `SERVER_PROFILE=1 ./scripts/run_graph_ablation_experiment.sh`, then `python scripts/generate_paper_artifacts.py`.
3. GPU: `bash scripts/run_ddr_downstream_gkt_multiseed.sh` → `merge_ddr_downstream.py`; `run_leak_injection.py` + `run_injection_auc.py`.
4. Inferential tables: `bootstrap_auc_ci.py`, `generate_phase_c_tables.py`, `generate_gkt_epoch_ablation.py`, `generate_training_parity.py` ([§4.10](#410-inferential-summaries-auc-cis-anova-epochparity)).
5. Optional GT: `python scripts/run_gt_cross_validation_junyi.py`.
6. Build the PDF ([§7.2](#72-build-the-pdf)).

---

## 8. Troubleshooting

**Import errors after `pip install -e .`** — Run commands from the repo root;
ensure the active interpreter is the venv you installed into.

**Split checker failures** — Inspect `python -m src.split_checker --config …`
(do not bypass assertions). Raw CSV paths and schema mapping in YAML must
match your download.

**`leakage_audit_log.csv` contains non-train-only edges** — Treat as a hard
protocol violation: fix the offending stage and rerun from a clean state.

**DDR plots missing** — Run `python -m src.dag_disruption --config configs/<ds>.yaml`
for each dataset, or `bash scripts/make_all_figures.sh`.

**Junyi GT CV missing mapping** — Ensure preprocess/graph ran so
`data/processed/junyi/kc_name_to_id.json` exists; expert CSV paths match
`run_gt_cross_validation_junyi.py` defaults.

**Baseline RAM / time** — Reduce folds in YAML (`split.n_folds`) or disable
heavy models in `configs/*.yaml` under `baselines:` while debugging structure-only stages.

**DDR→downstream (`ddr_downstream.py`)** — Needs `data/processed/<ds>/fold_*/e_pre_train_only.csv`
(run `graph_builder` first) and the pyKT backend ([§2.2](#22-dependencies)). It is GPU-bound;
calibrate with `--max-folds 1` to estimate per-run time before scaling up. Use `--dry-run`
(no torch) to validate the data path on CPU. Default `--model gkt` is the slowest; switch to
`--model dgekt`/`skt`/`dygkt` (native, lighter, same `E_pre` adjacency) for a faster sweep.
`--model gikt` is rejected because GIKT uses the question–concept graph, not `E_pre`.
Junyi GKT is impractical and intentionally omitted. Runs are **resumable**: re-running skips
`(dataset, model, fold, operator, p)` rows already in the output CSV.

**`baseline_backend=pykt` import errors** — Run `git submodule update --init --recursive`,
then `pip install -e ".[pykt]"` and `pip install -e third_party/pykt-toolkit`
(see [§2.2](#22-dependencies)).

**Junyi baseline RAM (Linux/macOS full pipeline)** — `scripts/run_all_datasets_full.sh` invokes `baseline_runner` without `--skip-cold-start`; Windows `run_all_datasets_full.ps1` adds `--skip-cold-start` for Junyi.
If you hit OOM on Bash/WSL, rerun only that stage:
`python -m src.baseline_runner --config configs/junyi.yaml --skip-cold-start`.

---

## 9. Project structure

```
leakage-controlled-kt-audit/
├── README.md
├── LICENSE                 # MIT (code); datasets remain under provider ToU
├── CITATION.cff            # machine-readable citation metadata
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── junyi.yaml
│   ├── assist2012.yaml
│   ├── xes3g5m.yaml
│   └── synthetic_c{2,5}.yaml
├── paper/
│   └── submission_APIN/        # CANONICAL submission-ready package (main_APIN.tex + assets)
├── docs/
│   ├── DDR_DOWNSTREAM_GKT.md   # GPU playbook: anchored DDR->downstream (GKT)
│   └── Q1_GPU_EXPERIMENTS.md   # GPU experiment tracking / wall-clock notes
├── data/
│   ├── raw/              # gitignored — place benchmarks here
│   └── processed/        # gitignored — parquet + fold exports
├── src/
│   ├── preprocess.py
│   ├── split_checker.py
│   ├── graph_builder.py
│   ├── dag_audit.py
│   ├── dag_disruption.py
│   ├── cold_start_report.py
│   ├── export_full_log_graph.py
│   ├── baseline_runner.py
│   ├── pykt_engine.py         # pyKT training loop (GKT/GIKT/… ; optional backend)
│   ├── pykt_export.py         # parquet splits → pyKT sequence CSVs + dense id maps
│   ├── pykt_graph_matrix.py   # edge CSVs → row-normalised GKT adjacency .npz
│   ├── gt_cross_validation.py
│   ├── report_generator.py
│   └── io_utils.py
├── third_party/
│   ├── README.md
│   └── pykt-toolkit/          # git submodule (optional pyKT backend)
├── scripts/
│   ├── run_junyi_minimal.sh
│   ├── run_assist_minimal.sh
│   ├── run_xes3g5m_minimal.sh
│   ├── run_all_datasets_full.sh
│   ├── run_all_datasets_full.ps1
│   ├── run_graph_ablation_experiment.sh
│   ├── run_graph_ablation_experiment.ps1
│   ├── GRAPH_ABLATION_EXPERIMENT.md
│   ├── make_all_figures.sh
│   ├── generate_paper_artifacts.py
│   ├── run_gt_cross_validation_junyi.py
│   ├── ddr_downstream.py             # DDR→downstream study, --model gkt/skt/dygkt/dgekt (§4.6)
│   ├── run_ddr_downstream_gkt_multiseed.sh / .ps1  # seeds + p=0.90 anchors (§4.6)
│   ├── merge_ddr_downstream.py       # merge per-seed shards
│   ├── plot_ddr_downstream.py        # DDR vs AUC-drop summary + figure
│   ├── reachability_disruption.py    # A6 reachability-disruption variant
│   ├── run_leak_injection.py         # controlled injection: builder mass / TBMR (§4.9)
│   ├── run_injection_auc.py          # downstream AUC on injected graphs (§4.9)
│   ├── bootstrap_auc_ci.py           # ΔAUC intervals, Table S16 (§4.10)
│   ├── generate_phase_c_tables.py    # significance + ANOVA tables (§4.10)
│   ├── generate_gkt_epoch_ablation.py / generate_training_parity.py  # S21 / S15
│   ├── compute_autocorrelation.py    # sequence-autocorrelation stats (§4.7)
│   ├── plot_autocorrelation.py       # repeat-rate vs AUC bar chart
│   ├── run_significance_testing.py   # paired t-test / Wilcoxon (§4.8)
│   └── …
├── tests/
├── results/
│   ├── tables/
│   ├── figures/
│   ├── reports/
│   └── gt_validation/junyi/   # after GT script
└── logs/
```

---

## 10. Citation, licence, and contact

### 10.1 Preferred citation (article)

If you use this software or protocol, please cite the companion article:

> Tuan Dao Minh, Khanh-Trinh Nguyen, Duong Nguyen Tien, Quoc Khanh Ngo,
> Van-Hau Nguyen, Le Hoang Son.
> *Leakage-Controlled Concept Graph Construction and Cold-Start Diagnostic
> Protocol for Knowledge Tracing*.
> Applied Intelligence (Springer Nature), 2026. Manuscript under submission.

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).
Bibliography source used in the paper: `paper/submission_APIN/refs_APIN.bib`
(volume / pages / DOI will be updated upon publication).

```bibtex
@article{dao2026leakage,
  title   = {Leakage-Controlled Concept Graph Construction and Cold-Start
             Diagnostic Protocol for Knowledge Tracing},
  author  = {Dao Minh, Tuan and Nguyen, Khanh-Trinh and Nguyen Tien, Duong
             and Ngo, Quoc Khanh and Nguyen, Van-Hau and Le, Hoang Son},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Manuscript under submission},
  publisher = {Springer Nature}
}
```

### 10.2 Licence

- **Code in this repository** is released under the [MIT License](LICENSE).
- **pyKT** (optional submodule `third_party/pykt-toolkit`) is MIT; see upstream
  [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit).
- **Datasets** are **not** redistributed here. Obtain them under each provider’s
  terms (Junyi / PSLC DataShop; ASSISTments 2012–2013; XES3G5M MIT; see
  manuscript Data availability and [§3](#3-data-download-and-preparation)).

### 10.3 Contact and contributions

- **Corresponding author:** Van-Hau Nguyen — `nvhau66@gmail.com`
- **Technical contact:** Tuan Dao Minh — `tuanymc@utehy.edu.vn`
- Repository: https://github.com/edu-risk-lab/leakage-controlled-kt-audit

Bug reports and reproducibility issues are welcome via GitHub Issues. For
substantial scientific changes, open an issue before a large pull request.
