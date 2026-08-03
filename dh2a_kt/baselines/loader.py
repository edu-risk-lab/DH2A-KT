"""Load P0's already-computed baseline results for direct reuse (RQ1
comparison column) instead of retraining BKT/DKT/AKT/simpleKT/GKT/GIKT/SKT/
DyGKT/DGEKT on the RTX 3090. This is the single biggest compute saving in
Idea D — see docs/idea-D-plan.md section 6.

This module is fully functional (plain pandas filtering over the vendored
CSVs) — unlike models/dh2_kt.py and the Tier-2 agents, there is nothing left
to implement here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "baselines" / "p0_reused"

VALID_DATASETS = {"assist2012", "junyi", "xes3g5m", "synthetic_c2", "synthetic_c5"}
VALID_MODELS = {"bkt", "akt", "dkt", "simplekt", "gkt", "gikt", "skt", "dygkt", "dgekt"}


def _load(name: str) -> pd.DataFrame:
    path = _BASELINE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected P0's vendored results at "
            f"{_BASELINE_DIR} — see baselines/p0_reused/PROVENANCE.md."
        )
    df = pd.read_csv(path)
    # P0's export includes one artefact row (bkt, status=pending_data, no dataset) — drop it.
    return df[df["dataset"].notna()] if "dataset" in df.columns else df


def load_p0_baseline_summary(
    dataset: str,
    *,
    model: str | None = None,
    graph_construction: str = "train_only",
) -> pd.DataFrame:
    """Aggregate (3-fold mean ± CI) baseline rows — mirrors P0 Table 8/9.

    Args:
        dataset: one of VALID_DATASETS (xes3g5m is P0's primary benchmark).
        model: optional filter, one of VALID_MODELS.
        graph_construction: "train_only" (leakage-controlled, the protocol
            default — use this for any headline comparison) or "full_log"
            (P0's leakage-ablation condition; only use for reproducing that
            specific ablation, not as a general baseline).
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(VALID_DATASETS)}")
    df = _load("baseline_results.csv")
    df = df[(df["dataset"] == dataset) & (df["graph_construction"] == graph_construction)]
    if model is not None:
        if model not in VALID_MODELS:
            raise ValueError(f"Unknown model {model!r}; expected one of {sorted(VALID_MODELS)}")
        df = df[df["model"] == model]
    return df.reset_index(drop=True)


def load_p0_baseline_folds(
    dataset: str,
    *,
    model: str | None = None,
) -> pd.DataFrame:
    """Fold-level rows (for paired significance tests against a new DH2-KT
    result under the same 3-fold learner-based split protocol, seeds 42-44).
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(VALID_DATASETS)}")
    df = _load("baseline_fold_results.csv")
    df = df[df["dataset"] == dataset]
    if model is not None:
        df = df[df["model"] == model]
    return df.reset_index(drop=True)


def compare_against_p0(dh2_kt_auc: float, dataset: str, model: str = "simplekt") -> dict:
    """Convenience: report a DH2-KT AUC alongside P0's reused baseline for the
    same dataset/model, with the delta. Does not run any statistical test —
    use load_p0_baseline_folds() + your own paired test for that (P0 itself
    only reports Wilcoxon at p_min=0.25 under 3-fold CV; do not over-claim
    significance from 3 folds — see P0 Section 4.4/5.3)."""
    row = load_p0_baseline_summary(dataset, model=model, graph_construction="train_only")
    if row.empty:
        raise ValueError(f"No P0 baseline found for dataset={dataset} model={model}")
    p0_auc = float(row.iloc[0]["auc"])
    return {
        "dataset": dataset,
        "reference_model": model,
        "p0_auc": p0_auc,
        "dh2_kt_auc": dh2_kt_auc,
        "delta_auc": dh2_kt_auc - p0_auc,
    }
