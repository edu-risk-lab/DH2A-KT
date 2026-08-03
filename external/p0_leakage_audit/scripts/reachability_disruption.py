#!/usr/bin/env python3
"""Offline reachability-disruption (A6) add-on for the DDR->downstream sweep.

The DDR downstream runner records edge-level ``ddr`` and test ``auc`` per
``(dataset, model, fold, operator, p)`` but not the *reachability* disruption a
graph-consuming backbone may actually care about. This script reconstructs each
perturbation deterministically from the train-only prerequisite graph (the same
operator/strength/seed the runner used) and computes, for every row:

  - ``ddr_check``           : DDR recomputed from the reconstruction (self-check
                              against the recorded ``ddr``; large mismatch means
                              the ``--perturb-seed`` does not match the run).
  - ``reach_disruption``    : ``1 - reachability_f1(original, perturbed)``.

It then reports, per ``(dataset, model)``, the Pearson correlation of test-AUC
drop with raw DDR *and* with reachability disruption, side by side. This is the
A6 contrast: does precedence-level disruption track accuracy loss better than
edge-edit distance? No GPU / no retraining required.

Usage
-----
# one seed's shard (perturb-seed must match the run that produced it)
python -m scripts.reachability_disruption \
    --results results/q1/ddr_downstream_gkt/ddr_downstream_gkt_seed42.csv \
    --perturb-seed 42

# the paper bundle (default perturb-seed 42, matching docs/DDR_DOWNSTREAM_GKT.md)
python -m scripts.reachability_disruption --results results/tables/ddr_downstream.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.dag_disruption import (
    apply_edge_drop,
    apply_node_drop,
    apply_prereq_preserve,
    apply_subgraph_sampling,
    apply_attribute_mask,
    compute_dag_disruption_rate,
    reachability_f1,
)

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

OPERATORS = {
    "edge_drop": apply_edge_drop,
    "node_drop": apply_node_drop,
    "prereq_preserve": apply_prereq_preserve,
    "subgraph": apply_subgraph_sampling,
    "attr_mask": apply_attribute_mask,
}

_EMPTY = pd.DataFrame(columns=["src_kc", "dst_kc", "weight"])


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and (best-effort) two-sided p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy import stats

        return r, float(stats.pearsonr(x, y)[1])
    except Exception:  # scipy optional
        return r, float("nan")


def _load_original(data_root: Path, dataset: str, fold: int) -> pd.DataFrame:
    path = data_root / dataset / f"fold_{fold}" / "e_pre_train_only.csv"
    if not path.exists():
        logger.warning("missing train-only graph: %s (rows for this fold skipped)", path)
        return _EMPTY.copy()
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="ddr_downstream CSV (one seed shard or merged bundle).")
    parser.add_argument("--perturb-seed", type=int, default=42, help="Augmentation seed the run used (must match).")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV (default: <results>_with_reach.csv).")
    parser.add_argument("--ddr-tol", type=float, default=2e-3, help="Tolerance for the DDR reconstruction self-check.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    df = pd.read_csv(args.results)
    if "p" in df.columns:
        df["p"] = df["p"].astype(float).round(4)
    required = {"dataset", "fold", "operator", "p"}
    if not required.issubset(df.columns):
        raise SystemExit(f"results CSV missing columns {required - set(df.columns)}")

    # Reconstruct each distinct (dataset, fold, operator, p) once.
    reach: dict[tuple, float] = {}
    ddr_chk: dict[tuple, float] = {}
    keys = df[["dataset", "fold", "operator", "p"]].drop_duplicates()
    originals: dict[tuple, pd.DataFrame] = {}
    for row in keys.itertuples(index=False):
        dataset, fold, operator, p = str(row.dataset), int(row.fold), str(row.operator), round(float(row.p), 4)
        ckey = (dataset, fold)
        if ckey not in originals:
            originals[ckey] = _load_original(args.data_root, dataset, fold)
        original = originals[ckey]
        k = (dataset, fold, operator, p)
        if operator == "none" or p == 0.0:
            reach[k], ddr_chk[k] = 0.0, 0.0
            continue
        if operator not in OPERATORS:
            reach[k], ddr_chk[k] = float("nan"), float("nan")
            continue
        perturbed = OPERATORS[operator](original, float(p), int(args.perturb_seed))
        reach[k] = 1.0 - reachability_f1(original, perturbed)
        ddr_chk[k] = compute_dag_disruption_rate(original, perturbed)

    df["reach_disruption"] = [reach[(str(r.dataset), int(r.fold), str(r.operator), round(float(r.p), 4))] for r in df.itertuples(index=False)]
    df["ddr_check"] = [ddr_chk[(str(r.dataset), int(r.fold), str(r.operator), round(float(r.p), 4))] for r in df.itertuples(index=False)]

    # Self-check: reconstructed DDR should match the recorded DDR for this seed.
    if "ddr" in df.columns:
        delta = (df["ddr"].astype(float) - df["ddr_check"].astype(float)).abs()
        bad = df[delta > args.ddr_tol]
        if len(bad):
            logger.warning(
                "%d/%d rows: reconstructed DDR != recorded DDR by >%.0e. "
                "Likely --perturb-seed mismatch (file made with a different seed). "
                "Reachability values for those rows are unreliable.",
                len(bad), len(df), args.ddr_tol,
            )
        else:
            logger.info("DDR reconstruction self-check passed for all %d rows (seed=%d).", len(df), args.perturb_seed)

    # AUC drop relative to the operator=none / DDR=0 baseline, per replicate.
    group_cols = [c for c in ["dataset", "model", "fold", "split_seed"] if c in df.columns]
    if "auc" in df.columns:
        base = df[df["operator"] == "none"][group_cols + ["auc"]].rename(columns={"auc": "baseline_auc"})
        df = df.merge(base, on=group_cols, how="left")
        df["auc_drop"] = df["baseline_auc"].astype(float) - df["auc"].astype(float)

    out = args.out or args.results.with_name(args.results.stem + "_with_reach.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows, +reach_disruption,+ddr_check%s)", out, len(df), ",+auc_drop" if "auc_drop" in df.columns else "")

    # A6 contrast: DDR vs reach disruption as predictors of AUC drop.
    if "auc_drop" in df.columns:
        pert = df[(df["operator"] != "none") & df["auc_drop"].notna()]
        model_col = "model" if "model" in pert.columns else None
        gb = pert.groupby(["dataset"] + ([model_col] if model_col else []))
        print("\n=== A6 contrast: Pearson correlation with AUC drop ===")
        print(f"{'dataset':<12} {'model':<8} {'n':>3}  {'r(DDR)':>8} {'p':>7}   {'r(reach)':>9} {'p':>7}")
        for gkey, part in gb:
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            dataset = gkey[0]
            model = gkey[1] if model_col else "-"
            rd, pd_ = _pearson(part["ddr"].to_numpy(float), part["auc_drop"].to_numpy(float))
            rr, pr = _pearson(part["reach_disruption"].to_numpy(float), part["auc_drop"].to_numpy(float))
            print(f"{str(dataset):<12} {str(model):<8} {len(part):>3}  {rd:>8.3f} {pd_:>7.3f}   {rr:>9.3f} {pr:>7.3f}")
        print("\nNote: with <3 seeds per cell these are descriptive; report alongside the")
        print("manipulation-check anchor (operator=edge_drop, p>=0.90) before interpreting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
