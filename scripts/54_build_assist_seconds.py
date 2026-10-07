"""Rebuild ASSISTments 2012 at one-second timestamp resolution.

The P0 export (``data/processed/assist2012.parquet``) stores ``start_time``
as ``datetime_int64 // 1e9``. When pandas parses the strings at microsecond
resolution that is unix seconds // 1000, so gaps are multiples of 1,000 s and
rows inside one window are ordered by item id. This script reruns the P0
preprocessing with a unit-safe parse and writes a separate dataset,
``assist2012_sec``; the original parquet and its graphs are left untouched.

Steps (P0 code at the pinned commit, only ``_parse_timestamp`` replaced):
  1. raw CSV -> ``external/p0_leakage_audit/data/processed/assist2012_sec.parquet``
  2. per learner fold: train-only E_pre / E_sim exactly as ``src.graph_builder``
     -> ``data/processed/assist2012_sec/fold_{f}/e_pre_train_only.csv``
  3. checks: same row count as the old parquet, and the old timestamps equal
     ``new // 1000`` as multisets (confirms the 1,000 s resolution).

    python scripts/54_build_assist_seconds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
P0_ROOT = REPO / "external" / "p0_leakage_audit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(P0_ROOT))

import src.preprocess as p0_pre  # noqa: E402
from src.graph_builder import (  # noqa: E402
    build_q_matrix_from_train,
    infer_prerequisites_from_train,
    infer_similarity_edges_from_train,
)
from src.io_utils import dump_csv, load_interactions, load_yaml  # noqa: E402
from src.split_checker import learner_based_folds  # noqa: E402

OLD_CFG = P0_ROOT / "configs" / "assist2012.yaml"
NEW_CFG = REPO / "configs" / "p0" / "assist2012_sec.yaml"
OUT_JSON = REPO / "results" / "tables" / "assist2012_sec_build.json"


def _parse_timestamp_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype("int64")
    numeric_try = pd.to_numeric(series, errors="coerce")
    if numeric_try.notna().all():
        return numeric_try.astype("int64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        bad_examples = series[parsed.isna()].head(5).tolist()
        raise ValueError(f"Could not parse timestamp examples: {bad_examples}")
    epoch = pd.Timestamp(0, tz="UTC")
    return ((parsed - epoch) // pd.Timedelta(seconds=1)).astype("int64")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(NEW_CFG)
    old_cfg = load_yaml(OLD_CFG)
    raw_cfg = {**cfg, "raw_interactions_file": str(P0_ROOT / cfg["raw_interactions_file"])}
    out_parquet = P0_ROOT / cfg["processed_path"]

    p0_pre._parse_timestamp = _parse_timestamp_seconds
    df = p0_pre._load_and_normalise_interactions(raw_cfg)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet} rows={len(df)}")

    old = load_interactions(P0_ROOT / old_cfg["processed_path"])
    new_ts = df["timestamp"].to_numpy(dtype=np.int64)
    old_ts = old["timestamp"].to_numpy(dtype=np.int64)
    same_rows = len(old) == len(df)
    floor_match = bool(same_rows and np.array_equal(np.sort(old_ts), np.sort(new_ts // 1000)))
    same_users = bool(set(old["user_id"].unique()) == set(df["user_id"].unique()))

    df = load_interactions(out_parquet)
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    graph_cfg = cfg.get("graph", {})
    folds = []
    for fold, split_seed, splits in learner_based_folds(df, ratios, cfg.get("split", {}), default_seed=args.seed):
        train = splits["train"].copy()
        train["split"] = "train"
        train["fold"] = fold
        q_train = build_q_matrix_from_train(train)
        pre = infer_prerequisites_from_train(
            train,
            q_train,
            max_edges=int(graph_cfg.get("e_pre_max_edges", 5000)),
            top_k_per_node=int(graph_cfg.get("e_pre_top_k_per_node", 10)),
            support_quantile=float(graph_cfg.get("e_pre_support_quantile", 0.90)),
        )
        sim = infer_similarity_edges_from_train(
            train,
            q_train,
            method=graph_cfg.get("e_sim_method", "jaccard"),
            threshold=float(graph_cfg.get("e_sim_threshold", 0.1)),
        )
        out_dir = P0_ROOT / "data" / "processed" / cfg["dataset"] / f"fold_{fold}"
        dump_csv(pre, out_dir / "e_pre_train_only.csv")
        dump_csv(sim, out_dir / "e_sim_train_only.csv")
        dump_csv(pre, out_dir / "edges_train_only.csv")
        old_pre_path = P0_ROOT / "data" / "processed" / old_cfg["dataset"] / f"fold_{fold}" / "e_pre_train_only.csv"
        overlap = None
        if old_pre_path.exists():
            old_pre = pd.read_csv(old_pre_path)
            a = set(zip(old_pre["src_kc"], old_pre["dst_kc"]))
            b = set(zip(pre["src_kc"], pre["dst_kc"]))
            overlap = len(a & b) / max(len(a | b), 1)
        folds.append(
            {
                "fold": fold,
                "split_seed": split_seed,
                "n_train_rows": int(len(train)),
                "n_e_pre": int(len(pre)),
                "n_e_sim": int(len(sim)),
                "e_pre_jaccard_vs_1000s": overlap,
            }
        )
        print(json.dumps(folds[-1]))

    payload = {
        "dataset": cfg["dataset"],
        "n_rows": int(len(df)),
        "same_row_count_as_1000s": same_rows,
        "same_learners_as_1000s": same_users,
        "old_timestamps_equal_new_floor_div_1000": floor_match,
        "median_timestamp_new": float(np.median(new_ts)),
        "median_timestamp_old": float(np.median(old_ts)),
        "folds": folds,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "folds"}))
    print(f"wrote {OUT_JSON}")
    if not (same_rows and same_users):
        print("WARNING: row or learner set differs from the 1,000 s export")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
