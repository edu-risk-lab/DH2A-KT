#!/usr/bin/env python3
"""Ingest FoundationalASSIST CSVs into P0-canonical parquet + build train-only E_pre.

Usage:
    python scripts/12_ingest_foundational_assist.py \\
        --raw-dir data/FoundationalASSIST --fold 0

Writes:
  external/p0_leakage_audit/data/processed/foundational_assist.parquet
  external/p0_leakage_audit/data/processed/foundational_assist_extended.parquet
  external/p0_leakage_audit/data/processed/foundational_assist/fold_{f}/e_pre_train_only.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "p0_leakage_audit"))

from dh2a_kt.p0_bridge import (  # noqa: E402
    audit_dag,
    build_q_matrix_from_train,
    infer_prerequisites_from_train,
    learner_based_folds,
    prune_cycles,
)


def _stable_int_id(value: str | int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


def ingest(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(raw_dir / "Interactions.csv")
    skills = pd.read_csv(raw_dir / "Skills.csv")

    # One primary skill per problem (first row) for canonical KT schema.
    skill_map = (
        skills.sort_values(["problem_id", "skill_id"])
        .drop_duplicates("problem_id", keep="first")
        .set_index("problem_id")["skill_id"]
        .astype("int64")
    )

    df = interactions.copy()
    if "user_id" not in df.columns and "user_xid" in df.columns:
        df = df.rename(columns={"user_xid": "user_id"})
    df["kc_id"] = df["problem_id"].map(skill_map)
    before = len(df)
    df = df.dropna(subset=["kc_id", "end_time", "discrete_score"]).copy()
    print(f"Dropped {before - len(df)} rows without skill/time/score")

    df["user_id"] = df["user_id"].map(_stable_int_id).astype("int64")
    df["item_id"] = df["problem_id"].astype("int64")
    df["kc_id"] = df["kc_id"].astype("int64")
    df["timestamp"] = (
        pd.to_datetime(df["end_time"], utc=True, format="ISO8601").astype("int64") // 10**9
    ).astype("int64")
    df["correct"] = (df["discrete_score"].astype(float) >= 0.999).astype("int64")
    df["hint_count"] = df["hint_count"].fillna(0).astype("int64")
    df["saw_answer"] = df["saw_answer"].fillna(False).astype(bool)
    df["hint_used"] = (df["hint_count"] > 0).astype("int64")

    canonical = df[["user_id", "item_id", "kc_id", "timestamp", "correct"]].copy()
    extended = df[
        ["user_id", "item_id", "kc_id", "timestamp", "correct", "hint_count", "hint_used", "saw_answer"]
    ].copy()
    print(
        f"Ingested n={len(canonical)} users={canonical['user_id'].nunique()} "
        f"items={canonical['item_id'].nunique()} kcs={canonical['kc_id'].nunique()} "
        f"hint_rate={extended['hint_used'].mean():.3f}"
    )
    return canonical, extended


def build_e_pre_for_fold(
    interactions: pd.DataFrame,
    *,
    fold: int,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    max_edges: int = 5000,
    top_k_per_node: int = 10,
    support_quantile: float = 0.90,
) -> pd.DataFrame:
    split_cfg = {"seed": seed, "n_folds": 3, "type": "learner_temporal"}
    splits = None
    for f, _s, parts in learner_based_folds(interactions, ratios, split_cfg):
        if f == fold:
            splits = parts
            break
    if splits is None:
        raise RuntimeError(f"fold {fold} not produced")
    train = splits["train"].copy()
    train["split"] = "train"
    train["fold"] = fold
    q_train = build_q_matrix_from_train(train)
    e_pre = infer_prerequisites_from_train(
        train,
        q_train,
        max_edges=max_edges,
        top_k_per_node=top_k_per_node,
        support_quantile=support_quantile,
    )
    e_pre, prune_log = prune_cycles(e_pre)
    report = audit_dag(e_pre)
    print(
        f"[fold {fold}] e_pre edges={len(e_pre)} pruned={len(prune_log)} "
        f"topo_ok={report.topo_sort_passed}"
    )
    return e_pre


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "FoundationalASSIST",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    args = parser.parse_args()

    out_proc = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed"
    out_proc.mkdir(parents=True, exist_ok=True)

    canonical, extended = ingest(args.raw_dir)
    canon_path = out_proc / "foundational_assist.parquet"
    ext_path = out_proc / "foundational_assist_extended.parquet"
    canonical.to_parquet(canon_path, index=False)
    extended.to_parquet(ext_path, index=False)
    print(f"Wrote {canon_path}")
    print(f"Wrote {ext_path}")

    folds = range(3) if args.all_folds else [args.fold]
    for fold in folds:
        e_pre = build_e_pre_for_fold(canonical, fold=fold)
        fold_dir = out_proc / "foundational_assist" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        e_path = fold_dir / "e_pre_train_only.csv"
        e_pre.to_csv(e_path, index=False)
        print(f"Wrote {e_path} ({len(e_pre)} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
