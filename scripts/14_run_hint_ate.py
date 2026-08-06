#!/usr/bin/env python3
"""Estimate IPW ATE of hint use on next-step correctness (FoundationalASSIST).

Treatment = hint_used at step t; outcome = correct at t+1 (never same-step
discrete_score, which is mechanically tied to hint requests).

Usage:
    python scripts/14_run_hint_ate.py configs/foundational_assist.yaml --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids
from dh2a_kt.models.causal_integration import build_hint_causal_dataset, estimate_hint_ate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tables",
    )
    args = parser.parse_args()

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    ext_rel = dh2_cfg.get("extended_processed_path")
    if not ext_rel:
        raise SystemExit("config needs extended_processed_path with hint columns")
    ext_path = REPO_ROOT / ext_rel
    if not ext_path.exists():
        raise SystemExit(f"Missing {ext_path}; run scripts/12_ingest_foundational_assist.py")

    canonical = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(canonical, p0_cfg, args.fold)
    train_users = set(splits["train"]["user_id"].astype("int64").tolist())

    extended = pd.read_parquet(ext_path)
    train_ext = extended[extended["user_id"].isin(train_users)].copy()
    print(
        f"[fold {args.fold}] dataset={dataset} train_rows={len(train_ext)} "
        f"hint_rate={train_ext['hint_used'].mean():.4f}"
    )

    causal = build_hint_causal_dataset(
        train_ext,
        max_users=args.max_users,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    ate, diagnostics = estimate_hint_ate(causal)

    payload = {
        "dataset": dataset,
        "fold": args.fold,
        "ate_ipw": ate,
        **diagnostics,
        "max_users": args.max_users,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "confounders": [
            "prior_accuracy",
            "log1p_n_prior",
            "same_kc_prior_accuracy",
            "normalized_position",
            "log1p_item_id",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{dataset}_fold{args.fold}_hint_ate.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ATE_IPW={ate:.4f} naive_diff={diagnostics['naive_mean_diff']:.4f}")
    print(
        f"propensity=[{diagnostics['propensity_min']:.3f}, {diagnostics['propensity_max']:.3f}] "
        f"n_treated={diagnostics['n_treated']} n_control={diagnostics['n_control']}"
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
