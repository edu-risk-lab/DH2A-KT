#!/usr/bin/env python3
"""Ablation: concept-prereq only vs +session(Hint) hyperedges on FoundationalASSIST.

Usage:
    python scripts/16_train_session_ablation.py configs/foundational_assist.yaml --fold 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.config_helpers import concept_prerequisite_spec_from_config
from dh2a_kt.hyperedge.construction import build_session_hyperedges
from dh2a_kt.hyperedge.p0_inputs import (
    get_fold_splits,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)
from dh2a_kt.train.tier1 import resolve_training_budget, train_and_evaluate_fold

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--skip-concept-only",
        action="store_true",
        help="Reuse --baseline-auc instead of retraining concept-only",
    )
    parser.add_argument(
        "--baseline-auc",
        type=float,
        default=0.7196426842152751,
        help="Concept-only AUC when --skip-concept-only (default: fold0 result)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "foundational_assist_session_ablation.json",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "local_default"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=train_cfg.get("max_seq_len"),
    )
    fold = args.fold
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, fold)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    e_pre = load_e_pre(p0_cfg, fold)
    hyperedge_spec = concept_prerequisite_spec_from_config(dh2_cfg, fold=fold)

    common = dict(
        budget=budget,
        device=args.device,
        hidden_dim=int(train_cfg.get("hidden_dim", 128)),
        n_hypergraph_layers=int(train_cfg.get("n_hypergraph_layers", 2)),
        graph_dropout=float(train_cfg.get("graph_dropout", 0.0)),
        graph_sensitivity_weight=float(train_cfg.get("graph_sensitivity_weight", 0.0)),
        graph_sensitivity_margin=float(train_cfg.get("graph_sensitivity_margin", 0.05)),
        graph_sensitivity_p=float(train_cfg.get("graph_sensitivity_p", 0.9)),
        hyperedge_spec=hyperedge_spec,
        max_users=args.max_users,
    )

    if args.skip_concept_only:
        print(f"=== Run A: concept-prerequisite only (cached AUC={args.baseline_auc:.4f}) ===")
        base_auc = float(args.baseline_auc)
        base_n = None
    else:
        print("=== Run A: concept-prerequisite only ===")
        base = train_and_evaluate_fold(splits["train"], eval_df, e_pre, **common)
        base.fold = fold
        base_auc = base.auc
        base_n = base.n_predictions
        print(f"[concept-only] AUC={base_auc:.4f} n={base_n}")

    print("=== Run B: concept-prerequisite + session (Hint-aware) ===")
    ext_path = REPO_ROOT / dh2_cfg["extended_processed_path"]
    extended = pd.read_parquet(ext_path)
    train_users = set(splits["train"]["user_id"].astype("int64"))
    train_ext = extended[extended["user_id"].isin(train_users)]
    gap = float(dh2_cfg.get("hyperedge", {}).get("session", {}).get("session_gap_seconds", 1800))
    session_hes = build_session_hyperedges(train_ext, fold=fold, session_gap_seconds=gap)
    print(f"Built {len(session_hes)} raw session hyperedges")

    plus = train_and_evaluate_fold(
        splits["train"],
        eval_df,
        e_pre,
        session_hyperedges=session_hes,
        **common,
    )
    plus.fold = fold
    print(f"[concept+session] AUC={plus.auc:.4f} n={plus.n_predictions}")

    payload = {
        "dataset": dh2_cfg["dataset"],
        "fold": fold,
        "concept_only_auc": base_auc,
        "concept_plus_session_auc": plus.auc,
        "delta_auc": plus.auc - base_auc,
        "n_predictions": plus.n_predictions,
        "concept_only_n_predictions": base_n,
        "concept_only_cached": bool(args.skip_concept_only),
        "n_raw_session_hyperedges": len(session_hes),
        "note": (
            "Session HEs projected to concept co-occurrence nodes; "
            "capped/selected via select_session_hyperedges_for_training (prefer hint)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nΔAUC={payload['delta_auc']:+.4f}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
