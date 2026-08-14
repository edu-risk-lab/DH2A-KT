#!/usr/bin/env python3
"""Pha 2-3 driver (M5): train DH2-KT and compare against P0 baselines.

Usage:
    python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda

v2 (Table 4): pass ``--architecture v2`` (and keep session hyperedges off).
v3 (default in configs/xes3g5m.yaml): per-concept memory + next-concept query
+ train-only session co-practice hyperedges. Writes
``results/checkpoints/<dataset>_fold<N>_v3.pt`` and
``results/tables/dh2_kt_v3_vs_p0.csv``.

    python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 0 --device cuda \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0_v3.pt
"""
from __future__ import annotations

import argparse
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
from dh2a_kt.hyperedge.indexing import select_session_hyperedges_for_training
from dh2a_kt.train.tier1 import (
    FoldResult,
    resolve_training_budget,
    train_and_evaluate_fold,
    write_comparison_table,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="DH2A-KT config, e.g. configs/xes3g5m.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true", help="Train/eval all P0 folds (0..2)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-users", type=int, default=None, help="Debug/smoke cap on train+eval users")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Comparison CSV (default: results/tables/dh2_kt_vs_p0.csv; v3 -> dh2_kt_v3_vs_p0.csv)",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        default=None,
        help="Write fold checkpoint (.pt). Default: results/checkpoints/<dataset>_fold<N>_<arch>.pt",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help="Override training.architecture (v2 or v3).",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Disable session co-practice hyperedges (v3 ablation: prereq-only).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=train_cfg.get("max_seq_len"),
    )
    comparison_models = train_cfg.get("comparison_models", ["gkt", "simplekt"])
    hidden_dim = int(train_cfg.get("hidden_dim", 128))
    n_hypergraph_layers = int(train_cfg.get("n_hypergraph_layers", 2))
    graph_dropout = float(train_cfg.get("graph_dropout", 0.0))
    graph_sensitivity_weight = float(train_cfg.get("graph_sensitivity_weight", 0.0))
    graph_sensitivity_margin = float(train_cfg.get("graph_sensitivity_margin", 0.05))
    graph_sensitivity_p = float(train_cfg.get("graph_sensitivity_p", 0.9))
    architecture = str(args.architecture or train_cfg.get("architecture", "v2"))
    diffusion_alpha = float(train_cfg.get("diffusion_alpha", 0.5))
    dropout = float(train_cfg.get("dropout", 0.2))
    session_cfg = dh2_cfg.get("hyperedge", {}).get("session", {})
    session_enabled = (
        bool(session_cfg.get("enabled", False))
        and architecture == "v3"
        and not args.no_session
    )
    output = args.output or (
        REPO_ROOT / "results" / "tables" / (
            "dh2_kt_v3_vs_p0.csv" if architecture == "v3" else "dh2_kt_vs_p0.csv"
        )
    )

    print(f"dataset={dh2_cfg['dataset']} architecture={architecture} "
          f"budget={budget.reference_model} "
          f"batch={budget.batch_size} epochs={budget.epochs} max_seq_len={budget.max_seq_len}")
    if budget.note:
        print(f"NOTE: {budget.note}")

    interactions = load_interactions_with_ids(p0_cfg)
    folds = list(range(int(p0_cfg.get("split", {}).get("n_folds", 3)))) if args.all_folds else [args.fold]

    results: list[FoldResult] = []
    for fold in folds:
        print(f"\n[fold {fold}] loading splits + e_pre...")
        splits = get_fold_splits(interactions, p0_cfg, fold)
        eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
        e_pre = load_e_pre(p0_cfg, fold)
        hyperedge_spec = concept_prerequisite_spec_from_config(dh2_cfg, fold=fold)
        session_hyperedges = None
        if session_enabled:
            raw_sessions = build_session_hyperedges(
                splits["train"],
                fold=fold,
                session_gap_seconds=float(session_cfg.get("session_gap_seconds", 1800)),
                train_only=True,
            )
            session_hyperedges = select_session_hyperedges_for_training(
                raw_sessions,
                max_hyperedges=int(session_cfg.get("max_hyperedges", 20_000)),
            )
            print(
                f"[fold {fold}] session hyperedges: built={len(raw_sessions)} "
                f"selected={len(session_hyperedges)}"
            )
        fold_result = train_and_evaluate_fold(
            splits["train"],
            eval_df,
            e_pre,
            budget,
            device=args.device,
            hidden_dim=hidden_dim,
            n_hypergraph_layers=n_hypergraph_layers,
            dropout=dropout,
            graph_dropout=graph_dropout,
            graph_sensitivity_weight=graph_sensitivity_weight,
            graph_sensitivity_margin=graph_sensitivity_margin,
            graph_sensitivity_p=graph_sensitivity_p,
            hyperedge_spec=hyperedge_spec,
            session_hyperedges=session_hyperedges,
            architecture=architecture,
            diffusion_alpha=diffusion_alpha,
            max_users=args.max_users,
            checkpoint_path=args.save_checkpoint
            or (
                REPO_ROOT
                / "results"
                / "checkpoints"
                / f"{dh2_cfg['dataset']}_fold{fold}_{architecture}.pt"
            ),
        )
        fold_result.fold = fold
        results.append(fold_result)
        print(f"[fold {fold}] AUC={fold_result.auc:.4f} n_predictions={fold_result.n_predictions}")

    table = write_comparison_table(
        results,
        dataset=dh2_cfg["dataset"],
        budget=budget,
        comparison_models=comparison_models,
        output_path=output,
    )
    print(f"\nWrote comparison table: {output}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
