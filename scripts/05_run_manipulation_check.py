#!/usr/bin/env python3
"""M6 driver: manipulation-check gating for DH2-KT (P0 DDR-style).

Usage:
    python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 0
    python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 0 --device cuda

Writes ``results/tables/<dataset>_fold<N>_manipulation_check.json``.
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
from dh2a_kt.eval.tier1_eval import run_manipulation_check_for_fold, write_manipulation_check_result
from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_e_pre, load_interactions_with_ids
from dh2a_kt.train.tier1 import resolve_training_budget

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--graph-sensitivity-weight",
        type=float,
        default=None,
        help="Override training.graph_sensitivity_weight (e.g. 0 = w/o L_aux ablation)",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Score a saved DH2-KT fold instead of retraining (required for v3 M6 after 03).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: results/tables/<dataset>_fold<N>_manipulation_check.json",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    mc_cfg = dh2_cfg.get("manipulation_check", {})
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=train_cfg.get("max_seq_len"),
    )
    hidden_dim = int(train_cfg.get("hidden_dim", 128))
    n_hypergraph_layers = int(train_cfg.get("n_hypergraph_layers", 2))
    graph_dropout = float(train_cfg.get("graph_dropout", 0.0))
    graph_sensitivity_weight = float(train_cfg.get("graph_sensitivity_weight", 0.0))
    if args.graph_sensitivity_weight is not None:
        graph_sensitivity_weight = float(args.graph_sensitivity_weight)
    graph_sensitivity_margin = float(train_cfg.get("graph_sensitivity_margin", 0.05))
    graph_sensitivity_p = float(train_cfg.get("graph_sensitivity_p", 0.9))

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    e_pre = load_e_pre(p0_cfg, args.fold)
    hyperedge_spec = concept_prerequisite_spec_from_config(dh2_cfg, fold=args.fold)

    trained = None
    if args.load_checkpoint is not None:
        from dh2a_kt.train.checkpoint import load_trained_fold
        from dh2a_kt.train.greykt import make_sequence_loader

        trained = load_trained_fold(args.load_checkpoint, device=args.device)
        eval_df_mc = eval_df
        if args.max_users is not None:
            users = eval_df_mc["user_id"].unique()[: args.max_users]
            eval_df_mc = eval_df_mc[eval_df_mc["user_id"].isin(users)]
        trained.eval_loader = make_sequence_loader(eval_df_mc, trained, budget, shuffle=False)
        print(f"[fold {args.fold}] loaded checkpoint {args.load_checkpoint} "
              f"architecture={getattr(trained.model.config, 'architecture', 'v2')}")

    print(
        f"[fold {args.fold}] manipulation_check p={mc_cfg.get('p', 0.90)} "
        f"operator={mc_cfg.get('operator', 'node_drop')} "
        f"hyperedge_source={hyperedge_spec.source} "
        f"graph_sensitivity_weight={graph_sensitivity_weight}"
    )
    result = run_manipulation_check_for_fold(
        splits["train"],
        eval_df,
        e_pre,
        fold=args.fold,
        budget=budget,
        device=args.device,
        hidden_dim=hidden_dim,
        n_hypergraph_layers=n_hypergraph_layers,
        graph_dropout=graph_dropout,
        graph_sensitivity_weight=graph_sensitivity_weight,
        graph_sensitivity_margin=graph_sensitivity_margin,
        graph_sensitivity_p=graph_sensitivity_p,
        hyperedge_spec=hyperedge_spec,
        architecture=str(train_cfg.get("architecture", "v2")),
        diffusion_alpha=float(train_cfg.get("diffusion_alpha", 0.5)),
        max_users=args.max_users,
        p=float(mc_cfg.get("p", 0.90)),
        operator=str(mc_cfg.get("operator", "node_drop")),
        seed=int(mc_cfg.get("seed", 42)),
        trained=trained,
    )

    dataset = dh2_cfg["dataset"]
    arch = (
        str(getattr(trained.model.config, "architecture", "v2"))
        if trained is not None
        else str(train_cfg.get("architecture", "v2"))
    )
    output = args.output or (
        REPO_ROOT / "results" / "tables" / (
            f"{dataset}_fold{args.fold}_manipulation_check.json"
            if arch == "v2"
            else f"{dataset}_fold{args.fold}_manipulation_check_{arch}.json"
        )
    )
    write_manipulation_check_result(
        result,
        dataset=dataset,
        fold=args.fold,
        output_path=output,
        extra={
            "p": float(mc_cfg.get("p", 0.90)),
            "operator": mc_cfg.get("operator", "node_drop"),
            "seed": int(mc_cfg.get("seed", 42)),
        },
    )

    print(f"passes_manipulation_check={result.passes_manipulation_check}")
    print(f"auc_clean={result.auc_clean:.4f} auc_destroyed={result.auc_destroyed:.4f} drop={result.auc_drop:.4f}")
    print(f"ddr={result.ddr:.4f}")
    print(f"verdict: {result.verdict}")
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
