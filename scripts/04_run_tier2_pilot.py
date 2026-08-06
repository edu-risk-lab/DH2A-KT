#!/usr/bin/env python3
"""Pha 4-5 driver (M8): Tier 2 agent pilot over Tier-1 predictions.

Usage:
    # Smoke (stub LLM, small train cap):
    python scripts/04_run_tier2_pilot.py configs/xes3g5m.yaml --llm-backend stub \\
        --max-train-users 64 --sample-size 20 --device cpu

    # Full pilot after saving a checkpoint from training:
    python scripts/04_run_tier2_pilot.py configs/xes3g5m.yaml --llm-backend ollama \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --sample-size 500 --device cuda

    # Train fold 0, save checkpoint, then run pilot (long — prefer load-checkpoint):
    python scripts/04_run_tier2_pilot.py configs/xes3g5m.yaml --fold 0 --device cuda \\
        --save-checkpoint results/checkpoints/xes3g5m_fold0.pt --sample-size 500
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.agents.llm_backends import resolve_llm_client
from dh2a_kt.config_helpers import concept_prerequisite_spec_from_config
from dh2a_kt.hyperedge.p0_inputs import (
    get_fold_splits,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)
from dh2a_kt.tier2.pilot import PilotSummary, run_tier2_pilot, sample_tier1_outputs, write_pilot_outputs
from dh2a_kt.train.checkpoint import load_trained_fold, save_trained_fold
from dh2a_kt.train.tier1 import resolve_training_budget, train_fold

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--llm-backend", choices=["stub", "ollama"], default="stub")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-train-users", type=int, default=None, help="Cap train users (smoke/debug)")
    parser.add_argument(
        "--pilot-epochs",
        type=int,
        default=None,
        help="Override training epochs for pilot (default: config budget; 1 when --max-train-users set)",
    )
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=None,
        help="Cap eval users for prediction sampling (defaults to max-train-users when set)",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        default=None,
        help="Save trained fold weights after training",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Skip training and load a saved fold checkpoint",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Train (or load) and save checkpoint only; skip Tier-2 agent pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tables",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for output files, e.g. 'ollama' -> *_tier2_pilot_ollama.jsonl",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
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
    graph_sensitivity_margin = float(train_cfg.get("graph_sensitivity_margin", 0.08))
    graph_sensitivity_p = float(train_cfg.get("graph_sensitivity_p", 0.9))
    if args.pilot_epochs is not None:
        budget = replace(budget, epochs=args.pilot_epochs)
    elif args.max_train_users is not None:
        budget = replace(budget, epochs=1)
        graph_sensitivity_weight = 0.0

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    max_eval = args.max_eval_users
    if max_eval is None and args.max_train_users is not None:
        max_eval = args.max_train_users
    if max_eval is not None:
        eval_users = eval_df["user_id"].unique()[:max_eval]
        pilot_eval_df = eval_df[eval_df["user_id"].isin(eval_users)]
    else:
        pilot_eval_df = eval_df

    if args.load_checkpoint is not None:
        logger.info("Loading checkpoint: %s", args.load_checkpoint)
        trained = load_trained_fold(args.load_checkpoint, device=args.device)
    else:
        e_pre = load_e_pre(p0_cfg, args.fold)
        hyperedge_spec = concept_prerequisite_spec_from_config(dh2_cfg, fold=args.fold)
        logger.info(
            "Training fold %d for Tier-2 pilot (max_train_users=%s)...",
            args.fold,
            args.max_train_users,
        )
        trained = train_fold(
            splits["train"],
            eval_df,
            e_pre,
            budget,
            device=args.device,
            hidden_dim=hidden_dim,
            n_hypergraph_layers=n_hypergraph_layers,
            graph_dropout=graph_dropout,
            graph_sensitivity_weight=graph_sensitivity_weight,
            graph_sensitivity_margin=graph_sensitivity_margin,
            graph_sensitivity_p=graph_sensitivity_p,
            hyperedge_spec=hyperedge_spec,
            max_users=args.max_train_users,
        )
        if args.save_checkpoint is not None:
            save_trained_fold(args.save_checkpoint, trained)
            logger.info("Saved checkpoint: %s", args.save_checkpoint)

    if args.checkpoint_only:
        if args.save_checkpoint is None and args.load_checkpoint is None:
            raise SystemExit("--checkpoint-only requires --save-checkpoint (or --load-checkpoint to verify load)")
        print(f"\nCheckpoint ready: {args.save_checkpoint or args.load_checkpoint}")
        return 0

    llm = resolve_llm_client(
        args.llm_backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )
    samples = sample_tier1_outputs(
        trained,
        pilot_eval_df,
        sample_size=args.sample_size,
        max_seq_len=budget.max_seq_len,
        seed=args.seed,
    )
    logger.info("Running Tier-2 pilot on %d samples (backend=%s)...", len(samples), args.llm_backend)
    records, hyperedges = run_tier2_pilot(samples, llm, fold=args.fold)

    n_flagged = sum(1 for r in records if r.critic_flagged)
    summary = PilotSummary(
        n_samples=len(records),
        n_flagged=n_flagged,
        flag_rate=n_flagged / len(records) if records else 0.0,
        n_session_hyperedges=len(hyperedges),
        llm_backend=args.llm_backend,
        fold=args.fold,
        dataset=dataset,
    )

    out_dir = Path(args.output_dir)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    stem = f"{dataset}_fold{args.fold}_tier2_pilot{tag}"
    records_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"
    hyperedges_path = out_dir / f"{stem}_session_hyperedges.json"
    write_pilot_outputs(
        records,
        summary,
        records_path=records_path,
        summary_path=summary_path,
        hyperedges_path=hyperedges_path,
        hyperedges=hyperedges,
    )

    print(f"\nTier-2 pilot complete: {len(records)} samples, flag_rate={summary.flag_rate:.3f}")
    print(f"  records:    {records_path}")
    print(f"  summary:    {summary_path}")
    print(f"  hyperedges: {hyperedges_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
