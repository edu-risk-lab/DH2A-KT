#!/usr/bin/env python3
"""Pha 2-3 driver (M5): train DH2-KT and compare against P0 baselines.

Usage:
    python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda

v2 (Table 4): pass ``--architecture v2`` (and keep session hyperedges off).
v3: per-concept memory + next-concept query (recorded negative result).
v4 (AUC track): LSTM backbone, DKT-style response alignment, residual concept
encoding and chunked windows. Add ``--use-questions`` for the item-aware
variant. Outputs are tagged, e.g. ``dh2_kt_v4_vs_p0.csv`` /
``dh2_kt_v4q_vs_p0.csv`` and ``<dataset>_fold<N>_<tag>.pt``.

    python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda \\
        --architecture v4 --batch-size 64 --epochs 20 --early-stop-patience 3

    python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 0 --device cuda \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0_v3.pt
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
        help="Override training.architecture (v2, v3, v4 or v5).",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Disable session co-practice hyperedges (v3/v4/v5 ablation: prereq-only).",
    )
    parser.add_argument(
        "--hyperedge-source",
        choices=("chain", "pairwise", "neighborhood"),
        default=None,
        help="Override hyperedge.concept_prerequisite.source. 'chain' enumerates "
        "every length-3..8 walk (446k hyperedges on XES3G5M); 'neighborhood' keeps "
        "one hyperedge per concept plus its direct prerequisites.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Train with zero hyperedges (graph-contribution ablation).",
    )
    parser.add_argument(
        "--use-questions",
        action="store_true",
        help="v4/v5: add Rasch-style item difficulty (compare against simplekt/akt/gikt).",
    )
    parser.add_argument(
        "--memory-dim",
        type=int,
        default=None,
        help="v5 only: width of the per-concept mastery memory (default 16).",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help="v5 only: cap on hyperedge neighbours transported per concept.",
    )
    parser.add_argument(
        "--graph-transport",
        type=float,
        default=None,
        help="v5 only: damping on evidence written to a neighbour (0 = inert write).",
    )
    parser.add_argument(
        "--max-kcs",
        type=int,
        default=None,
        help="v5 only: pad width for the KC set of each question attempt.",
    )
    parser.add_argument(
        "--priority-hypergraph",
        action="store_true",
        help="v5: enable the priority hypergraph group — attention pool over "
        "multi-KC sets, star (concept↔hyperedge↔concept) transport, learned "
        "question-hyperedge embeddings, and kind-conditioned concept projections.",
    )
    parser.add_argument(
        "--event-pool",
        choices=("mean", "attention"),
        default=None,
        help="v5: how to pool the KC set of an event (default mean; attention with --priority-hypergraph).",
    )
    parser.add_argument(
        "--transport",
        choices=("clique", "star"),
        default=None,
        help="v5: clique expands hyperedges to pairs; star keeps the hyperedge node.",
    )
    parser.add_argument(
        "--hyperedge-embed",
        action="store_true",
        help="v5: add a learned embedding of the question-as-hyperedge to each event.",
    )
    parser.add_argument(
        "--kind-conditioned",
        action="store_true",
        help="v5: project each hyperedge kind's concept states with its own Linear.",
    )
    parser.add_argument(
        "--mask-repeats",
        action="store_true",
        help="Clean protocol: drop targets that only repeat the previous row's "
        "attempt at the same question. pyKT's KC-level export splits one "
        "multi-concept question into rows sharing the answer, so 14.4%% of "
        "XES3G5M eval positions are answerable by copying the input. AUC under "
        "this flag is NOT comparable to the P0 baseline table.",
    )
    parser.add_argument(
        "--window-mode",
        choices=("first", "chunked"),
        default=None,
        help="'first' keeps one max_seq_len window per user (Table 4); "
        "'chunked' tiles the whole log, matching P0 eval coverage.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override the P0-matched batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override the P0-matched epoch count")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lstm-layers", type=int, default=None, help="v4 LSTM depth")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override pyKT/P0 max_seq_len (chunk/window length). Do not edit the YAML for sweeps.",
    )
    parser.add_argument(
        "--recap-attention",
        action="store_true",
        help="v4: causal attention over LSTM history before the bilinear readout (E2).",
    )
    parser.add_argument(
        "--question-kc-agg",
        action="store_true",
        help="v4: add mean of observed question→KC concepts to the Rasch vector (E3).",
    )
    parser.add_argument(
        "--question-graph",
        action="store_true",
        help="v4: GIKT-style question embedding refined by observed Q–KC incidence "
        "(no response transport). Hard ablation twin: omit this flag.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Fraction of TRAIN users held out to select the best epoch by AUC.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed weight init and batch order so a run is reproducible. The "
        "internal selection split stays fixed, so multi-seed runs differ only "
        "in optimisation noise.",
    )
    parser.add_argument(
        "--graph-dropout",
        type=float,
        default=None,
        help="Override training.graph_dropout (0 disables empty-graph batches).",
    )
    parser.add_argument(
        "--graph-sensitivity-weight",
        type=float,
        default=None,
        help="Override the M6 auxiliary loss weight (0 disables it).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix for default output/checkpoint names, e.g. 'v4q_tuned'.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=args.max_seq_len if args.max_seq_len is not None else train_cfg.get("max_seq_len"),
    )
    p0_batch_size, p0_epochs = budget.batch_size, budget.epochs
    if args.batch_size is not None:
        budget.batch_size = args.batch_size
    if args.epochs is not None:
        budget.epochs = args.epochs
    if args.lr is not None:
        budget.lr = args.lr
    # Only an override that actually differs from P0 breaks the matched claim:
    # passing --batch-size 4 --epochs 10 reproduces the reference budget.
    if budget.batch_size != p0_batch_size or budget.epochs != p0_epochs:
        budget.matched_p0 = False
        budget.note = (
            f"tuned budget (batch={budget.batch_size}, epochs={budget.epochs}); "
            f"P0 {budget.reference_model} used batch={p0_batch_size}, epochs={p0_epochs}"
        )
    if args.mask_repeats:
        # P0 scores every KC row of a multi-concept attempt, so a repeat-masked
        # AUC measures a different protocol rather than a better model.
        budget.matched_p0 = False
        budget.note = (
            f"{budget.note}; " if budget.note else ""
        ) + "clean protocol (repeat-row targets masked): AUC is NOT comparable to the P0 baselines"

    comparison_models = train_cfg.get("comparison_models", ["gkt", "simplekt"])
    hidden_dim = int(args.hidden_dim or train_cfg.get("hidden_dim", 128))
    n_hypergraph_layers = int(train_cfg.get("n_hypergraph_layers", 2))
    graph_dropout = float(
        args.graph_dropout if args.graph_dropout is not None
        else train_cfg.get("graph_dropout", 0.0)
    )
    graph_sensitivity_weight = float(
        args.graph_sensitivity_weight if args.graph_sensitivity_weight is not None
        else train_cfg.get("graph_sensitivity_weight", 0.0)
    )
    graph_sensitivity_margin = float(train_cfg.get("graph_sensitivity_margin", 0.05))
    graph_sensitivity_p = float(train_cfg.get("graph_sensitivity_p", 0.9))
    architecture = str(args.architecture or train_cfg.get("architecture", "v2"))
    diffusion_alpha = float(train_cfg.get("diffusion_alpha", 0.5))
    dropout = float(args.dropout if args.dropout is not None else train_cfg.get("dropout", 0.2))
    n_lstm_layers = int(args.lstm_layers or train_cfg.get("n_lstm_layers", 1))
    use_questions = bool(args.use_questions or train_cfg.get("use_questions", False))
    recap_attention = bool(args.recap_attention or train_cfg.get("recap_attention", False))
    question_kc_agg = bool(args.question_kc_agg or train_cfg.get("question_kc_agg", False))
    question_graph = bool(args.question_graph or train_cfg.get("question_graph", False))
    memory_dim = int(args.memory_dim if args.memory_dim is not None else train_cfg.get("memory_dim", 16))
    max_degree = int(args.max_degree if args.max_degree is not None else train_cfg.get("max_degree", 16))
    graph_transport = float(
        args.graph_transport if args.graph_transport is not None
        else train_cfg.get("graph_transport", 0.5)
    )
    max_kcs = int(args.max_kcs if args.max_kcs is not None else train_cfg.get("max_kcs", 6))
    priority = bool(args.priority_hypergraph)
    event_pool = str(
        args.event_pool
        or train_cfg.get("event_pool", "attention" if priority else "mean")
    )
    transport = str(
        args.transport
        or train_cfg.get("transport", "star" if priority else "clique")
    )
    use_hyperedge_embed = bool(
        args.hyperedge_embed or priority or train_cfg.get("use_hyperedge_embed", False)
    )
    kind_conditioned = bool(
        args.kind_conditioned or priority or train_cfg.get("kind_conditioned", False)
    )
    window_mode = str(
        args.window_mode
        or train_cfg.get(
            "window_mode",
            "chunked" if architecture in ("v4", "v5") else "first",
        )
    )
    val_frac = float(
        args.val_frac if args.val_frac is not None
        else train_cfg.get("val_frac", 0.1 if architecture in ("v4", "v5") else 0.0)
    )
    early_stop_patience = args.early_stop_patience or train_cfg.get("early_stop_patience")
    session_cfg = dh2_cfg.get("hyperedge", {}).get("session", {})
    session_enabled = (
        bool(session_cfg.get("enabled", False))
        and architecture in ("v3", "v4", "v5")
        and not args.no_session
        and not args.no_graph
    )
    tag = args.tag or (f"{architecture}q" if use_questions else architecture)
    output = args.output or (
        REPO_ROOT / "results" / "tables" / (
            "dh2_kt_vs_p0.csv" if architecture == "v2" and not args.tag
            else f"dh2_kt_{tag}_vs_p0.csv"
        )
    )

    print(f"dataset={dh2_cfg['dataset']} architecture={architecture} tag={tag} "
          f"hyperedge_source={args.hyperedge_source or 'from config'} no_graph={args.no_graph} "
          f"use_questions={use_questions} window_mode={window_mode} val_frac={val_frac} "
          f"mask_repeats={args.mask_repeats} seed={args.seed} "
          f"recap_attention={recap_attention} question_kc_agg={question_kc_agg} "
          f"question_graph={question_graph} "
          f"memory_dim={memory_dim} max_degree={max_degree} graph_transport={graph_transport} "
          f"event_pool={event_pool} transport={transport} "
          f"hyperedge_embed={use_hyperedge_embed} kind_conditioned={kind_conditioned} "
          f"graph_dropout={graph_dropout} laux={graph_sensitivity_weight} "
          f"budget={budget.reference_model} matched_p0={budget.matched_p0} "
          f"batch={budget.batch_size} epochs={budget.epochs} lr={budget.lr} "
          f"max_seq_len={budget.max_seq_len}")
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
        if args.hyperedge_source is not None:
            hyperedge_spec = replace(hyperedge_spec, source=args.hyperedge_source)
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
            use_questions=use_questions,
            n_lstm_layers=n_lstm_layers,
            window_mode=window_mode,
            val_frac=val_frac,
            early_stop_patience=early_stop_patience,
            use_graph=not args.no_graph,
            mask_repeats=args.mask_repeats,
            seed=args.seed,
            memory_dim=memory_dim,
            max_degree=max_degree,
            graph_transport=graph_transport,
            max_kcs=max_kcs,
            event_pool=event_pool,
            transport=transport,
            use_hyperedge_embed=use_hyperedge_embed,
            kind_conditioned=kind_conditioned,
            recap_attention=recap_attention,
            question_kc_agg=question_kc_agg,
            question_graph=question_graph,
            checkpoint_path=args.save_checkpoint
            or (
                REPO_ROOT
                / "results"
                / "checkpoints"
                / f"{dh2_cfg['dataset']}_fold{fold}_{tag}.pt"
            ),
        )
        fold_result.fold = fold
        results.append(fold_result)
        val_note = (
            f" val_auc={fold_result.val_auc:.4f}"
            if fold_result.val_auc is not None
            else ""
        )
        print(
            f"[fold {fold}] AUC={fold_result.auc:.4f}{val_note} "
            f"n_predictions={fold_result.n_predictions}"
        )

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
