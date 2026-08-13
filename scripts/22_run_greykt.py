#!/usr/bin/env python3
"""GreyKT GPU driver: C_B diagnostic gate, then wrap-eval or fused train.

Run on the RTX 3090 host (torch + torch_geometric + P0 processed data).

    # Predictive C_B is SUPPORTED on XES3G5M fold 0. Wrap frozen DH2-KT first.
    python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive

    # Then optional fused train (C_B = |2p-1| is detached).
    python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.config_helpers import concept_prerequisite_spec_from_config
from dh2a_kt.diagnostics.black_confidence import diagnostic_json_name
from dh2a_kt.hyperedge.p0_inputs import (
    get_fold_splits,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)
from dh2a_kt.train.checkpoint import load_trained_fold
from dh2a_kt.train.greykt import (
    collect_greykt_outputs,
    enforce_cb_diagnostic,
    fit_temperature_on_loader,
    make_sequence_loader,
    train_greykt_one_fold,
    wrap_trained_fold,
    write_greykt_report,
)
from dh2a_kt.train.greykt_inputs import (
    concept_training_frequency,
    prereq_edge_index_from_e_pre,
    prior_mean_from_train,
)
from dh2a_kt.train.tier1 import resolve_training_budget, train_fold

logger = logging.getLogger(__name__)


def _diagnostic_path(dataset: str, fold: int, signal: str) -> Path:
    return REPO_ROOT / "results" / "tables" / diagnostic_json_name(dataset, fold, signal)


def _enforce_diagnostic(path: Path, *, force: bool) -> str | None:
    try:
        verdict = enforce_cb_diagnostic(path, force=force)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if verdict is not None:
        print(f"diagnostic verdict={verdict} ({path})")
    return verdict


def _load_or_train_black_box(args, dh2_cfg, p0_cfg, splits, e_pre, budget, hyperedge_spec):
    train_cfg = dh2_cfg.get("training", {})
    kwargs = dict(
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
    if args.load_checkpoint is not None:
        trained = load_trained_fold(args.load_checkpoint, device=args.device)
        eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
        if args.max_users is not None:
            users = eval_df["user_id"].unique()[: args.max_users]
            eval_df = eval_df[eval_df["user_id"].isin(users)]
        trained.eval_loader = make_sequence_loader(eval_df, trained, budget, shuffle=False)
        return trained
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    return train_fold(splits["train"], eval_df, e_pre, budget, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["diagnose", "wrap", "train"], required=True)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--variant",
        choices=["v3", "v4a", "v4b", "v4ab"],
        default="v3",
        help="v3 = relative gate only; v4a = temperature; v4b = absolute backoff",
    )
    parser.add_argument("--force", action="store_true", help="Ignore NOT_SUPPORTED diagnostic")
    parser.add_argument(
        "--signal",
        choices=["frequency", "mc_dropout", "predictive"],
        default="predictive",
        help="Black-box C_B. Default predictive: frequency and mc_dropout "
        "already failed on XES3G5M fold 0.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="MC-dropout samples when --signal mc_dropout (default greykt.mc_samples or 8).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == "diagnose":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "21_diagnose_black_confidence.py"),
            str(args.config),
            "--fold",
            str(args.fold),
            "--device",
            args.device,
        ]
        if args.max_users is not None:
            cmd += ["--max-users", str(args.max_users)]
        if args.load_checkpoint is not None:
            cmd += ["--load-checkpoint", str(args.load_checkpoint)]
        cmd += ["--signal", args.signal]
        if args.mc_samples is not None:
            cmd += ["--mc-samples", str(args.mc_samples)]
        return subprocess.call(cmd)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    verdict = _enforce_diagnostic(
        _diagnostic_path(dataset, args.fold, args.signal), force=args.force
    )

    train_cfg = dh2_cfg.get("training", {})
    greykt_cfg = dh2_cfg.get("greykt", {})
    mc_samples = (
        0
        if args.signal != "mc_dropout"
        else (
            args.mc_samples
            if args.mc_samples is not None
            else int(greykt_cfg.get("mc_samples", 8))
        )
    )
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=train_cfg.get("max_seq_len"),
    )
    hyperedge_spec = concept_prerequisite_spec_from_config(dh2_cfg, fold=args.fold)
    max_hops = int(greykt_cfg.get("max_hops", hyperedge_spec.max_chain_len))
    use_backoff = args.variant in ("v4b", "v4ab")
    temperature = 1.0  # v4a fitted after wrap if requested

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    train_df = splits["train"]
    if args.max_users is not None:
        users = train_df["user_id"].unique()[: args.max_users]
        train_df = train_df[train_df["user_id"].isin(users)]
    e_pre = load_e_pre(p0_cfg, args.fold)

    trained = _load_or_train_black_box(
        args, dh2_cfg, p0_cfg, splits, e_pre, budget, hyperedge_spec
    )
    prior_mean = prior_mean_from_train(train_df)
    freq = concept_training_frequency(train_df, trained.kc_to_idx)
    prereq = prereq_edge_index_from_e_pre(e_pre, trained.kc_to_idx)
    freq_t = torch.tensor(freq, dtype=torch.float32)

    model = wrap_trained_fold(
        trained,
        prior_mean=prior_mean,
        max_hops=max_hops,
        greykt_cfg=greykt_cfg,
        temperature=temperature,
        use_absolute_backoff=use_backoff,
        black_confidence_mode="predictive" if args.signal == "predictive" else "frequency",
    )

    if args.mode == "train":
        train_loader = make_sequence_loader(train_df, trained, budget, shuffle=True)
        train_greykt_one_fold(
            model,
            trained,
            train_loader,
            prereq,
            freq_t,
            budget,
            graph_dropout=float(train_cfg.get("graph_dropout", 0.0)),
            graph_sensitivity_weight=float(train_cfg.get("graph_sensitivity_weight", 0.0)),
            graph_sensitivity_margin=float(train_cfg.get("graph_sensitivity_margin", 0.05)),
            graph_sensitivity_p=float(train_cfg.get("graph_sensitivity_p", 0.9)),
            mc_samples=mc_samples,
        )

    if args.variant in ("v4a", "v4ab"):
        valid_df = splits["valid"]
        if args.max_users is not None:
            users = valid_df["user_id"].unique()[: args.max_users]
            valid_df = valid_df[valid_df["user_id"].isin(users)]
        valid_loader = make_sequence_loader(valid_df, trained, budget, shuffle=False)
        t = fit_temperature_on_loader(model, trained, valid_loader, prereq, freq_t)
        print(f"fitted temperature T={t:.4f} (v4a, validation only)")

    report = collect_greykt_outputs(model, trained, prereq, freq_t, mc_samples=mc_samples)
    report.fold = args.fold
    report.variant = args.variant
    report.diagnostic_verdict = verdict
    out = args.output or (
        REPO_ROOT
        / "results"
        / "tables"
        / f"{dataset}_fold{args.fold}_greykt_{args.mode}_{args.variant}_{args.signal}.json"
    )
    write_greykt_report(report, out)
    print(
        f"\nGreyKT {args.mode}/{args.variant}: n={report.n_predictions} "
        f"fused_auc={report.fused_auc:.4f} black_auc={report.black_auc:.4f} "
        f"white_auc={report.white_auc:.4f} mean_gate={report.mean_gate:.3f}"
    )
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
