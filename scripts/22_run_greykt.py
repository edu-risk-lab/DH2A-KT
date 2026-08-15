#!/usr/bin/env python3
"""GreyKT GPU driver: C_B diagnostic gate, then wrap-eval or fused train.

Run on the RTX 3090 host (torch + torch_geometric + P0 processed data).

    # Predictive C_B is SUPPORTED on XES3G5M fold 0. Wrap frozen DH2-KT first.
    python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive

    # Then optional fused train (C_B = |2p-1| is detached). Checkpoints
    # latest.pt / best.pt under results/checkpoints/greykt_* each epoch;
    # re-run with --resume after a crash.
    python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive --resume
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
    assert_black_auc_matches_native,
    collect_greykt_outputs,
    enforce_cb_diagnostic,
    fit_temperature_on_loader,
    greykt_checkpoint_dir,
    make_sequence_loader,
    make_whitebox_cached_loader,
    native_dh2kt_auc,
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


def _cap_users(df: pd.DataFrame, max_users: int | None) -> pd.DataFrame:
    if max_users is None:
        return df
    users = df["user_id"].unique()[:max_users]
    return df[df["user_id"].isin(users)]


def _eval_frames(splits: dict, max_users: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validation-only, test-only, and Table-4 comparable valid+test frames."""
    valid_df = _cap_users(splits["valid"], max_users)
    test_df = _cap_users(splits["test"], max_users)
    combined = pd.concat([valid_df, test_df], ignore_index=True)
    return valid_df, test_df, combined


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
    valid_df, test_df, combined = _eval_frames(splits, args.max_users)
    if args.load_checkpoint is not None:
        trained = load_trained_fold(args.load_checkpoint, device=args.device)
        # Default eval_loader is valid+test so wrap black_auc is Table-4 comparable.
        # Checkpoint *selection* uses a separate valid_loader (never this one).
        trained.eval_loader = make_sequence_loader(combined, trained, budget, shuffle=False)
        return trained, valid_df, test_df, combined
    trained = train_fold(splits["train"], combined, e_pre, budget, **kwargs)
    return trained, valid_df, test_df, combined


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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Dir for GreyKT latest.pt / best.pt (default: results/checkpoints/greykt_<dataset>_foldN_<variant>_<signal>).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume train from checkpoint-dir/latest.pt if present.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Explicit GreyKT train checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override GreyKT train batch size (default: matched GKT budget, 4). "
        "Larger batches are faster but are not a matched-budget comparison.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Mixed-precision (fp16) GreyKT train. Frees activation VRAM for larger batches.",
    )
    parser.add_argument(
        "--freeze-hypergraph",
        action="store_true",
        help="Encode concept graph once per epoch (no graph backward). Dual-gate "
        "still trains. Uses leftover VRAM on larger sequence batches instead of "
        "rebuilding 400k hyperedges every step.",
    )
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
    if args.batch_size is not None:
        from dataclasses import replace

        budget = replace(
            budget,
            batch_size=int(args.batch_size),
            matched_p0=False,
            note=(budget.note + " " if budget.note else "")
            + f"GreyKT train batch_size override={args.batch_size} (not matched GKT=4).",
        )
        print(f"NOTE: {budget.note}")
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

    trained, valid_df, test_df, combined_df = _load_or_train_black_box(
        args, dh2_cfg, p0_cfg, splits, e_pre, budget, hyperedge_spec
    )
    valid_loader = make_sequence_loader(valid_df, trained, budget, shuffle=False)
    test_loader = make_sequence_loader(test_df, trained, budget, shuffle=False)
    combined_loader = trained.eval_loader  # valid+test, Table 4 protocol
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
        train_loader = make_whitebox_cached_loader(
            train_df,
            trained,
            budget,
            model,
            prereq,
            shuffle=True,
        )
        ckpt_dir = args.checkpoint_dir or greykt_checkpoint_dir(
            dataset=dataset,
            fold=args.fold,
            variant=args.variant,
            signal=args.signal,
            root=REPO_ROOT / "results" / "checkpoints",
        )
        resume_from = args.resume_from
        if resume_from is None and args.resume:
            latest = Path(ckpt_dir) / "latest.pt"
            resume_from = latest if latest.exists() else None
            if resume_from is None:
                print(f"--resume set but no checkpoint at {latest}; training from scratch")
            else:
                print(f"resuming from {resume_from}")
        print(f"GreyKT checkpoints -> {ckpt_dir}")
        valid_cached = make_whitebox_cached_loader(
            valid_df,
            trained,
            budget,
            model,
            prereq,
            shuffle=False,
        )
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
            checkpoint_dir=ckpt_dir,
            resume_from=resume_from,
            use_amp=bool(args.amp),
            freeze_hypergraph=bool(args.freeze_hypergraph),
            valid_loader=valid_cached,
            checkpoint_meta={
                "dataset": dataset,
                "fold": args.fold,
                "variant": args.variant,
                "signal": args.signal,
                "batch_size": budget.batch_size,
                "matched_p0": budget.matched_p0,
                "amp": bool(args.amp),
                "freeze_hypergraph": bool(args.freeze_hypergraph),
                "selection_metric": "valid_nll",
            },
        )

    if args.variant in ("v4a", "v4ab"):
        t = fit_temperature_on_loader(model, trained, valid_loader, prereq, freq_t)
        print(f"fitted temperature T={t:.4f} (v4a, validation only)")

    trained.eval_loader = combined_loader
    native_auc, native_n = native_dh2kt_auc(trained)
    print(f"DH2KT native evaluate_auc on valid+test: {native_auc:.4f} (n={native_n})")

    split_reports = {}
    for split_name, loader in (
        ("valid", valid_loader),
        ("test", test_loader),
        ("valid+test", combined_loader),
    ):
        trained.eval_loader = loader
        split_report = collect_greykt_outputs(
            model, trained, prereq, freq_t, mc_samples=mc_samples
        )
        split_report.fold = args.fold
        split_report.variant = args.variant
        split_report.diagnostic_verdict = verdict
        split_report.eval_split = split_name
        split_reports[split_name] = split_report
        print(
            f"  {split_name}: n={split_report.n_predictions} "
            f"fused={split_report.fused_auc:.4f} black={split_report.black_auc:.4f} "
            f"white={split_report.white_auc:.4f}"
        )

    report = split_reports["valid+test"]
    report.dh2kt_native_auc = native_auc
    from dataclasses import asdict as _asdict

    report.splits = {
        name: {k: v for k, v in _asdict(r).items() if k != "splits"}
        for name, r in split_reports.items()
    }
    try:
        assert_black_auc_matches_native(report.black_auc, native_auc)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"wrap audit OK: |black_auc - native| = "
        f"{abs(report.black_auc - native_auc):.4f} (tol=0.01)"
    )
    out = args.output or (
        REPO_ROOT
        / "results"
        / "tables"
        / f"{dataset}_fold{args.fold}_greykt_{args.mode}_{args.variant}_{args.signal}.json"
    )
    write_greykt_report(report, out)
    print(
        f"\nGreyKT {args.mode}/{args.variant} valid+test: n={report.n_predictions} "
        f"fused_auc={report.fused_auc:.4f} black_auc={report.black_auc:.4f} "
        f"white_auc={report.white_auc:.4f} native_auc={native_auc:.4f} "
        f"mean_gate={report.mean_gate:.3f}"
    )
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
