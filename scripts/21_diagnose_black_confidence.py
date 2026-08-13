#!/usr/bin/env python3
"""Precondition test for GreyKT's reliability gate: does C_B predict black-box error?

GreyKT fuses its two branches with g_B = C_B / (C_B + C_W + eps), where
C_B(c) = N_c^train / (N_c^train + kappa_B) stands in for "how much can the
black-box branch be trusted on concept c". That substitution is an
assumption, not a measurement. This script measures it.

Run this BEFORE committing to a full GreyKT training run. If the verdict
is NOT_SUPPORTED, the gate is steering on a signal that carries no
information about where the black box actually errs, and the fix is a
different confidence signal (MC-dropout, ensemble variance) -- not more
architecture on top of a broken premise.

Deliberately evaluates on the VALIDATION split only (not test): this
diagnostic informs design decisions, so letting it see test data would
contaminate the final evaluation. Pass --include-test only for a final
confirmatory run after the design is frozen.

Usage:
    python scripts/21_diagnose_black_confidence.py configs/xes3g5m.yaml --fold 0
    python scripts/21_diagnose_black_confidence.py configs/xes3g5m.yaml --fold 0 --device cuda

Writes ``results/tables/<dataset>_fold<N>_black_confidence_diagnostic.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.config_helpers import concept_prerequisite_spec_from_config
from dh2a_kt.diagnostics.black_confidence import (
    diagnostic_json_name,
    format_report,
    stratify_by_black_confidence,
    stratify_confidence_vs_error,
    confidence_from_mc_std,
    confidence_from_predictive_prob,
)
from dh2a_kt.hyperedge.p0_inputs import (
    get_fold_splits,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)
from dh2a_kt.train.checkpoint import load_trained_fold
from dh2a_kt.train.greykt import make_sequence_loader
from dh2a_kt.train.greykt_inputs import concept_training_frequency
from dh2a_kt.train.tier1 import resolve_training_budget, train_fold

logger = logging.getLogger(__name__)


def collect_predictions_with_concepts(trained) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black-box probabilities, labels, and the TARGET concept id per prediction.

    Mirrors ``dh2a_kt.train.tier1.collect_predictions`` exactly, adding the
    concept id. Alignment matters: the model predicts position ``t+1`` from
    positions ``<= t``, so predictions are ``logits[:, :-1]`` against labels
    ``correct[:, 1:]``, and the concept being answered is likewise
    ``concept_ids[:, 1:]`` -- NOT ``concept_ids[:, :-1]``. Getting this
    off by one would silently attribute each prediction to the previous
    exercise's concept and quietly invalidate the whole diagnostic.
    """
    import torch

    from dh2a_kt.models.dh2_kt import DH2KTBatch
    from dh2a_kt.train.tier1 import precompute_concept_states

    model = trained.model
    device = trained.device
    hyperedge_index = trained.clean_hyperedge_index

    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    concepts: list[int] = []

    with torch.no_grad():
        concept_states = precompute_concept_states(model, hyperedge_index, device)
        for batch in trained.eval_loader:
            lengths = batch["lengths"].to(device)
            dh2_batch = DH2KTBatch(
                concept_ids=batch["concept_ids"].to(device),
                exercise_ids=batch["exercise_ids"].to(device),
                responses=batch["responses"].to(device),
                hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
                concept_states=concept_states,
            )
            logits = model(dh2_batch)
            pred = logits[:, :-1, 0]
            target = batch["correct"].to(device)[:, 1:]
            target_concepts = batch["concept_ids"].to(device)[:, 1:]
            mask = torch.arange(pred.size(1), device=device).unsqueeze(0) < (
                lengths.unsqueeze(1) - 1
            )
            probs.extend(torch.sigmoid(pred[mask]).cpu().tolist())
            labels.extend(target[mask].cpu().tolist())
            concepts.extend(target_concepts[mask].cpu().tolist())

    return (
        np.asarray(probs, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(concepts, dtype=np.int64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--kappa-b",
        type=float,
        default=4.0,
        help="Pseudo-count in C_B; must match GreyKTConfig.kappa_b (default 4.0)",
    )
    parser.add_argument("--n-buckets", type=int, default=5)
    parser.add_argument(
        "--strategy",
        default="quantile",
        choices=["quantile", "equal_width"],
        help="quantile (default) keeps buckets balanced; C_B saturates toward 1, "
        "so equal_width collapses most predictions into the top bucket",
    )
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.05,
        help="Minimum |Spearman rho| to call the premise SUPPORTED. At 10^5+ "
        "predictions, p-values alone certify negligible effects (default 0.05)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also evaluate on the test split. Off by default: this diagnostic "
        "drives design decisions, so test data must stay held out until the "
        "design is frozen.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Reuse a trained DH2-KT fold instead of retraining (preferred on GPU).",
    )
    parser.add_argument(
        "--signal",
        choices=["frequency", "mc_dropout", "predictive"],
        default="frequency",
        help="Black-box confidence to test. frequency and mc_dropout are "
        "NOT_SUPPORTED on XES3G5M fold 0; predictive is |2p_B-1| (one eval pass).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Stochastic graph encodes for --signal mc_dropout (default: greykt.mc_samples or 8).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    greykt_cfg = dh2_cfg.get("greykt", {})
    if args.kappa_b == 4.0 and "kappa_b" in greykt_cfg:
        args.kappa_b = float(greykt_cfg["kappa_b"])
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=train_cfg.get("max_seq_len"),
    )

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    eval_df = (
        pd.concat([splits["valid"], splits["test"]], ignore_index=True)
        if args.include_test
        else splits["valid"]
    )
    e_pre = load_e_pre(p0_cfg, args.fold)
    hyperedge_spec = concept_prerequisite_spec_from_config(dh2_cfg, fold=args.fold)

    split_name = "valid+test" if args.include_test else "valid"
    mc_samples = (
        args.mc_samples
        if args.mc_samples is not None
        else int(greykt_cfg.get("mc_samples", 8))
    )
    print(
        f"[fold {args.fold}] C_B premise diagnostic on '{split_name}' split "
        f"signal={args.signal} (kappa_B={args.kappa_b}, {args.n_buckets} {args.strategy} buckets)"
    )

    if args.load_checkpoint is not None:
        trained = load_trained_fold(args.load_checkpoint, device=args.device)
        eval_for_loader = eval_df
        if args.max_users is not None:
            users = eval_for_loader["user_id"].unique()[: args.max_users]
            eval_for_loader = eval_for_loader[eval_for_loader["user_id"].isin(users)]
        trained.eval_loader = make_sequence_loader(
            eval_for_loader, trained, budget, shuffle=False
        )
    else:
        trained = train_fold(
            splits["train"],
            eval_df,
            e_pre,
            budget,
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

    freq = concept_training_frequency(splits["train"], trained.kc_to_idx)
    extra: dict = {}
    if args.signal == "mc_dropout":
        from dh2a_kt.diagnostics.mc_dropout import collect_eval_probs_and_mc_std

        probs, labels, concept_ids, stds = collect_eval_probs_and_mc_std(
            trained, n_samples=mc_samples
        )
        extra = {
            "mc_samples": mc_samples,
            "mc_std_min": float(stds.min()) if stds.size else float("nan"),
            "mc_std_median": float(np.median(stds)) if stds.size else float("nan"),
            "mc_std_max": float(stds.max()) if stds.size else float("nan"),
        }
        print(
            f"collected {len(probs)} predictions; MC-dropout n_samples={mc_samples} "
            f"std min/median/max="
            f"{extra['mc_std_min']:.5f}/{extra['mc_std_median']:.5f}/{extra['mc_std_max']:.5f}"
        )
        diag = stratify_confidence_vs_error(
            probs,
            labels,
            confidence_from_mc_std(stds),
            n_buckets=args.n_buckets,
            strategy=args.strategy,
            min_effect_size=args.min_effect_size,
            signal="mc_dropout",
        )
    else:
        probs, labels, concept_ids = collect_predictions_with_concepts(trained)
        print(
            f"collected {len(probs)} predictions over {len(trained.kc_to_idx)} concepts "
            f"(N_c^train: min={freq.min():.0f} median={np.median(freq):.0f} max={freq.max():.0f}, "
            f"{int((freq == 0).sum())} concepts unseen in training)"
        )
        if args.signal == "predictive":
            cb = confidence_from_predictive_prob(probs)
            extra["predictive_cb_mean"] = float(cb.mean())
            print(
                f"predictive C_B=|2p-1|: mean={cb.mean():.3f} "
                f"p05={np.quantile(cb, 0.05):.3f} p95={np.quantile(cb, 0.95):.3f}"
            )
            diag = stratify_confidence_vs_error(
                probs,
                labels,
                cb,
                n_buckets=args.n_buckets,
                strategy=args.strategy,
                min_effect_size=args.min_effect_size,
                signal="predictive",
            )
        else:
            diag = stratify_by_black_confidence(
                probs,
                labels,
                concept_ids,
                freq,
                kappa_b=args.kappa_b,
                n_buckets=args.n_buckets,
                strategy=args.strategy,
                min_effect_size=args.min_effect_size,
            )

    print()
    print(format_report(diag))

    dataset = dh2_cfg["dataset"]
    output = args.output or (
        REPO_ROOT / "results" / "tables" / diagnostic_json_name(dataset, args.fold, args.signal)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = diag.as_dict()
    payload.update(
        {
            "dataset": dataset,
            "fold": args.fold,
            "split": split_name,
            "signal": args.signal,
            "n_concepts": int(len(trained.kc_to_idx)),
            "n_concepts_unseen_in_training": int((freq == 0).sum()),
            "max_users": args.max_users,
            **extra,
        }
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
