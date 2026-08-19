#!/usr/bin/env python3
"""Score saved DH2-KT folds under both the P0 protocol and the clean protocol.

The P0 baseline table scores every KC row produced by pyKT's KC-level export.
That export splits one attempt at a multi-concept question into consecutive rows
sharing user, item, timestamp and answer, so a large share of eval positions are
answerable by copying the answer the model was just given. This driver reports
the standard AUC and the repeat-masked AUC side by side, for checkpoints that
already exist -- it never trains.

Usage:
    python scripts/23_eval_clean_protocol.py configs/xes3g5m.yaml --fold 0 --device cuda \
        --checkpoint v4=results/checkpoints/xes3g5m_fold0_v4.pt \
        --eval-split test --window-mode chunked --max-seq-len 400
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)
from dh2a_kt.train.tier1 import evaluate_auc, resolve_training_budget  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeatable, e.g. v4q=results/checkpoints/xes3g5m_fold0_v4q.pt",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help=(
            "Directory for <name>_<protocol>.npz with per-position label, probability "
            "and learner id, so a bootstrap can resample learners."
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=("test", "valid+test"),
        default="valid+test",
        help=(
            "P0's run_pykt_fold scores the fold -1 test file only, so its published "
            "AUC is test-only even though its results CSV labels the row 'valid+test'. "
            "Use 'test' to compare against that table."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=("first", "last", "chunked"),
        default=None,
        help=(
            "Position set to score. Default follows the architecture. Use 'last' to "
            "match P0's pyKT export, which slices the final max_seq_len rows per "
            "learner; 'chunked' scores the whole log and is not the same position set."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help=(
            "Override sequence window length. Default follows the DH2 training "
            "config (currently 200). L=400 checkpoints must pass 400 here or they "
            "are scored on a truncated position set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: results/tables/<dataset>_fold<N>_clean_protocol.csv",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from dh2a_kt.models.dh2_kt import empty_hyperedge_index
    from dh2a_kt.train.checkpoint import load_trained_fold
    from dh2a_kt.train.greykt import make_sequence_loader

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=(
            args.max_seq_len
            if args.max_seq_len is not None
            else train_cfg.get("max_seq_len")
        ),
    )
    budget = replace(budget, batch_size=int(args.eval_batch_size))

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    if args.eval_split == "test":
        eval_df = splits["test"]
    else:
        eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)

    rows: list[dict] = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        trained = load_trained_fold(REPO_ROOT / raw_path, device=args.device)
        index = trained.clean_hyperedge_index
        if not trained.clean_hyperedges:
            index = empty_hyperedge_index(
                trained.device, kinds=tuple(trained.model.config.hyperedge_kinds)
            )
        loader = make_sequence_loader(
            eval_df, trained, budget, shuffle=False, window_mode=args.window_mode
        )
        p0_auc, p0_n = evaluate_auc(
            trained.model, loader, index, trained.device, mask_repeats=False
        )
        clean_auc, clean_n = evaluate_auc(
            trained.model, loader, index, trained.device, mask_repeats=True
        )
        if args.save_predictions is not None:
            from dh2a_kt.train.tier1 import collect_predictions

            args.save_predictions.mkdir(parents=True, exist_ok=True)
            for protocol, mask_repeats in (("p0", False), ("clean", True)):
                ps, ts, us = collect_predictions(
                    trained.model,
                    loader,
                    index,
                    trained.device,
                    mask_repeats=mask_repeats,
                    return_users=True,
                )
                np.savez(
                    args.save_predictions / f"{name}_{protocol}.npz", ts=ts, ps=ps, us=us
                )
        rows.append(
            {
                "checkpoint": name,
                "eval_split": args.eval_split,
                "window_mode": args.window_mode or "architecture_default",
                "architecture": trained.model.config.architecture,
                "use_questions": bool(trained.model.config.use_questions),
                "n_hyperedges": len(trained.clean_hyperedges),
                "p0_protocol_auc": p0_auc,
                "p0_protocol_n": p0_n,
                "clean_protocol_auc": clean_auc,
                "clean_protocol_n": clean_n,
                "auc_attributable_to_repeats": p0_auc - clean_auc,
            }
        )
        print(
            f"[{name}] P0 protocol AUC={p0_auc:.4f} (n={p0_n}) | "
            f"clean AUC={clean_auc:.4f} (n={clean_n}) | "
            f"difference={p0_auc - clean_auc:+.4f}"
        )

    table = pd.DataFrame(rows)
    dataset = dh2_cfg["dataset"]
    output = args.output or (
        REPO_ROOT / "results" / "tables" / f"{dataset}_fold{args.fold}_clean_protocol.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nwritten: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
