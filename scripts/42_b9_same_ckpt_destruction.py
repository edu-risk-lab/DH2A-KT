#!/usr/bin/env python3
"""Hau B9: one clean score per fold, then every destruction on that loader.

Does not retrain. Uses the v2 diagnostic checkpoints
``results/checkpoints/xes3g5m_fold{N}.pt``. Writes a single table so
``auc_clean`` is shared across operators.

    python scripts/42_b9_same_ckpt_destruction.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.eval.manipulation_check import run_manipulation_check
from dh2a_kt.eval.tier1_eval import make_eval_auc_fn
from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids
from dh2a_kt.train.checkpoint import load_trained_fold
from dh2a_kt.train.greykt import make_sequence_loader
from dh2a_kt.train.tier1 import resolve_training_budget

OUT_CSV = REPO_ROOT / "results" / "tables" / "b9_same_ckpt_destruction.csv"
OUT_JSON = REPO_ROOT / "results" / "tables" / "b9_same_ckpt_destruction.json"
OPS = ("node_drop", "degree_preserving_rewire", "relation_label_permute")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--checkpoint-template",
        default="results/checkpoints/xes3g5m_fold{fold}.pt",
    )
    args = parser.parse_args()

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / "configs" / "xes3g5m.yaml")
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=train_cfg.get("max_seq_len"),
    )
    interactions = load_interactions_with_ids(p0_cfg)
    rows: list[dict[str, object]] = []
    for fold in args.folds:
        ckpt = REPO_ROOT / args.checkpoint_template.format(fold=fold)
        if not ckpt.is_file():
            print(f"SKIP fold {fold}: missing {ckpt}", flush=True)
            continue
        trained = load_trained_fold(ckpt, device=args.device)
        eval_df = pd.concat(
            [
                get_fold_splits(interactions, p0_cfg, fold)["valid"],
                get_fold_splits(interactions, p0_cfg, fold)["test"],
            ],
            ignore_index=True,
        )
        trained.eval_loader = make_sequence_loader(
            eval_df, trained, budget, shuffle=False
        )
        hyperedges = list(trained.clean_hyperedges)
        eval_fn = make_eval_auc_fn(trained)
        auc_clean = float(eval_fn(hyperedges))
        clean_hash = hashlib.sha256(f"{fold}:{auc_clean:.10f}:{ckpt}".encode()).hexdigest()[:16]
        print(f"[fold {fold}] shared auc_clean={auc_clean:.6f} n_he={len(hyperedges)}", flush=True)
        for op in OPS:
            result = run_manipulation_check(
                hyperedges,
                eval_auc_fn=eval_fn,
                operator=op,
                seed=42,
            )
            # Re-use the shared clean AUC; ignore the operator-local recompute
            # if it drifted (that drift is the B9 bug).
            rows.append(
                {
                    "fold": fold,
                    "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                    "operator": op,
                    "auc_clean_shared": auc_clean,
                    "auc_clean_operator_local": result.auc_clean,
                    "clean_auc_abs_diff": abs(result.auc_clean - auc_clean),
                    "auc_destroyed": result.auc_destroyed,
                    "auc_drop_from_shared": auc_clean - result.auc_destroyed,
                    "ddr": result.ddr,
                    "clean_hash": clean_hash,
                    "n_hyperedges": len(hyperedges),
                }
            )
            print(
                f"  {op}: destroyed={result.auc_destroyed:.6f} "
                f"local_clean={result.auc_clean:.6f} "
                f"|local-shared|={abs(result.auc_clean - auc_clean):.6f}",
                flush=True,
            )
    if not rows:
        return 2
    frame = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)
    max_drift = float(frame["clean_auc_abs_diff"].max())
    payload = {
        "n_rows": int(len(frame)),
        "max_operator_local_vs_shared_clean": max_drift,
        "pass_same_clean": max_drift < 5e-4,
        "csv": str(OUT_CSV.relative_to(REPO_ROOT)),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
