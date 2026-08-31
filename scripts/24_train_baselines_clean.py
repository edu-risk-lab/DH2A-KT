#!/usr/bin/env python3
"""Train pyKT baselines under a chosen window rule and repeat-masking protocol.

The vendored P0 driver (``external/p0_leakage_audit/src/pykt_engine.py``) is used
unchanged: every model, its loss and its AUC are P0's. Only the exported sequence
CSVs differ, because pyKT routes both the loss and the metric through the single
mask it derives from the ``selectmasks`` column. Three configurations matter:

  A  --window-mode last    (no mask)  reproduces P0's published table
  B  --window-mode chunked (no mask)  same position set as DH2-KT v4, still leaky
  C  --window-mode chunked --mask-repeats   the clean protocol

A must reproduce P0's numbers before B or C mean anything: it is the check that
this harness does not accidentally weaken the baselines.

Usage:
    python scripts/24_train_baselines_clean.py configs/xes3g5m.yaml --fold 0 \
        --model dkt --window-mode last --tag A_repro --device cuda
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
P0_ROOT = REPO_ROOT / "external" / "p0_leakage_audit"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(P0_ROOT))

from dh2a_kt.baselines.pykt_clean import export_pykt_sequences  # noqa: E402
from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)

logger = logging.getLogger(__name__)

# P0's published xes3g5m fold-0 numbers (baselines/p0_reused/baseline_fold_results.csv,
# graph_construction=train_only). The AUC there is computed on the fold -1 test
# file with tail windows, so only configuration A is comparable to it.
P0_PUBLISHED = {
    "xes3g5m": {
        0: {
            "dkt": 0.8577610832042228,
            "simplekt": 0.8744411161851224,
            "akt": 0.8740823569517637,
            "gkt": 0.8345570064591779,
            "gikt": 0.8775823299875001,
            "dgekt": 0.8409904983626277,
        }
    }
}


def _hyperparams(p0_cfg: dict, model: str) -> dict:
    for item in p0_cfg.get("baselines") or []:
        if item.get("name") == model and isinstance(item.get("hyperparams"), dict):
            return dict(item["hyperparams"])
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", required=True, help="pyKT name, e.g. dkt / simplekt / gkt")
    parser.add_argument("--window-mode", choices=("last", "first", "chunked"), default="last")
    parser.add_argument("--mask-repeats", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", required=True, help="Names the work dir and the output row")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: P0's setting")
    parser.add_argument("--epochs", type=int, default=None, help="Default: P0's setting")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stop patience on valid AUC (default: P0/engine 3). Use 5 to match DH2.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override pyKT/P0 max_seq_len for export and model seq_len (fair L=400 vs DH2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.device == "cpu":
        import os

        os.environ["FORCE_CPU"] = "1"

    from src.pykt_engine import run_pykt_fold  # noqa: E402
    from src.pykt_export import build_dense_maps  # noqa: E402

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    pykt_cfg = p0_cfg.get("pykt", {}) if isinstance(p0_cfg.get("pykt"), dict) else {}
    max_seq_len = int(
        args.max_seq_len if args.max_seq_len is not None else pykt_cfg.get("max_seq_len", 200)
    )
    graph_tag = str(pykt_cfg.get("gkt_graph_tag", "p0_protocol"))

    hp = _hyperparams(p0_cfg, args.model)
    epochs = int(args.epochs or hp.get("epochs", pykt_cfg.get("epochs", 30)))
    batch_size = int(args.batch_size or hp.get("batch_size", pykt_cfg.get("batch_size", 64)))
    lr = float(hp.get("lr", pykt_cfg.get("lr", 1e-3)))
    if args.patience is not None:
        hp["patience"] = int(args.patience)

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]

    q_map, c_map = build_dense_maps(train_df)
    work_dir = REPO_ROOT / "results" / "pykt_work_clean" / dataset / f"fold_{args.fold}" / args.tag
    stats = export_pykt_sequences(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        q_map=q_map,
        c_map=c_map,
        out_dir=work_dir,
        max_seq_len=max_seq_len,
        window_mode=args.window_mode,
        mask_repeats=args.mask_repeats,
    )
    logger.info(
        "export: %s window=%s mask_repeats=%s | test sequences=%d scored=%d masked=%d",
        args.tag,
        args.window_mode,
        args.mask_repeats,
        stats["test_sequences"],
        stats["test_positions_scored"],
        stats["test_positions_masked"],
    )

    graph_npz = None
    if args.model in ("gkt", "skt", "dygkt", "dgekt"):
        from src.pykt_graph_matrix import edges_to_gkt_matrix, write_gkt_graph_npz

        graph_npz = work_dir / f"gkt_graph_{graph_tag}.npz"
        if not graph_npz.exists():
            edge_csvs = [
                P0_ROOT / "data" / "processed" / dataset / f"fold_{args.fold}" / name
                for name in ("e_pre_train_only.csv", "e_sim_train_only.csv")
            ]
            matrix = edges_to_gkt_matrix(stats["num_c"], edge_csvs, c_map)
            write_gkt_graph_npz(graph_npz, matrix)

    started = time.time()
    auc, acc, nll, note, ts, ps, us, cs = run_pykt_fold(
        display_model=args.model,
        pykt_name=args.model,
        work_dir=work_dir,
        num_q=stats["num_q"],
        num_c=stats["num_c"],
        graph_npz=graph_npz,
        graph_tag=graph_tag,
        hyperparams=hp,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=int(args.seed) + int(args.fold) * 97,
        max_seq_len=max_seq_len,
        include_valid_predictions=False,
    )
    elapsed = time.time() - started

    # Per-position predictions with learner ids enable a bootstrap that resamples
    # learners, which is how the DH2-KT comparison gets a confidence interval.
    np.savez(work_dir / "test_predictions.npz", ts=ts, ps=ps, cs=cs, us=us)

    published = P0_PUBLISHED.get(dataset, {}).get(args.fold, {}).get(args.model)
    row = {
        "dataset": dataset,
        "fold": args.fold,
        "model": args.model,
        "tag": args.tag,
        "window_mode": args.window_mode,
        "mask_repeats": bool(args.mask_repeats),
        "eval_split": "test",
        "auc": auc,
        "acc": acc,
        "nll": nll,
        "n_scored": int(len(ts)),
        "batch_size": batch_size,
        "epochs": epochs,
        "max_seq_len": max_seq_len,
        "lr": lr,
        "p0_published_auc": published,
        "delta_vs_published": (auc - published) if published is not None else None,
        "minutes": round(elapsed / 60.0, 1),
        "note": note,
    }

    output = args.output or (
        REPO_ROOT / "results" / "tables" / f"{dataset}_fold{args.fold}_baselines_protocol.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    if output.exists():
        table = pd.concat([pd.read_csv(output), table], ignore_index=True)
    table.to_csv(output, index=False)

    print()
    print(
        f"[{args.tag}] {args.model} AUC={auc:.6f} acc={acc:.4f} n={len(ts)} "
        f"L={max_seq_len} ({elapsed / 60:.1f} min)"
    )
    if published is not None:
        print(f"  P0 published: {published:.6f} | delta: {auc - published:+.6f}")
        if args.window_mode == "last" and not args.mask_repeats:
            verdict = "REPRODUCED" if abs(auc - published) < 0.01 else "MISMATCH"
            print(f"  configuration A reproduction check: {verdict} (tolerance 0.01)")
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
