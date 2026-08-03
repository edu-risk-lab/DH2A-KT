"""Compute per-edge held-out evidence share distribution (B7.1).

Usage:
    python -m scripts.compute_edge_share_distribution
    python -m scripts.compute_edge_share_distribution --sync-tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.graph_builder import build_q_matrix_from_train
from src.io_utils import load_interactions, load_yaml
from src.leakage_metrics import compute_edge_heldout_shares, summarize_edge_heldout_shares
from src.split_checker import learner_based_folds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "configs/xes3g5m.yaml",
    "configs/assist2012.yaml",
    "configs/junyi.yaml",
    "configs/synthetic_c2.yaml",
    "configs/synthetic_c5.yaml",
]


def _merge_edge_share_rows(rows: list[dict], out_path: Path) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        current = pd.read_csv(out_path)
        if not current.empty and "dataset" in current.columns:
            drop_ds = set(new_df["dataset"].astype(str))
            current = current[~current["dataset"].astype(str).isin(drop_ds)]
            merged = pd.concat([current, new_df], ignore_index=True)
        else:
            merged = new_df
    else:
        merged = new_df
    merged = merged.sort_values(["dataset", "fold"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/tables/edge_share_summary.csv",
    )
    parser.add_argument("--sync-tex", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    for cfg_path in [Path(p) for p in (args.config or DEFAULT_CONFIGS)]:
        if not cfg_path.exists():
            print(f"SKIP missing config: {cfg_path}")
            continue
        cfg = load_yaml(cfg_path)
        dataset = str(cfg["dataset"])
        processed = Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet"))
        if not processed.exists():
            print(f"SKIP {dataset}: missing {processed}")
            continue

        df = load_interactions(processed)
        ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
        split_cfg = cfg.get("split", {})

        for fold, _split_seed, splits in learner_based_folds(
            df, ratios, split_cfg, default_seed=int(split_cfg.get("seed", 42))
        ):
            fold_dir = ROOT / "data/processed" / dataset / f"fold_{fold}"
            pre_path = fold_dir / "e_pre_train_only.csv"
            sim_path = fold_dir / "e_sim_train_only.csv"
            if not pre_path.exists():
                print(f"SKIP {dataset} fold {fold}: missing {pre_path}")
                continue
            pre_df = pd.read_csv(pre_path)
            sim_df = (
                pd.read_csv(sim_path)
                if sim_path.exists()
                else pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"])
            )
            held_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
            q_train = build_q_matrix_from_train(splits["train"])
            shares = compute_edge_heldout_shares(
                pre_df, sim_df, splits["train"], held_df, q_train
            )
            summary = summarize_edge_heldout_shares(shares)
            rows.append({"dataset": dataset, "fold": int(fold), **summary})

    if not rows:
        raise SystemExit("No edge-share rows computed.")

    out_df = _merge_edge_share_rows(rows, args.out)
    print(f"Wrote {args.out} ({len(out_df)} rows)")
    print(out_df.groupby("dataset")[["frac_share_gt_50", "share_median", "share_p90"]].mean().round(4))

    if args.sync_tex:
        from scripts.generate_phase_c_tables import write_leakage_metrics_tex

        leak_path = ROOT / "results/tables/leakage_metrics.csv"
        if not leak_path.exists():
            raise SystemExit(f"Missing {leak_path}")
        write_leakage_metrics_tex(pd.read_csv(leak_path), ROOT / "results/tables/leakage_metrics.tex")
        for mirror in (ROOT / "paper/submission_APIN/leakage_metrics.tex",):
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_text(
                (ROOT / "results/tables/leakage_metrics.tex").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            print(f"Synced {mirror}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
