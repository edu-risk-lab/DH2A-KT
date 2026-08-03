"""Recompute results/tables/leakage_metrics.csv from saved fold graphs (CPU).

The legacy ``eoc`` column now stores |rho| (Eq. rho-edge-outcome). Older CSV
rows may still hold sqrt(2+2rho^2); this script overwrites them from artefacts.

Usage:
    python -m scripts.regenerate_leakage_metrics
    python -m scripts.regenerate_leakage_metrics --sync-tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.graph_builder import build_q_matrix_from_train
from src.io_utils import load_interactions, load_yaml
from src.leakage_metrics import compute_leakage_row, merge_leakage_metrics_csv
from src.split_checker import learner_based_folds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "configs/xes3g5m.yaml",
    "configs/assist2012.yaml",
    "configs/junyi.yaml",
    "configs/synthetic_c2.yaml",
    "configs/synthetic_c5.yaml",
]


def _train_ratio(cfg: dict) -> float:
    ratios = cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2])
    return float(ratios[0]) if ratios else 0.7


def recompute_dataset(config_path: Path) -> list[dict]:
    cfg = load_yaml(config_path)
    dataset = str(cfg["dataset"])
    processed = Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet"))
    df = load_interactions(processed)
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    split_cfg = cfg.get("split", {})
    train_ratio = _train_ratio(cfg)
    rows: list[dict] = []

    for fold, _split_seed, splits in learner_based_folds(
        df, ratios, split_cfg, default_seed=int(split_cfg.get("seed", 42))
    ):
        fold_dir = ROOT / "data/processed" / dataset / f"fold_{fold}"
        pre_path = fold_dir / "e_pre_train_only.csv"
        sim_path = fold_dir / "e_sim_train_only.csv"
        if not pre_path.exists():
            raise FileNotFoundError(f"Missing prerequisite graph: {pre_path}")
        pre_df = pd.read_csv(pre_path)
        sim_df = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame(
            columns=["src_kc", "dst_kc", "weight", "source"]
        )
        q_train = build_q_matrix_from_train(splits["train"])
        rows.append(
            compute_leakage_row(
                dataset=dataset,
                fold=int(fold),
                splits=splits,
                pre_df=pre_df,
                sim_df=sim_df,
                q_train=q_train,
                train_ratio=train_ratio,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Dataset config (repeatable). Default: all five benchmark configs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/tables/leakage_metrics.csv",
    )
    parser.add_argument(
        "--sync-tex",
        action="store_true",
        help="Also write leakage_metrics.tex via generate_phase_c_tables writer.",
    )
    args = parser.parse_args()
    configs = [Path(p) for p in (args.config or DEFAULT_CONFIGS)]

    for cfg_path in configs:
        if not cfg_path.exists():
            print(f"SKIP missing config: {cfg_path}")
            continue
        cfg = load_yaml(cfg_path)
        processed = Path(cfg.get("processed_path", f"data/processed/{cfg['dataset']}.parquet"))
        if not processed.exists():
            print(f"SKIP {cfg['dataset']}: missing {processed}")
            continue
        rows = recompute_dataset(cfg_path)
        merge_leakage_metrics_csv(rows, str(rows[0]["dataset"]), args.out)
        print(f"Updated {args.out} for {rows[0]['dataset']} ({len(rows)} folds)")

    df = pd.read_csv(args.out)
    summary = df.groupby("dataset")["eoc"].agg(["mean", "std"])
    print("\n|rho| summary (eoc column):")
    print(summary.to_string())

    if args.sync_tex:
        from scripts.generate_phase_c_tables import write_leakage_metrics_tex

        tex_path = args.out.with_suffix(".tex")
        write_leakage_metrics_tex(df, tex_path)
        print(f"Wrote {tex_path}")

        for mirror in (ROOT / "paper/submission_APIN/leakage_metrics.tex",):
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Synced {mirror}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
