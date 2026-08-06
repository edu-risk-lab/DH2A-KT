#!/usr/bin/env python3
"""M7 driver: Junyi ground-truth cross-validation (E_pre vs hyperedge chains).

Usage:
    python scripts/06_run_gt_crossval_junyi.py configs/junyi.yaml --fold 0

Requires P0 Junyi preprocessing + graph_builder (``kc_name_to_id.json``,
``e_pre_train_only.csv``, expert ``junyi_dag.csv``).

Writes ``results/tables/junyi_fold<N>_gt_crossval/`` (Table-15-style CSVs).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.eval.gt_crossval import build_gt_crossval_from_p0_exports, write_gt_crossval_result
from dh2a_kt.hyperedge.p0_inputs import P0_ROOT, load_configs

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, default=Path("configs/junyi.yaml"), nargs="?")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/tables/<dataset>_fold<N>_gt_crossval/",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dh2_cfg, p0_cfg, p0_config_path = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    fold = args.fold

    print(f"[fold {fold}] dataset={dataset} role={dh2_cfg.get('role')}")
    print(f"  p0_config: {p0_config_path}")

    result = build_gt_crossval_from_p0_exports(
        dh2_cfg,
        p0_cfg,
        fold=fold,
        p0_root=P0_ROOT,
    )

    output_dir = args.output_dir or (
        REPO_ROOT / "results" / "tables" / f"{dataset}_fold{fold}_gt_crossval"
    )
    write_gt_crossval_result(result, output_dir)

    e_k = result.e_pre_summary["K_equal_|expert|"]
    h_k = result.hyperedge_summary["K_equal_|expert|"]
    print(f"  expert edges (matched): {result.n_expert_edges_matched}")
    print(f"  e_pre edges:            {result.n_e_pre_edges}")
    print(f"  hyperedge inferred:     {result.n_hyperedge_inferred_edges}")
    print(
        f"  E_pre @ |E_expert|: prec={e_k['edge_precision']:.3f} "
        f"rec={e_k['edge_recall']:.3f} f1={e_k['edge_f1']:.3f}"
    )
    print(
        f"  Hyperedge @ |E_expert|: prec={h_k['edge_precision']:.3f} "
        f"rec={h_k['edge_recall']:.3f} f1={h_k['edge_f1']:.3f}"
    )
    print(f"  alignment_rate:         {result.alignment_report.get('alignment_rate', 0):.3f}")
    print(f"  written:                {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
