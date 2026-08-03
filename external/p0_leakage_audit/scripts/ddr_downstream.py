#!/usr/bin/env python3
"""DDR -> downstream graph-KT accuracy study (``small_downstream``).

For each fold of a dataset we perturb the train-only prerequisite graph
``E_pre`` with one augmentation operator at a given strength ``p``, rebuild the
KC--KC adjacency (perturbed ``E_pre`` unioned with the unchanged ``E_sim``),
retrain a graph-consuming KT model, and record the test AUC alongside the DDR of
that perturbation. This links the structural DDR diagnostic (Section: DAG
Disruption Rate) to downstream KT accuracy: if DDR is meaningful, higher DDR
should track larger AUC degradation, and the prerequisite-preserving operator
should degrade accuracy less than label-agnostic dropping at a matched budget.

Model choice (``--model``). Only models that ingest the KC--KC prerequisite
adjacency are valid here, because DDR perturbs ``E_pre``:
  - ``gkt`` (default): the canonical graph-KT model; most graph-dependent but
    the slowest to train (small batch, per-step graph propagation).
  - ``skt`` / ``dygkt`` / ``dgekt``: native lightweight models that consume the
    same adjacency (``graph_npz`` -> ``adj_matrix``) and train substantially
    faster; good for a cheaper sweep or cross-architecture robustness.
``gikt`` is intentionally rejected: it consumes the question--concept bipartite
graph and ignores ``E_pre``, so perturbing ``E_pre`` would have no effect.

Within a fold the pyKT sequence files are written once and reused across all
(operator, p) variants; only the graph ``.npz`` changes, so the marginal cost
per variant is one model training run.

Outputs (appended row-by-row, resumable): results/tables/ddr_downstream.csv

Examples
--------
# real run on the GPU server
python -m scripts.ddr_downstream --config configs/assist2012.yaml
python -m scripts.ddr_downstream --config configs/xes3g5m.yaml

# CPU smoke test of the data path (no torch / no training)
python -m scripts.ddr_downstream --config configs/synthetic_c2.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.dag_disruption import (
    apply_attribute_mask,
    apply_edge_drop,
    apply_node_drop,
    apply_prereq_preserve,
    apply_subgraph_sampling,
    compute_dag_disruption_rate,
)
from src.io_utils import load_interactions, load_yaml
from src.pykt_export import build_dense_maps, dataframe_to_pykt_csvs
from src.pykt_graph_matrix import edges_to_gkt_matrix, write_gkt_graph_npz
from src.split_checker import learner_based_folds

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

OPERATORS = {
    "edge_drop": apply_edge_drop,
    "node_drop": apply_node_drop,
    "prereq_preserve": apply_prereq_preserve,
    "subgraph": apply_subgraph_sampling,
    "attr_mask": apply_attribute_mask,
}

GRAPH_MODELS = ("gkt", "skt", "dygkt", "dgekt")
_KEY_COLS = ["dataset", "model", "fold", "operator", "p"]


def _load_done(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if not set(_KEY_COLS).issubset(df.columns):
        return set()
    return {
        (str(r.dataset), str(r.model), int(r.fold), str(r.operator), round(float(r.p), 4))
        for r in df.itertuples()
    }


def _append_row(out_csv: Path, row: dict) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="gkt",
        choices=GRAPH_MODELS,
        help="Graph-KT model that ingests the KC--KC E_pre adjacency. GIKT is not allowed (it ignores E_pre).",
    )
    parser.add_argument("--operators", nargs="+", default=["edge_drop", "node_drop", "prereq_preserve"])
    parser.add_argument("--ps", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Train only the unperturbed graph (operator=none, p=0). Skips augmentation grid.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override GKT batch size from config (DDR multiseed attested batch=8 on XES3G5M).",
    )
    parser.add_argument("--perturb-seed", type=int, default=42, help="Seed for the augmentation operator.")
    parser.add_argument("--experiment-seed", type=int, default=42, help="Base seed for fold splitting and GKT fit.")
    parser.add_argument("--out", type=Path, default=ROOT / "results/tables/ddr_downstream.csv")
    parser.add_argument("--max-folds", type=int, default=None, help="Process at most this many folds (calibration/smoke test).")
    parser.add_argument("--only-fold", type=int, default=None, help="Process ONLY this specific fold.")
    parser.add_argument("--dry-run", action="store_true", help="Build graphs/npz and record DDR only; skip GKT training (no torch).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    run_pykt_fold = None
    if not args.dry_run:
        import torch  # noqa: F401  (fail fast if the GPU stack is missing)

        from src.pykt_engine import run_pykt_fold as _rpf

        run_pykt_fold = _rpf

    cfg = load_yaml(args.config)
    if "split" in cfg and isinstance(cfg["split"], dict):
        cfg["split"]["seed"] = args.experiment_seed
    dataset = cfg["dataset"]
    df = load_interactions(Path(cfg.get("processed_path", f"data/processed/{dataset}.parquet")))
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))

    py_all = cfg.get("pykt", {}) if isinstance(cfg.get("pykt"), dict) else {}
    max_seq_len = int(py_all.get("max_seq_len", 200))
    graph_tag = str(py_all.get("gkt_graph_tag", "p0_protocol"))
    hp: dict = {}
    for item in cfg.get("baselines") or []:
        if item.get("name") == args.model and isinstance(item.get("hyperparams"), dict):
            hp = dict(item["hyperparams"])
            break
    epochs = int(hp.get("epochs", py_all.get("epochs", 30)))
    batch_size = int(hp.get("batch_size", py_all.get("batch_size", 64)))
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    lr = float(hp.get("lr", py_all.get("lr", 1e-3)))

    done = _load_done(args.out)
    # ("none", 0.0) is the DDR=0 baseline graph; the rest are perturbations.
    variants: list[tuple[str, float]] = [("none", 0.0)]
    if not args.baseline_only:
        variants += [(op, round(float(pp), 4)) for op in args.operators for pp in args.ps]

    for fold, split_seed, splits in learner_based_folds(df, ratios, cfg.get("split", {}), default_seed=args.experiment_seed):
        if args.max_folds is not None and int(fold) >= int(args.max_folds):
            break
        if args.only_fold is not None and int(fold) != int(args.only_fold):
            continue
        train = splits["train"]
        q_map, c_map = build_dense_maps(train)
        work_dir = ROOT / "results/pykt_work" / dataset / f"fold_{fold}_seed_{split_seed}" / "ddr_downstream" / args.model
        num_q, num_c = dataframe_to_pykt_csvs(
            train_df=train,
            valid_df=splits["valid"],
            test_df=splits["test"],
            q_map=q_map,
            c_map=c_map,
            out_dir=work_dir,
            max_seq_len=max_seq_len,
        )

        epre_path = ROOT / "data/processed" / dataset / f"fold_{fold}" / "e_pre_train_only.csv"
        esim_path = ROOT / "data/processed" / dataset / f"fold_{fold}" / "e_sim_train_only.csv"
        original = pd.read_csv(epre_path) if epre_path.exists() else pd.DataFrame(columns=["src_kc", "dst_kc", "weight"])

        for operator, pp in variants:
            key = (dataset, args.model, int(fold), operator, round(float(pp), 4))
            if key in done:
                logger.info("skip (already done): %s", key)
                continue

            if operator == "none":
                perturbed = original.copy()
                ddr = 0.0
            else:
                perturbed = OPERATORS[operator](original, float(pp), int(args.perturb_seed))
                ddr = compute_dag_disruption_rate(original, perturbed)

            pert_csv = work_dir / f"e_pre_{operator}_{pp:.2f}_seed{args.experiment_seed}.csv"
            perturbed.to_csv(pert_csv, index=False)
            npz = work_dir / f"gkt_graph_{operator}_{pp:.2f}_seed{args.experiment_seed}.npz"
            mat = edges_to_gkt_matrix(num_c, [pert_csv, esim_path], c_map)
            write_gkt_graph_npz(npz, mat)

            row: dict = {
                "dataset": dataset,
                "model": args.model,
                "fold": int(fold),
                "split_seed": int(split_seed),
                "operator": operator,
                "p": float(pp),
                "ddr": round(float(ddr), 6),
                "n_edges_orig": int(len(original)),
                "n_edges_pert": int(len(perturbed)),
                "num_c": int(num_c),
            }

            if args.dry_run:
                row.update({"auc": float("nan"), "acc": float("nan"), "nll": float("nan"), "status": "dry_run"})
                logger.info("[dry-run] %s fold=%s ddr=%.3f edges %d->%d", (operator, pp), fold, ddr, len(original), len(perturbed))
            else:
                fit_seed = int(args.experiment_seed) + int(fold) * 97
                auc, acc, nll, note, *_ = run_pykt_fold(
                    display_model=args.model,
                    pykt_name=args.model,
                    work_dir=work_dir,
                    num_q=num_q,
                    num_c=num_c,
                    graph_npz=npz,
                    graph_tag=f"{operator}_{pp:.2f}_seed{args.experiment_seed}",
                    hyperparams=hp,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=fit_seed,
                    max_seq_len=max_seq_len,
                )
                row.update({"auc": float(auc), "acc": float(acc), "nll": float(nll), "status": "pykt_checkpoint"})
                logger.info("done %s fold=%s ddr=%.3f auc=%.4f", (operator, pp), fold, ddr, auc)

            _append_row(args.out, row)

    logger.info("Finished. Results in %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
