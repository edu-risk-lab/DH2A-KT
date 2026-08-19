"""Slice fold-0 AUC by target-concept rarity, learner position, and item multi-KC status.

Motivation: overall AUC is flat between the graph and no-graph v4 checkpoints
(0.8487 vs 0.8489). A global mean can hide a structural contribution that only
exists where the sequence model starves -- rare concepts and the first few
interactions of a learner. This scores saved checkpoints only; it never trains.

Usage:
    python results/tables/_diag_slice_auc.py configs/xes3g5m.yaml --fold 0 \
        --checkpoint graph=results/checkpoints/xes3g5m_fold0_v4.pt \
        --checkpoint nograph=results/checkpoints/xes3g5m_fold0_v4_nograph.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)
from dh2a_kt.train.tier1 import collect_predictions, resolve_training_budget  # noqa: E402

logger = logging.getLogger(__name__)

FREQ_BINS = [-1, 0, 99, 499, 1999, 9999, np.inf]
FREQ_LABELS = ["unseen", "1-99", "100-499", "500-1999", "2000-9999", "10000+"]
POS_BINS = [0, 5, 20, 50, 100, np.inf]
POS_LABELS = ["1-5", "6-20", "21-50", "51-100", "101+"]


def scored_target_rows(dataset, *, mask_repeats: bool = False) -> np.ndarray:
    """Absolute row indices of every scored target, in loader order.

    Mirrors ``collect_predictions``: position ``t`` predicts ``t+1``, and the
    mask keeps ``t < length - 1``. With ``mask_repeats``, drop targets that
    continue a multi-KC attempt (``is_repeat`` at ``t+1``).
    """
    targets: list[np.ndarray] = []
    for start, raw_end in zip(dataset._starts, dataset._ends, strict=True):
        end = min(raw_end, start + dataset.max_seq_len)
        length = end - start
        if length < 2:
            continue
        rows = np.arange(start + 1, start + length, dtype=np.int64)
        if mask_repeats:
            rows = rows[~dataset._is_repeat[rows]]
        if rows.size:
            targets.append(rows)
    if not targets:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(targets)


def row_level_features(eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-row features in dataset row order (same sort key as UserSequenceDataset).

    ``is_repeat`` marks a row that continues the *same attempt* as its
    predecessor: pykt's KC-level export splits one multi-concept question into
    consecutive rows sharing user/item/timestamp, so such a row carries the
    answer the model has just been shown.
    """
    sorted_df = eval_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    user_ids = sorted_df["user_id"].to_numpy()
    item_ids = sorted_df["item_id"].to_numpy()
    timestamps = sorted_df["timestamp"].to_numpy()
    correct = sorted_df["correct"].to_numpy()
    n = len(user_ids)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return {"position": empty, "is_repeat": empty.astype(bool), "prev_same_label": empty.astype(bool)}
    boundaries = np.where(user_ids[1:] != user_ids[:-1])[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    row_user_start = np.repeat(starts, ends - starts)

    same_user = np.zeros(n, dtype=bool)
    same_user[1:] = user_ids[1:] == user_ids[:-1]
    same_item = np.zeros(n, dtype=bool)
    same_item[1:] = item_ids[1:] == item_ids[:-1]
    same_time = np.zeros(n, dtype=bool)
    same_time[1:] = timestamps[1:] == timestamps[:-1]
    prev_same_label = np.zeros(n, dtype=bool)
    prev_same_label[1:] = correct[1:] == correct[:-1]
    prev_correct = np.zeros(n, dtype=np.float64)
    prev_correct[1:] = correct[:-1]
    return {
        "position": np.arange(n, dtype=np.int64) - row_user_start + 1,
        "is_repeat": same_user & same_item & same_time,
        "prev_same_label": prev_same_label,
        "prev_correct": prev_correct,
    }


def bucket_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def paired_delta_ci(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    *,
    n_boot: int = 300,
    cap: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile CI for AUC(a) - AUC(b) on the same positions (paired resample)."""
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx_pool = np.arange(len(labels))
    if len(idx_pool) > cap:
        idx_pool = rng.choice(idx_pool, size=cap, replace=False)
    y, pa, pb = labels[idx_pool], probs_a[idx_pool], probs_b[idx_pool]
    deltas: list[float] = []
    for _ in range(n_boot):
        take = rng.integers(0, len(y), len(y))
        ys = y[take]
        if len(np.unique(ys)) < 2:
            continue
        deltas.append(roc_auc_score(ys, pa[take]) - roc_auc_score(ys, pb[take]))
    if not deltas:
        return (float("nan"), float("nan"))
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


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
        help="Repeatable. First is treated as the graph arm, second as the baseline.",
    )
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--output-prefix", default="results/tables/xes3g5m_fold0_slice_auc")
    parser.add_argument("--eval-split", choices=("test", "valid+test"), default="valid+test")
    parser.add_argument("--window-mode", choices=("first", "last", "chunked"), default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--mask-repeats", action="store_true")
    parser.add_argument(
        "--extended-parquet",
        type=Path,
        default=None,
        help="Optional FoundationalASSIST extended parquet with hint_used. "
        "Adds target_hint_used slices (has_hint / no_hint).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    import torch

    from dh2a_kt.models.dh2_kt import empty_hyperedge_index
    from dh2a_kt.train.checkpoint import load_trained_fold
    from dh2a_kt.train.greykt import make_sequence_loader

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=args.max_seq_len if args.max_seq_len is not None else train_cfg.get("max_seq_len"),
    )
    budget = replace(budget, batch_size=int(args.eval_batch_size))

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, args.fold)
    if args.eval_split == "test":
        eval_df = splits["test"]
    else:
        eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    logger.info("eval rows=%d train rows=%d", len(eval_df), len(splits["train"]))

    # Rarity is a property of the training split; multi-KC status is item metadata.
    train_freq_by_kc = splits["train"]["kc_id"].value_counts()
    n_kc_per_item = interactions.groupby("item_id")["kc_id"].nunique()

    row_features = row_level_features(eval_df)

    arms: dict[str, dict] = {}
    order: list[str] = []
    target_rows: np.ndarray | None = None
    labels_ref: np.ndarray | None = None
    meta: dict[str, np.ndarray] = {}

    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        path = REPO_ROOT / raw_path
        trained = load_trained_fold(path, device=args.device)
        n_edges = len(trained.clean_hyperedges)
        index = trained.clean_hyperedge_index
        if n_edges == 0:
            index = empty_hyperedge_index(
                trained.device, kinds=tuple(trained.model.config.hyperedge_kinds)
            )
        loader = make_sequence_loader(
            eval_df, trained, budget, shuffle=False, window_mode=args.window_mode
        )
        logger.info(
            "arm=%s arch=%s use_questions=%s hyperedges=%d windows=%d",
            name,
            trained.model.config.architecture,
            trained.model.config.use_questions,
            n_edges,
            len(loader.dataset),
        )
        probs, labels = collect_predictions(
            trained.model,
            loader,
            index,
            trained.device,
            mask_repeats=args.mask_repeats,
        )

        rows = scored_target_rows(loader.dataset, mask_repeats=args.mask_repeats)
        if len(rows) != len(probs):
            raise SystemExit(
                f"alignment failure for {name}: {len(rows)} target rows vs {len(probs)} predictions"
            )
        if target_rows is None:
            target_rows = rows
            labels_ref = labels
            kc_idx_to_id = {v: k for k, v in trained.kc_to_idx.items()}
            item_idx_to_id = {v: k for k, v in trained.item_to_idx.items()}
            tgt_kc_idx = loader.dataset._kc_ids[rows]
            tgt_item_idx = loader.dataset._item_ids[rows]
            tgt_kc_id = np.array([kc_idx_to_id[int(i)] for i in tgt_kc_idx], dtype=np.int64)
            tgt_item_id = np.array([item_idx_to_id[int(i)] for i in tgt_item_idx], dtype=np.int64)
            meta["freq"] = (
                pd.Series(tgt_kc_id).map(train_freq_by_kc).fillna(0).to_numpy(dtype=np.int64)
            )
            meta["pos"] = row_features["position"][rows]
            meta["is_repeat"] = row_features["is_repeat"][rows]
            meta["prev_same_label"] = row_features["prev_same_label"][rows]
            meta["prev_correct"] = row_features["prev_correct"][rows]
            meta["n_kc"] = (
                pd.Series(tgt_item_id).map(n_kc_per_item).fillna(1).to_numpy(dtype=np.int64)
            )
            if args.extended_parquet is not None:
                ext_path = (
                    args.extended_parquet
                    if args.extended_parquet.is_absolute()
                    else REPO_ROOT / args.extended_parquet
                )
                extended = pd.read_parquet(ext_path)
                sorted_eval = eval_df.sort_values(
                    ["user_id", "timestamp", "item_id"]
                ).reset_index(drop=True)
                hint_map = (
                    extended.drop_duplicates(["user_id", "item_id", "timestamp"])
                    .set_index(["user_id", "item_id", "timestamp"])["hint_used"]
                )
                keys = list(
                    zip(
                        sorted_eval["user_id"].astype("int64"),
                        sorted_eval["item_id"].astype("int64"),
                        sorted_eval["timestamp"].astype("int64"),
                    )
                )
                hint_rows = hint_map.reindex(keys).fillna(0).to_numpy(dtype=np.int64)
                meta["hint_used"] = hint_rows[rows] > 0
        else:
            if not np.array_equal(rows, target_rows):
                raise SystemExit(f"arm {name} scored different rows than the first arm")
            if not np.allclose(labels, labels_ref):
                raise SystemExit(f"arm {name} produced different labels than the first arm")

        arms[name] = {"probs": probs, "n_edges": n_edges}
        order.append(name)
        logger.info("arm=%s overall AUC=%.4f n=%d", name, bucket_auc(labels, probs), len(probs))

    assert labels_ref is not None and target_rows is not None
    labels = labels_ref
    freq_bucket = pd.cut(meta["freq"], bins=FREQ_BINS, labels=FREQ_LABELS)
    pos_bucket = pd.cut(meta["pos"], bins=POS_BINS, labels=POS_LABELS)
    multi_bucket = np.where(meta["n_kc"] >= 2, "multi_kc_item", "single_kc_item")

    slices: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(len(labels), bool))]
    for label in FREQ_LABELS:
        slices.append(("target_concept_train_freq", label, np.asarray(freq_bucket == label)))
    for label in POS_LABELS:
        slices.append(("position_in_learner_log", label, np.asarray(pos_bucket == label)))
    for label in ("single_kc_item", "multi_kc_item"):
        slices.append(("target_item_kc_count", label, multi_bucket == label))
    if "hint_used" in meta:
        hint_mask = meta["hint_used"].astype(bool)
        slices.append(("target_hint_used", "has_hint", hint_mask))
        slices.append(("target_hint_used", "no_hint", ~hint_mask))
    repeat_mask = meta["is_repeat"].astype(bool)
    slices.append(("target_is_repeat_row", "repeat_of_same_attempt", repeat_mask))
    slices.append(("target_is_repeat_row", "fresh_attempt", ~repeat_mask))

    from dh2a_kt.data.aux_signals import log_time_gaps

    sorted_eval = eval_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    gap_at_target = log_time_gaps(
        sorted_eval["timestamp"].to_numpy(dtype=np.int64),
        sorted_eval["user_id"].to_numpy(),
    )[target_rows]
    try:
        gap_bucket = pd.qcut(gap_at_target, q=4, duplicates="drop")
    except ValueError:
        gap_bucket = None
    if gap_bucket is not None:
        for label in gap_bucket.categories:
            slices.append(
                (
                    "target_time_gap_quartile",
                    str(label),
                    np.asarray(gap_bucket == label),
                )
            )

    n_repeat = int(repeat_mask.sum())
    leak_rate = (
        float(meta["prev_same_label"][repeat_mask].mean()) if n_repeat else float("nan")
    )
    print(
        f"\nrepeat rows scored: {n_repeat} / {len(labels)} "
        f"({100.0 * n_repeat / max(len(labels), 1):.2f}%); "
        f"share the previous row's answer in {100.0 * leak_rate:.2f}% of cases"
    )

    # How much of the reported AUC is reachable with no learning at all? These
    # rules use only the previous row's answer, which the standard protocol
    # feeds to the model as input.
    base_rate = float(labels.mean())
    prev = meta["prev_correct"]
    copy_repeat_only = np.where(repeat_mask, np.where(prev > 0.5, 0.99, 0.01), base_rate)
    copy_always = np.where(prev > 0.5, 0.99, 0.01)
    rules = {
        "rule_copy_prev_answer_on_repeats_only": copy_repeat_only,
        "rule_copy_prev_answer_always": copy_always,
    }
    print("\nzero-learning reference rules, standard (leaky) protocol:")
    for rule_name, rule_probs in rules.items():
        print(
            f"  {rule_name}: overall AUC={bucket_auc(labels, rule_probs):.4f} "
            f"| fresh-attempt AUC={bucket_auc(labels[~repeat_mask], rule_probs[~repeat_mask]):.4f}"
        )
    print()

    rows_out: list[dict] = []
    primary, baseline = order[0], (order[1] if len(order) > 1 else None)
    for dim, label, mask in slices:
        n = int(mask.sum())
        row = {
            "slice_dim": dim,
            "slice": label,
            "n": n,
            "positive_rate": float(labels[mask].mean()) if n else float("nan"),
        }
        for name in order:
            row[f"auc_{name}"] = bucket_auc(labels[mask], arms[name]["probs"][mask])
        if baseline is not None:
            row["delta"] = row[f"auc_{primary}"] - row[f"auc_{baseline}"]
            lo, hi = paired_delta_ci(
                labels[mask],
                arms[primary]["probs"][mask],
                arms[baseline]["probs"][mask],
                n_boot=args.n_boot,
            )
            row["delta_ci_lo"], row["delta_ci_hi"] = lo, hi
            row["significant"] = bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0)
        rows_out.append(row)

    table = pd.DataFrame(rows_out)
    csv_path = REPO_ROOT / f"{args.output_prefix}.csv"
    json_path = REPO_ROOT / f"{args.output_prefix}.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "fold": args.fold,
                "n_predictions": int(len(labels)),
                "arms": {k: {"n_hyperedges": v["n_edges"]} for k, v in arms.items()},
                "primary": primary,
                "baseline": baseline,
                "slices": rows_out,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwritten: {csv_path}")
    print(f"written: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
