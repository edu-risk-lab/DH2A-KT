"""Direct leakage diagnostics (ECR_flag, ECR_overlap, |rho|, TBVR) for P0 graph construction.

ECR_overlap counts edges whose retained patterns align with held-out interaction mass.
ECR_flag is a structural indicator: whether any learner appears in more than one split.
The legacy CSV column ``eoc`` stores |rho| (Pearson edge-weight vs test-outcome correlation;
Eq. rho-edge-outcome). Older exports may hold sqrt(2+2rho^2); migrate with
``python -m scripts.regenerate_leakage_metrics`` or the legacy converter in
``generate_phase_c_tables._legacy_eoc_to_rho_abs``.
TBVR measures within-train temporal leakage: fraction of post-cutoff train
interactions that participate in prerequisite transitions supporting retained edges,
using an inner cutoff aligned to the protocol train ratio (default 0.7).

Cross-split sanity (held-out rows never fed to builders) is assumed by design;
TBVR is strictly positive only when prerequisite inference uses transitions that
touch the ``future'' tail of each learner's train timeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user consecutive KC pairs (src_kc, dst_kc) with dst row index."""
    parts = []
    for user_id, part in df.groupby("user_id", sort=False):
        part = part.sort_values("timestamp").reset_index(drop=True)
        if len(part) < 2:
            continue
        src = part["kc_id"].iloc[:-1].to_numpy()
        dst = part["kc_id"].iloc[1:].to_numpy()
        mask = src != dst
        if not mask.any():
            continue
        parts.append(
            pd.DataFrame({
                "user_id": user_id,
                "src_kc": src[mask],
                "dst_kc": dst[mask],
            })
        )
    if not parts:
        return pd.DataFrame(columns=["user_id", "src_kc", "dst_kc"])
    return pd.concat(parts, ignore_index=True)


def compute_ecr_flag(splits: dict[str, pd.DataFrame]) -> float:
    """Return 1.0 iff any learner id appears in more than one of train/valid/test; else 0.0."""
    sets: list[set[int]] = []
    for key in ("train", "valid", "test"):
        df = splits.get(key)
        if df is None or df.empty or "user_id" not in df.columns:
            sets.append(set())
            continue
        sets.append(set(df["user_id"].astype(int)))
    utr, uva, ute = sets
    if (utr & uva) or (utr & ute) or (uva & ute):
        return 1.0
    return 0.0


def _held_item_sets_by_kc(held: pd.DataFrame) -> dict[int, set[int]]:
    d: dict[int, set[int]] = {}
    for row in held.itertuples(index=False):
        kc = int(getattr(row, "kc_id"))
        item = int(getattr(row, "item_id"))
        d.setdefault(kc, set()).add(item)
    return d


def compute_ecr_overlap(
    pre_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    held_df: pd.DataFrame,
    q_train: pd.DataFrame,
) -> float:
    """Held-out pattern overlap rate over prerequisite ∪ similarity edges (edge-level diagnostic)."""
    if held_df.empty:
        return 0.0
    pairs = _transition_pairs(held_df)
    held_transition_support: set[tuple[int, int]] = set()
    if not pairs.empty:
        held_transition_support = set(zip(pairs["src_kc"].astype(int), pairs["dst_kc"].astype(int), strict=True))

    hi_items = _held_item_sets_by_kc(held_df)
    train_items = (
        q_train.groupby("kc_id")["item_id"].apply(lambda s: set(s.astype(int))).to_dict()
        if not q_train.empty
        else {}
    )

    contaminated = 0
    n_edges = 0
    for df in (pre_df, sim_df):
        if df is None or df.empty:
            continue
        for row in df.itertuples(index=False):
            n_edges += 1
            s = int(getattr(row, "src_kc"))
            d = int(getattr(row, "dst_kc"))
            chi = False
            if (s, d) in held_transition_support:
                chi = True
            else:
                ts = train_items.get(s, set())
                td = train_items.get(d, set())
                hi_s = hi_items.get(s, set())
                hi_d = hi_items.get(d, set())
                if hi_s and hi_d and (hi_s & hi_d):
                    chi = True
            if chi:
                contaminated += 1

    if n_edges == 0:
        return 0.0
    return contaminated / n_edges


def _pair_count_map(df: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Directed KC transition counts (src, dst) -> multiplicity."""
    pairs = _transition_pairs(df)
    if pairs.empty:
        return {}
    grouped = pairs.groupby(["src_kc", "dst_kc"]).size()
    return {(int(s), int(d)): int(c) for (s, d), c in grouped.items()}


def compute_edge_heldout_shares(
    pre_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    train_df: pd.DataFrame,
    held_df: pd.DataFrame,
    q_train: pd.DataFrame,
) -> list[float]:
    """Per-edge held-out evidence share: n_held / (n_train + n_held).

    Uses held-out transition counts when available; if overlap is item-based only
    (same logic as ``compute_ecr_overlap``), assigns n_held=1.
    """
    if held_df.empty:
        return []
    train_counts = _pair_count_map(train_df)
    held_counts = _pair_count_map(held_df)
    hi_items = _held_item_sets_by_kc(held_df)
    train_items = (
        q_train.groupby("kc_id")["item_id"].apply(lambda s: set(s.astype(int))).to_dict()
        if not q_train.empty
        else {}
    )

    shares: list[float] = []
    for df in (pre_df, sim_df):
        if df is None or df.empty:
            continue
        for row in df.itertuples(index=False):
            s = int(getattr(row, "src_kc"))
            d = int(getattr(row, "dst_kc"))
            n_train = int(train_counts.get((s, d), 0))
            n_held = int(held_counts.get((s, d), 0))
            if n_held == 0:
                ts = train_items.get(s, set())
                td = train_items.get(d, set())
                hi_s = hi_items.get(s, set())
                hi_d = hi_items.get(d, set())
                if hi_s and hi_d and (hi_s & hi_d):
                    n_held = 1
            denom = n_train + n_held
            shares.append(float(n_held / denom) if denom > 0 else 0.0)
    return shares


def summarize_edge_heldout_shares(shares: list[float]) -> dict[str, float | int]:
    """Distribution summary for edge-level held-out throughput shares."""
    if not shares:
        return {
            "n_edges": 0,
            "frac_share_gt_50": 0.0,
            "share_median": 0.0,
            "share_p90": 0.0,
        }
    arr = np.asarray(shares, dtype=np.float64)
    return {
        "n_edges": int(len(arr)),
        "frac_share_gt_50": float(np.mean(arr > 0.5)),
        "share_median": float(np.median(arr)),
        "share_p90": float(np.quantile(arr, 0.9)),
    }


def compute_rho_edge_outcome(
    pre_df: pd.DataFrame, sim_df: pd.DataFrame, test_df: pd.DataFrame
) -> float:
    """Pearson |rho| between edge weights and test-fold endpoint outcome summaries (Eq. rho-edge-outcome)."""
    if test_df.empty:
        return 0.0
    stats = test_df.groupby("kc_id")["correct"].mean()
    w_list: list[float] = []
    y_list: list[float] = []
    for df in (pre_df, sim_df):
        if df is None or df.empty or "weight" not in df.columns:
            continue
        for row in df.itertuples(index=False):
            s = int(getattr(row, "src_kc"))
            d = int(getattr(row, "dst_kc"))
            if s not in stats.index or d not in stats.index:
                continue
            yi = (float(stats.loc[s]) + float(stats.loc[d])) / 2.0
            w_list.append(float(getattr(row, "weight")))
            y_list.append(yi)
    if len(w_list) < 2:
        return 0.0
    w_arr = np.asarray(w_list, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.float64)
    if np.std(w_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    w_c = w_arr - w_arr.mean()
    y_c = y_arr - y_arr.mean()
    denom = float(np.sqrt(np.sum(w_c**2) * np.sum(y_c**2)))
    if denom <= 0:
        return 0.0
    rho = float(np.sum(w_c * y_c) / denom)
    rho = max(-1.0, min(1.0, rho))
    return float(abs(rho))


def compute_eoc(pre_df: pd.DataFrame, sim_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Backward-compatible alias: stores |rho| in the legacy ``eoc`` CSV column."""
    return compute_rho_edge_outcome(pre_df, sim_df, test_df)


def compute_tbvr(train_df: pd.DataFrame, pre_df: pd.DataFrame, train_ratio: float = 0.7) -> float:
    """Temporal boundary violations within the train fold (Eq.~\\ref{eq:tbvr} operationalisation)."""
    if train_df.empty or pre_df.empty:
        return 0.0
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie in (0, 1)")
    retained = set(zip(pre_df["src_kc"].astype(int), pre_df["dst_kc"].astype(int), strict=True))

    post_token_keys: set[tuple[int, int]] = set()
    violating_keys: set[tuple[int, int]] = set()

    for user_id, part in train_df.groupby("user_id", sort=False):
        ordered = part.sort_values("timestamp").reset_index(drop=True)
        n = len(ordered)
        cutoff_idx = int(np.floor(train_ratio * n))
        if cutoff_idx >= n:
            continue
        post_positions = range(cutoff_idx, n)
        for j in post_positions:
            post_token_keys.add((int(user_id), int(j)))

        kcs = ordered["kc_id"].to_numpy()
        for i in range(n - 1):
            s_kc, d_kc = int(kcs[i]), int(kcs[i + 1])
            if s_kc == d_kc:
                continue
            if (s_kc, d_kc) not in retained:
                continue
            if i >= cutoff_idx or (i + 1) >= cutoff_idx:
                if i >= cutoff_idx:
                    violating_keys.add((int(user_id), i))
                if (i + 1) >= cutoff_idx:
                    violating_keys.add((int(user_id), i + 1))

    if not post_token_keys:
        return 0.0
    bad = len(violating_keys & post_token_keys)
    return bad / len(post_token_keys)


def compute_leakage_row(
    *,
    dataset: str,
    fold: int,
    splits: dict[str, pd.DataFrame],
    pre_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    q_train: pd.DataFrame,
    train_ratio: float = 0.7,
) -> dict[str, float | int | str]:
    """One CSV row: dataset, fold, ecr_flag, ecr_overlap, eoc, tbvr."""
    train_df = splits["train"]
    held_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    test_df = splits["test"]
    ecr_flag = compute_ecr_flag(splits)
    ecr_overlap = compute_ecr_overlap(pre_df, sim_df, held_df, q_train)
    eoc = compute_eoc(pre_df, sim_df, test_df)
    tbvr = compute_tbvr(train_df, pre_df, train_ratio=train_ratio)
    logger.info(
        "Leakage metrics dataset=%s fold=%s ECR_flag=%.1f ECR_overlap=%.6f |rho|=%.6f TBVR=%.6f",
        dataset,
        fold,
        ecr_flag,
        ecr_overlap,
        eoc,
        tbvr,
    )
    return {
        "dataset": dataset,
        "fold": fold,
        "ecr_flag": ecr_flag,
        "ecr_overlap": ecr_overlap,
        "eoc": eoc,
        "tbvr": tbvr,
    }


# Back-compat alias
compute_ecr = compute_ecr_overlap


def merge_leakage_metrics_csv(rows: Iterable[dict[str, float | int | str]], dataset: str, path: str | Path = "results/tables/leakage_metrics.csv") -> None:
    """Replace rows for ``dataset`` and append new fold metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty and "dataset" in current.columns:
        current = current[current["dataset"].astype(str) != dataset]
    merged = pd.concat([current, pd.DataFrame(list(rows))], ignore_index=True)
    if not merged.empty:
        merged = merged.sort_values(["dataset", "fold"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
