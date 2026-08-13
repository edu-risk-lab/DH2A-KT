"""Fixed GreyKT statistics: E_pre pairs + per-concept training frequency.

Both are computed once from the train fold (not learned). Used by the
white-box branch (``prereq_edge_index``) and the reliability gate
(``concept_train_freq`` / C_B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concept_training_frequency(
    train_df: pd.DataFrame, kc_to_idx: dict[int, int]
) -> np.ndarray:
    """N_c^train per contiguous concept index, length = len(kc_to_idx).

    Concepts that appear in E_pre but never in the training interactions
    get count 0 (C_B = 0): the black-box has no training signal for them.
    """
    n_concepts = len(kc_to_idx)
    freq = np.zeros(n_concepts, dtype=float)
    mapped = train_df["kc_id"].astype(int).map(kc_to_idx).dropna().astype(int)
    counts = mapped.value_counts()
    freq[counts.index.to_numpy()] = counts.to_numpy(dtype=float)
    return freq


def prior_mean_from_train(train_df: pd.DataFrame) -> float:
    """Global train correct-rate; GreyKTConfig.prior_mean must use this."""
    if train_df.empty or "correct" not in train_df.columns:
        return 0.5
    return float(train_df["correct"].mean())


def prereq_edge_index_from_e_pre(e_pre: pd.DataFrame, kc_to_idx: dict[int, int]):
    """Directed (prereq -> target) COO index in contiguous concept ids.

    Drops edges whose endpoints are missing from ``kc_to_idx``.
    Returns a torch LongTensor of shape (2, E).
    """
    import torch

    if e_pre is None or e_pre.empty:
        return torch.zeros(2, 0, dtype=torch.long)
    src = e_pre["src_kc"].map(kc_to_idx)
    dst = e_pre["dst_kc"].map(kc_to_idx)
    ok = src.notna() & dst.notna()
    src_i = src[ok].astype(int).to_numpy()
    dst_i = dst[ok].astype(int).to_numpy()
    if src_i.size == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(np.vstack([src_i, dst_i]), dtype=torch.long)
