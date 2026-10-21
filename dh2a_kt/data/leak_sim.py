"""Synthetic leakage classes for the round-2 simulation (docs/preregistration.md, Section 6).

Two classes, each with a leaky scalar and a clean twin, built per interaction
row. Neither is evidence about a public corpus; they test whether the audit
registers the leak and how much AUC it adds.

cross-modal
    Mean correctness of item ``q`` over the full log (every learner, every
    split) with start time ``> t``, shrunk toward the fit-split mean with
    ``SHRINK`` pseudo-counts. The clean twin pools fit-split learners with
    start time ``< t``. Injected on the query side.
intervention-timing
    A hint flag at row ``t`` drawn with knowledge of the learner's next
    response ``y_{t+1}``. The clean twin is drawn independently at the
    leaky flag's fit-split rate. Injected on the recurrent input at ``t``.

Rows are ordered as ``UserSequenceDataset`` orders them: user, timestamp,
item, with ties kept in input order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHRINK = 10.0
HINT_SEED = 20261008
P_HINT_IF_WRONG = 0.10
P_HINT_IF_RIGHT = 0.03
FRACTIONS = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0)

# arm -> (class, leaky, injection mode of DH2KTConfig.aux_scalar_mode)
ARMS = {
    "crossmodal-leak": ("crossmodal", True, "query"),
    "crossmodal-clean": ("crossmodal", False, "query"),
    "hint-leak": ("hint", True, "lstm"),
    "hint-clean": ("hint", False, "lstm"),
}


def _within_group_cumsum(values: np.ndarray, group_start: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum that restarts at every group start."""
    total = np.cumsum(values)
    before = (total - values)[group_start]
    return total - before[group_idx]


def crossmodal_parts(df: pd.DataFrame, fit_users: set) -> pd.DataFrame:
    """Leaky and clean item scalars plus their support counts, aligned to ``df``.

    ``n_leak_support`` counts rows of the item after ``t`` (all post-cutoff);
    ``n_clean_support`` counts fit-split rows of the item before ``t``.
    """
    items = df["item_id"].to_numpy()
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    y = df["correct"].to_numpy(dtype=np.float64)
    fit = df["user_id"].isin(fit_users).to_numpy()
    if not fit.any():
        raise ValueError("crossmodal_parts: no fit-split rows")
    global_mean = float(y[fit].mean())

    order = np.lexsort((ts, items))
    it, t, yy, ff = items[order], ts[order], y[order], fit[order].astype(np.float64)
    n = len(it)
    new_item = np.r_[True, it[1:] != it[:-1]]
    new_run = np.r_[True, (it[1:] != it[:-1]) | (t[1:] != t[:-1])]
    item_start = np.flatnonzero(new_item)
    item_idx = np.cumsum(new_item) - 1
    item_end = np.r_[item_start[1:] - 1, n - 1]
    run_start = np.flatnonzero(new_run)
    run_idx = np.cumsum(new_run) - 1
    run_end = np.r_[run_start[1:] - 1, n - 1]

    ones = np.ones(n)
    cnt = _within_group_cumsum(ones, item_start, item_idx)
    sm = _within_group_cumsum(yy, item_start, item_idx)
    cnt_w = _within_group_cumsum(ff, item_start, item_idx)
    sm_w = _within_group_cumsum(ff * yy, item_start, item_idx)

    # Strictly after t: item total minus everything at or before t.
    n_post = cnt[item_end][item_idx] - cnt[run_end][run_idx]
    s_post = sm[item_end][item_idx] - sm[run_end][run_idx]
    # Strictly before t, fit-split learners only.
    n_past = (cnt_w - ff)[run_start][run_idx]
    s_past = (sm_w - ff * yy)[run_start][run_idx]

    out = np.empty((n, 4))
    out[order, 0] = (s_post + SHRINK * global_mean) / (n_post + SHRINK)
    out[order, 1] = (s_past + SHRINK * global_mean) / (n_past + SHRINK)
    out[order, 2] = n_post
    out[order, 3] = n_past
    return pd.DataFrame(
        out,
        columns=["leak_value", "clean_value", "n_leak_support", "n_clean_support"],
        index=df.index,
    )


def hint_parts(df: pd.DataFrame, fit_users: set) -> pd.DataFrame:
    """Leaky and clean hint flags, aligned to ``df``.

    The leaky flag at row ``t`` depends on ``y_{t+1}`` of the same learner;
    a learner's last row has no next response and gets no leaky hint.
    """
    users = df["user_id"].to_numpy()
    order = np.lexsort((df["item_id"].to_numpy(), df["timestamp"].to_numpy(dtype=np.int64), users))
    u = users[order]
    y = df["correct"].to_numpy(dtype=np.float64)[order]
    n = len(u)
    has_next = np.r_[u[1:] == u[:-1], False]
    y_next = np.r_[y[1:], 0.0]
    p_hint = np.where(y_next < 0.5, P_HINT_IF_WRONG, P_HINT_IF_RIGHT)
    leak = (np.random.default_rng(HINT_SEED).random(n) < p_hint) & has_next
    fit = df["user_id"].isin(fit_users).to_numpy()[order]
    rate = float(leak[fit].mean())
    clean = np.random.default_rng(HINT_SEED + 1).random(n) < rate

    out = np.empty((n, 2))
    out[order, 0] = leak
    out[order, 1] = clean
    frame = pd.DataFrame(out, columns=["leak_value", "clean_value"], index=df.index)
    frame.attrs["fit_rate"] = rate
    return frame


def build_parts(kind: str, df: pd.DataFrame, fit_users: set) -> pd.DataFrame:
    if kind == "crossmodal":
        return crossmodal_parts(df, fit_users)
    if kind == "hint":
        return hint_parts(df, fit_users)
    raise ValueError(kind)


def attach_leak_scalar(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    arm: str,
    fit_users: set,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Add an ``aux_scalar`` column for ``arm``; returns frames and injection mode."""
    kind, leaky, mode = ARMS[arm]
    both = pd.concat([train_df, eval_df], ignore_index=True)
    parts = build_parts(kind, both, fit_users)
    values = parts["leak_value" if leaky else "clean_value"].to_numpy(dtype=np.float32)
    n_train = len(train_df)
    return (
        train_df.assign(aux_scalar=values[:n_train]),
        eval_df.assign(aux_scalar=values[n_train:]),
        mode,
    )


def graded_crossmodal_share(parts: pd.DataFrame, rows: np.ndarray, fraction: float, seed: int) -> float:
    """Support-weighted post-cutoff share when ``fraction`` of ``rows`` are built leaky."""
    sub = parts.iloc[rows]
    chosen = np.random.default_rng(seed).random(len(sub)) < fraction
    post = sub["n_leak_support"].to_numpy()[chosen].sum()
    total = post + sub["n_clean_support"].to_numpy()[~chosen].sum()
    return float(post / total) if total > 0 else 0.0


def graded_hint_rate(parts: pd.DataFrame, rows: np.ndarray, fraction: float, seed: int) -> float:
    """Share of hint rows selected after the outcome when ``fraction`` of ``rows`` are leaky."""
    sub = parts.iloc[rows]
    chosen = np.random.default_rng(seed).random(len(sub)) < fraction
    leak = sub["leak_value"].to_numpy() > 0.5
    clean = sub["clean_value"].to_numpy() > 0.5
    flagged = np.where(chosen, leak, clean)
    n_flagged = int(flagged.sum())
    return float((flagged & chosen).sum() / n_flagged) if n_flagged else 0.0
