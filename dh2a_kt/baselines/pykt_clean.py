"""Export pyKT sequence CSVs with an explicit window rule and repeat masking.

P0's ``src/pykt_export.py`` hard-codes two choices that make its baseline numbers
hard to compare against DH2-KT:

* it keeps ``q_list[-max_seq_len:]``, so only each learner's final window is
  scored, while DH2-KT v4 tiles the whole log;
* it writes ``selectmasks`` as all ones, so the repeat rows that pyKT's KC-level
  export emits for multi-concept questions are scored even though their label is
  the answer the model was just given.

Both choices live entirely in the exported CSV. pyKT derives its select mask as
``selectmasks[:, 1:] != -1`` and routes *both* the training loss and the AUC
through that one mask, so writing ``-1`` at a position drops it everywhere at
once, for every model, with no change to any model, loss or metric code. This
module therefore reproduces P0's export and adds the two options, leaving the
vendored P0 driver untouched.

A third difference is a latent bug rather than a choice: P0 pads ``selectmasks``
with ``"0"``, which is not pyKT's pad value, so padded positions are scored with
label 0. On xes3g5m every learner logs more than ``max_seq_len`` interactions, so
the tail window never pads and the bug is inert; chunked windows do pad, so this
module pads with ``-1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SORT_COLUMNS = ["user_id", "timestamp", "item_id"]
PAD = "-1"


def windows(n_rows: int, max_seq_len: int, window_mode: str) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` slices of one learner's log.

    ``last`` reproduces P0's export, ``chunked`` matches DH2-KT v4, and ``first``
    matches DH2-KT v2/v3. ``last`` and ``first`` both yield one window per learner
    and therefore the same position count while scoring different rows.
    """
    if n_rows < 2:
        return []
    if window_mode == "last":
        return [(max(0, n_rows - max_seq_len), n_rows)]
    if window_mode == "first":
        return [(0, min(n_rows, max_seq_len))]
    if window_mode == "chunked":
        out = []
        for start in range(0, n_rows, max_seq_len):
            end = min(start + max_seq_len, n_rows)
            if end - start >= 2:
                out.append((start, end))
        return out
    raise ValueError(f"window_mode must be 'last', 'first' or 'chunked', got {window_mode!r}")


def repeat_flags(user_ids: np.ndarray, item_ids: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Rows continuing the previous row's attempt at the same question.

    Mirrors ``dh2a_kt.train.tier1._repeat_flags`` so the baselines and DH2-KT
    agree on which positions the clean protocol removes.
    """
    n = len(user_ids)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags
    flags[1:] = (
        (user_ids[1:] == user_ids[:-1])
        & (item_ids[1:] == item_ids[:-1])
        & (timestamps[1:] == timestamps[:-1])
    )
    return flags


def _build_rows(
    df: pd.DataFrame,
    fold_val: int,
    q_map: dict[int, int],
    c_map: dict[int, int],
    max_seq_len: int,
    *,
    window_mode: str,
    mask_repeats: bool,
) -> tuple[pd.DataFrame, int, int]:
    """Returns the sequence rows plus (scored positions, positions masked as repeats)."""
    columns = ["fold", "uid", "questions", "concepts", "responses", "selectmasks", "timestamps"]
    mapped = df.copy()
    mapped["qi"] = mapped["item_id"].map(q_map)
    mapped["ci"] = mapped["kc_id"].map(c_map)
    mapped = mapped.dropna(subset=["qi", "ci"])
    if mapped.empty:
        return pd.DataFrame(columns=columns), 0, 0
    mapped["qi"] = mapped["qi"].astype(int)
    mapped["ci"] = mapped["ci"].astype(int)
    mapped = mapped[(mapped["qi"] >= 0) & (mapped["ci"] >= 0)]
    if mapped.empty:
        return pd.DataFrame(columns=columns), 0, 0

    mapped = mapped.sort_values(SORT_COLUMNS).reset_index(drop=True)
    # Flags come from the full sorted frame, not per window, so a repeat is
    # recognised by the row it continues even when a window boundary splits them.
    is_repeat = repeat_flags(
        mapped["user_id"].to_numpy(),
        mapped["item_id"].to_numpy(),
        mapped["timestamp"].to_numpy(),
    )

    rows: list[dict] = []
    scored = 0
    masked = 0
    for _uid, group in mapped.groupby("user_id", sort=False):
        idx = group.index.to_numpy()
        q_list = group["qi"].to_numpy()
        c_list = group["ci"].to_numpy()
        r_list = group["correct"].to_numpy().astype(int)
        rep = is_repeat[idx]

        for start, end in windows(len(idx), max_seq_len, window_mode):
            length = end - start
            pad_n = max_seq_len - length
            smasks = ["1"] * length
            if mask_repeats:
                for local, flag in enumerate(rep[start:end]):
                    if flag:
                        smasks[local] = PAD
            # pyKT scores selectmasks[1:], so position 0 is never a target.
            scored += length - 1
            masked += sum(1 for m in smasks[1:] if m == PAD)

            rows.append(
                {
                    "fold": fold_val,
                    "uid": int(_uid),
                    "questions": ",".join(str(x) for x in q_list[start:end]) 
                    + ("," + ",".join([PAD] * pad_n) if pad_n else ""),
                    "concepts": ",".join(str(x) for x in c_list[start:end])
                    + ("," + ",".join([PAD] * pad_n) if pad_n else ""),
                    "responses": ",".join(str(x) for x in r_list[start:end])
                    + ("," + ",".join(["0"] * pad_n) if pad_n else ""),
                    "selectmasks": ",".join(smasks + [PAD] * pad_n),
                    "timestamps": ",".join(str(t) for t in range(length))
                    + ("," + ",".join([PAD] * pad_n) if pad_n else ""),
                }
            )
    return pd.DataFrame(rows, columns=columns), scored, masked


def export_pykt_sequences(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    q_map: dict[int, int],
    c_map: dict[int, int],
    out_dir: Path,
    max_seq_len: int,
    window_mode: str,
    mask_repeats: bool,
) -> dict:
    """Write ``train_valid_sequences.csv`` (fold 0/1) and ``test_sequences.csv`` (fold -1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # pyKT caches a parsed .pkl next to the CSV; a stale one would silently
    # serve the previous protocol's masks.
    for stale in out_dir.glob("*.pkl"):
        stale.unlink()

    kwargs = dict(window_mode=window_mode, mask_repeats=mask_repeats)
    tr, _, _ = _build_rows(train_df, 0, q_map, c_map, max_seq_len, **kwargs)
    va, _, _ = _build_rows(valid_df, 1, q_map, c_map, max_seq_len, **kwargs)
    te, te_scored, te_masked = _build_rows(test_df, -1, q_map, c_map, max_seq_len, **kwargs)

    pd.concat([tr, va], ignore_index=True).to_csv(out_dir / "train_valid_sequences.csv", index=False)
    te.to_csv(out_dir / "test_sequences.csv", index=False)

    bipartite = train_df.copy()
    bipartite["qi"] = bipartite["item_id"].map(q_map)
    bipartite["ci"] = bipartite["kc_id"].map(c_map)
    bipartite = bipartite.dropna(subset=["qi", "ci"])[["qi", "ci"]].astype(int).drop_duplicates()
    bipartite.columns = ["question", "concept"]
    bipartite.to_csv(out_dir / "gikt_bipartite.csv", index=False)

    return {
        "num_q": int(max(q_map.values(), default=-1) + 1),
        "num_c": int(max(c_map.values(), default=-1) + 1),
        "test_sequences": int(len(te)),
        "test_positions_before_mask": int(te_scored),
        "test_positions_masked": int(te_masked),
        "test_positions_scored": int(te_scored - te_masked),
    }
