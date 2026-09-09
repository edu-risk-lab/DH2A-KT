"""A3: chunk-boundary scoring cannot promote a repeat-row into a target."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dh2a_kt.eval.attempt_safety import (
    SAFE_START_NOT_A_TARGET,
    classify_chunk_start,
    scored_target_rows,
    start_row_is_scored_target,
)
from dh2a_kt.train.tier1 import UserSequenceDataset, _repeat_flags, next_step_mask


def _straddle_logs(*, max_seq_len: int = 4) -> pd.DataFrame:
    """Chunk 0 ends on the first KC of a 3-KC attempt; chunk 1 starts on KC2."""
    rows = []
    t = 0
    for i in range(max_seq_len - 1):
        rows.append((0, 100 + i, 10 + i, t, 1))
        t += 1
    item = 999
    ts = 10_000
    rows.append((0, item, 1, ts, 0))
    rows.append((0, item, 2, ts, 0))
    rows.append((0, item, 3, ts, 0))
    rows.append((0, 200, 4, ts + 1, 1))
    rows.append((0, 201, 5, ts + 2, 1))
    return pd.DataFrame(
        rows, columns=["user_id", "item_id", "kc_id", "timestamp", "correct"]
    )


def test_numpy_target_rows_match_next_step_mask():
    is_repeat = np.array([False, False, True, True, False])
    length = 5
    rows = scored_target_rows(length, is_repeat)
    mask = next_step_mask(
        4,
        torch.tensor([length]),
        is_repeat=torch.tensor(is_repeat).unsqueeze(0),
    )
    torch_rows = (mask[0].nonzero(as_tuple=False).flatten() + 1).tolist()
    assert rows.tolist() == torch_rows
    assert 0 not in rows.tolist()


def test_chunk_start_repeat_is_never_a_scored_target():
    logs = _straddle_logs(max_seq_len=4)
    kc_to_idx = {int(k): i for i, k in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {int(k): i for i, k in enumerate(sorted(logs["item_id"].unique()))}
    ds = UserSequenceDataset(
        logs, kc_to_idx, item_to_idx, max_seq_len=4, window_mode="chunked"
    )
    assert len(ds) == 2
    second = ds[1]
    length = int(second["length"])
    flags = second["is_repeat"][:length].numpy()
    assert bool(flags[0]), "chunk 1 must start on the globally repeated KC-row"
    assert not start_row_is_scored_target(length, flags)
    scored = scored_target_rows(length, flags)
    assert 0 not in scored.tolist()
    census = classify_chunk_start(
        length=length,
        is_repeat=flags,
        prev_row_same_attempt=True,
    )
    assert census["unknown"] is False
    assert census["class"] == SAFE_START_NOT_A_TARGET
    assert census["start_row_scored"] is False
    assert census["same_attempt_label_read_for_scored_target"] is False


def test_dataset_keeps_global_repeat_flags_across_chunks():
    logs = _straddle_logs(max_seq_len=4).sort_values(
        ["user_id", "timestamp", "item_id"]
    ).reset_index(drop=True)
    global_flags = _repeat_flags(logs)
    kc_to_idx = {int(k): i for i, k in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {int(k): i for i, k in enumerate(sorted(logs["item_id"].unique()))}
    ds = UserSequenceDataset(
        logs, kc_to_idx, item_to_idx, max_seq_len=4, window_mode="chunked"
    )
    start, end = ds._starts[1], ds._ends[1]
    window_flags = ds[1]["is_repeat"][: end - start].numpy()
    assert window_flags.tolist() == global_flags[start:end].tolist()
    assert bool(global_flags[start])


def test_every_scored_target_is_a_new_attempt():
    logs = _straddle_logs(max_seq_len=4)
    kc_to_idx = {int(k): i for i, k in enumerate(sorted(logs["kc_id"].unique()))}
    item_to_idx = {int(k): i for i, k in enumerate(sorted(logs["item_id"].unique()))}
    ds = UserSequenceDataset(
        logs, kc_to_idx, item_to_idx, max_seq_len=4, window_mode="chunked"
    )
    for idx in range(len(ds)):
        item = ds[idx]
        length = int(item["length"])
        flags = item["is_repeat"][:length].numpy()
        for target in scored_target_rows(length, flags):
            assert not bool(flags[int(target)])
            mask = next_step_mask(
                max(length - 1, 1),
                item["length"].unsqueeze(0),
                is_repeat=item["is_repeat"].unsqueeze(0),
            )
            pred_t = int(target) - 1
            assert bool(mask[0, pred_t])
