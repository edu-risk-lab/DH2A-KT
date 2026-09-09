"""Attempt-safe scoring rules for the repeat-target-masked KC-expanded protocol.

``logits[:, t]`` predicts row ``t + 1``. The first row of a chunk is therefore
never a target. Repeat flags are computed on the full learner log (not
recomputed inside a window), so a globally repeated KC-row that lands at
local index 0 stays a repeat and cannot be promoted to a first-of-attempt
target.

This is not pyKT question-level all-in-one fusion: later KCs of the same
attempt are masked, not pooled into one ``p_a``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SAFE_START_NOT_A_TARGET = "safe_start_not_a_target"
SAFE_SCORED_TARGET_NEW_ATTEMPT = "safe_scored_target_is_new_attempt"


def scored_target_rows(length: int, is_repeat: np.ndarray) -> np.ndarray:
    """Local row indices that ``next_step_mask`` would score as targets.

    Target of prediction ``t`` is row ``t + 1``, for ``t = 0 .. length-2``,
    excluding rows whose global ``is_repeat`` flag is True. Row 0 is never
    a target.
    """
    if length < 2:
        return np.zeros(0, dtype=np.int64)
    flags = np.asarray(is_repeat[:length], dtype=bool)
    rows = np.arange(1, length, dtype=np.int64)
    return rows[~flags[1:length]]


def start_row_is_scored_target(length: int, is_repeat: np.ndarray) -> bool:
    return bool(np.any(scored_target_rows(length, is_repeat) == 0))


def scored_targets_are_new_attempts(length: int, is_repeat: np.ndarray) -> bool:
    """Every scored target row is not a continuation of the previous row."""
    flags = np.asarray(is_repeat[:length], dtype=bool)
    for row in scored_target_rows(length, is_repeat):
        if flags[int(row)]:
            return False
    return True


def classify_chunk_start(
    *,
    length: int,
    is_repeat: np.ndarray,
    prev_row_same_attempt: bool | None,
) -> dict[str, Any]:
    """Classify one window that begins on a globally repeated KC-row."""
    flags = np.asarray(is_repeat[:length], dtype=bool)
    if length < 1 or not bool(flags[0]):
        return {
            "class": "not_a_chunk_start_repeat",
            "unknown": False,
            "start_row_scored": False,
        }
    start_scored = start_row_is_scored_target(length, flags)
    new_attempts_ok = scored_targets_are_new_attempts(length, flags)
    first_scored = scored_target_rows(length, flags)
    first_scored_row = int(first_scored[0]) if first_scored.size else None
    if start_scored:
        kind = "unsafe_start_row_scored"
        unknown = False
    elif not new_attempts_ok:
        kind = "unsafe_scored_continuation"
        unknown = False
    else:
        kind = SAFE_START_NOT_A_TARGET
        unknown = False
    return {
        "class": kind,
        "unknown": unknown,
        "start_row_scored": start_scored,
        "scored_targets_are_new_attempts": new_attempts_ok,
        "prev_row_same_attempt": prev_row_same_attempt,
        "n_scored_in_window": int(first_scored.size),
        "first_scored_local_row": first_scored_row,
        "same_attempt_label_read_for_scored_target": False,
    }
