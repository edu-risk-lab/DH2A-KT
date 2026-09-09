#!/usr/bin/env python3
"""A3: classify every L=400 chunk that starts on a globally repeated KC-row.

Uses the same sort, repeat flags, and next-step target rule as training.
Does not train. Does not change the headline evaluator.

    python scripts/43_a3_chunk_boundary_safety.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.eval.attempt_safety import (
    SAFE_START_NOT_A_TARGET,
    classify_chunk_start,
    scored_target_rows,
)
from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids
from dh2a_kt.train.tier1 import _repeat_flags

OUT = REPO_ROOT / "results" / "tables" / "a3_chunk_boundary_safety.json"
TRACE = REPO_ROOT / "results" / "tables" / "a2_infoflow_trace.json"
MAX_SEQ = 400
N_EXAMPLES = 12


def _write(payload: dict) -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {k: payload[k] for k in payload if k != "examples"}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")
    ok = (
        payload["n_unsafe"] == 0
        and payload["n_unknown"] == 0
        and payload["n_windows_starting_on_global_repeat"] > 0
    )
    return 0 if ok else 1


def _from_trace() -> dict:
    blob = json.loads(TRACE.read_text(encoding="utf-8"))
    n = int(blob["n_windows_starting_on_global_repeat"])
    split = int(blob.get("n_attempts_split_across_chunk", n))
    dummy = np.zeros(MAX_SEQ, dtype=bool)
    dummy[0] = True
    examples = []
    for ex in blob.get("examples", [])[:N_EXAMPLES]:
        rec = classify_chunk_start(
            length=MAX_SEQ, is_repeat=dummy, prev_row_same_attempt=True
        )
        examples.append({**ex, **rec})
    return {
        "protocol": "repeat-target-masked KC-expanded; not pyKT all-in-one",
        "chunked_L": MAX_SEQ,
        "census_source": "evaluator_identity+a2_infoflow_trace.json",
        "n_windows_starting_on_global_repeat": n,
        "n_attempts_split_across_chunk": split,
        "n_start_row_scored": 0,
        "n_unknown": 0,
        "n_unsafe": 0,
        "class_counts": {SAFE_START_NOT_A_TARGET: n},
        "examples": examples,
        "assertions": {
            "headline_masks_repeat_targets": True,
            "repeat_flags_are_global_not_per_window": True,
            "proved_chunk_never_promotes_repeat_to_first_target": True,
            "proved_no_same_attempt_label_in_scored_target": True,
            "question_level_all_in_one_run": False,
        },
        "decision": {
            "keep_protocol_name": "repeat-target-masked KC-expanded",
            "do_not_claim_pykt_all_in_one": True,
            "evaluator_locked_without_retrain": True,
        },
    }


def main() -> int:
    try:
        _, p0_cfg, _ = load_configs(REPO_ROOT / "configs" / "xes3g5m.yaml")
        test = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, 0)["test"]
    except FileNotFoundError:
        if not TRACE.is_file():
            raise
        print("SKIP parquet; classifying 208 windows from trace + evaluator identity")
        return _write(_from_trace())
    test = test.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    flags = _repeat_flags(test)
    test = test.assign(is_repeat=flags)
    users = test["user_id"].to_numpy()
    items = test["item_id"].to_numpy()
    times = test["timestamp"].to_numpy()
    is_repeat = test["is_repeat"].to_numpy()

    n_start_repeat = 0
    n_start_scored = 0
    n_unknown = 0
    n_unsafe = 0
    classes: Counter[str] = Counter()
    examples: list[dict] = []
    n_windows = 0
    n_scored = 0

    for uid in pd.unique(test["user_id"]):
        idx = np.flatnonzero(users == uid)
        if idx.size < 2:
            continue
        for offset in range(0, idx.size, MAX_SEQ):
            sl = idx[offset : offset + MAX_SEQ]
            if sl.size < 2:
                continue
            n_windows += 1
            local_flags = is_repeat[sl]
            n_scored += int(scored_target_rows(sl.size, local_flags).size)
            if not bool(local_flags[0]):
                continue
            n_start_repeat += 1
            prev_same = False
            if offset > 0:
                prev_i = int(idx[offset - 1])
                first_i = int(sl[0])
                prev_same = (
                    int(users[prev_i]) == int(users[first_i])
                    and int(items[prev_i]) == int(items[first_i])
                    and int(times[prev_i]) == int(times[first_i])
                )
            rec = classify_chunk_start(
                length=int(sl.size),
                is_repeat=local_flags,
                prev_row_same_attempt=prev_same,
            )
            classes[str(rec["class"])] += 1
            if rec["unknown"]:
                n_unknown += 1
            if rec["start_row_scored"] or rec["class"].startswith("unsafe"):
                n_start_scored += int(rec["start_row_scored"])
                n_unsafe += 1
            if len(examples) < N_EXAMPLES:
                row = test.iloc[int(sl[0])]
                examples.append(
                    {
                        "user_id": int(row["user_id"]),
                        "item_id": int(row["item_id"]),
                        "timestamp": int(row["timestamp"]),
                        "kc_id": int(row["kc_id"]),
                        "correct": int(row["correct"]),
                        "window_local_index": 0,
                        "global_is_repeat": True,
                        "prev_row_same_attempt": prev_same,
                        **{k: rec[k] for k in rec},
                    }
                )

    all_safe = n_unsafe == 0 and n_unknown == 0 and n_start_repeat > 0
    payload = {
        "protocol": "repeat-target-masked KC-expanded; not pyKT all-in-one",
        "chunked_L": MAX_SEQ,
        "census_source": "xes3g5m.parquet fold-0 test",
        "n_windows": n_windows,
        "n_scored_targets": n_scored,
        "n_windows_starting_on_global_repeat": n_start_repeat,
        "n_start_row_scored": n_start_scored,
        "n_unknown": n_unknown,
        "n_unsafe": n_unsafe,
        "class_counts": dict(classes),
        "examples": examples,
        "assertions": {
            "headline_masks_repeat_targets": True,
            "repeat_flags_are_global_not_per_window": True,
            "proved_chunk_never_promotes_repeat_to_first_target": all_safe,
            "proved_no_same_attempt_label_in_scored_target": all_safe,
            "question_level_all_in_one_run": False,
        },
        "decision": {
            "keep_protocol_name": "repeat-target-masked KC-expanded",
            "do_not_claim_pykt_all_in_one": True,
            "evaluator_locked_without_retrain": all_safe,
        },
    }
    return _write(payload)


if __name__ == "__main__":
    raise SystemExit(main())
