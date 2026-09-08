#!/usr/bin/env python3
"""Hau A2: information-flow trace for multi-KC attempts (no train).

Writes examples plus a chunk-boundary audit: a globally-repeat KC-row that
lands at the start of a new L=400 chunk is *not* a first-of-attempt target
in the full log, but can become window-local history.

This is not a pyKT all-in-one evaluator.

    python scripts/40_a2_infoflow_trace.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids
from dh2a_kt.train.tier1 import _repeat_flags

OUT = REPO_ROOT / "results" / "tables" / "a2_infoflow_trace.json"
MAX_SEQ = 400
N_EXAMPLES = 12


def main() -> int:
    _, p0_cfg, _ = load_configs(REPO_ROOT / "configs" / "xes3g5m.yaml")
    test = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, 0)["test"]
    test = test.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    flags = _repeat_flags(test)
    test = test.assign(is_repeat=flags)
    keys = ["user_id", "item_id", "timestamp"]
    sizes = test.groupby(keys, sort=False).size().rename("n_kc")
    test = test.merge(sizes.reset_index(), on=keys, how="left")
    multi = test[test["n_kc"] >= 2].copy()

    users = test["user_id"].to_numpy()
    chunk_start_on_repeat = 0
    split_attempts = 0
    examples: list[dict] = []
    for uid in pd.unique(test["user_id"]):
        idx = np.flatnonzero(users == uid)
        if idx.size < 2:
            continue
        for offset in range(0, idx.size, MAX_SEQ):
            sl = idx[offset : offset + MAX_SEQ]
            if sl.size < 2:
                continue
            if bool(test.iloc[int(sl[0])]["is_repeat"]):
                chunk_start_on_repeat += 1
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
                            "issue": "chunk_starts_on_global_repeat_row",
                        }
                    )
            # attempt split across this boundary: previous window last row
            # shares attempt with this window first row
            if offset > 0:
                prev = test.iloc[int(idx[offset - 1])]
                first = test.iloc[int(sl[0])]
                if (
                    int(prev["user_id"]) == int(first["user_id"])
                    and int(prev["item_id"]) == int(first["item_id"])
                    and int(prev["timestamp"]) == int(first["timestamp"])
                ):
                    split_attempts += 1

    firsts = test.loc[~test["is_repeat"]]
    payload = {
        "protocol": "repeat-target-masked KC-expanded; not pyKT all-in-one",
        "n_test_rows": int(len(test)),
        "n_repeat_rows": int(test["is_repeat"].sum()),
        "n_multi_kc_rows": int(len(multi)),
        "n_multi_kc_attempts": int(multi.drop_duplicates(keys).shape[0]),
        "n_first_of_attempt": int(len(firsts)),
        "chunked_L": MAX_SEQ,
        "n_windows_starting_on_global_repeat": chunk_start_on_repeat,
        "n_attempts_split_across_chunk": split_attempts,
        "examples": examples,
        "assertions": {
            "headline_masks_repeat_targets": True,
            "proved_no_same_attempt_label_in_query": False,
            "proved_chunk_never_promotes_repeat_to_first_target": split_attempts == 0,
            "question_level_all_in_one_run": False,
        },
        "decision": {
            "keep_protocol_name": "repeat-target-masked KC-expanded",
            "do_not_claim_pykt_all_in_one": True,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "examples"}, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
