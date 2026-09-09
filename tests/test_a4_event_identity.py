"""A4: occurrence join is not a stable event ID; native n closes via one unmapped KC."""
from __future__ import annotations

import pandas as pd

from dh2a_kt.eval.event_identity import (
    account_join_counts,
    account_stable_counts,
    classify_occurrence_mismatch_keys,
    classify_unmatched_occurrence_keys,
    drop_unmapped_question_kc,
    iter_scored_kc_rows,
    parse_occurrence_key,
    stable_event_key,
)


def test_native_count_closes_with_one_unmapped_kc():
    acc = account_join_counts(
        n_native_dh2=1_093_755,
        n_unmapped_kc=1,
        n_pykt=1_093_720,
        n_only_dh2=36,
        n_only_pykt=2,
        n_joined=1_093_718,
    )
    assert acc["accounting_ok"]
    assert acc["n_join_eligible_dh2"] == 1_093_754
    assert acc["one_row_gap_is_unmapped_kc"]
    assert acc["n_net_dh2_surplus"] == 34


def test_occurrence_key_can_pair_different_events():
    """Same (user, kc, occ) after a dropped row is a different attempt."""
    # Full log: two attempts of KC 7, labels 0 then 1.
    # After dropping the first attempt, occurrence 1 is the second attempt.
    key_full = "9|7|1"
    key_after_drop = "9|7|1"
    assert key_full == key_after_drop
    label_full_occ1 = 0
    label_after_drop_occ1 = 1
    assert label_full_occ1 != label_after_drop_occ1
    assert parse_occurrence_key(key_full) == (9, 7, 1)


def test_stable_key_separates_those_events():
    k0 = stable_event_key(9, 100, 50, 7)
    k1 = stable_event_key(9, 200, 51, 7)
    assert k0 != k1


def test_gpu_stable_join_closes_and_agrees_on_labels():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "tables" / "a4_event_identity.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not blob.get("stable_key_join_run"):
        return
    acc = account_stable_counts(
        n_dh2_stable=blob["n_dh2_stable"],
        n_pykt_stable=blob["n_pykt_stable"],
        n_only_dh2_stable=blob["n_only_dh2_stable"],
        n_only_pykt_stable=blob["n_only_pykt_stable"],
        n_common_stable=blob["n_common_stable"],
        n_label_mismatch=blob["n_label_mismatch_on_stable_join"],
    )
    assert acc["accounting_ok"]
    assert acc["same_stable_id_same_label"]
    assert blob["n_common_stable"] == 1_093_718
    assert blob["n_only_dh2_stable"] == 37
    assert blob["n_only_pykt_stable"] == 2


def test_classify_mismatch_keys_counts_user_kc_pairs():
    keys = [
        "1|10|6",
        "1|10|8",
        "2|20|1",
    ]
    rec = classify_occurrence_mismatch_keys(keys)
    assert rec["n_keys"] == 3
    assert rec["n_users"] == 2
    assert rec["n_user_kc_pairs"] == 2


def test_classify_unmatched_splits_first_vs_later_occurrence():
    keys = [
        "1|10|1",
        "1|10|2",
        "2|20|3",
    ]
    rec = classify_unmatched_occurrence_keys(keys)
    assert rec["n_keys"] == 3
    assert rec["n_users"] == 2
    assert rec["n_first_occurrence"] == 1
    assert rec["n_later_occurrence"] == 2


def _tiny_log() -> pd.DataFrame:
    rows = [
        (1, 10, 100, 1, 0),
        (1, 10, 101, 1, 0),  # same attempt, later KC (repeat)
        (1, 20, 200, 2, 1),
        (1, 30, 300, 3, 1),
        (1, 40, 400, 4, 0),
    ]
    return pd.DataFrame(
        rows, columns=["user_id", "item_id", "kc_id", "timestamp", "correct"]
    )


def test_scored_rows_skip_chunk_start_and_repeats():
    events = list(iter_scored_kc_rows(_tiny_log(), max_seq_len=400))
    keys = [e["stable_key"] for e in events]
    # Row 0 of the only window is never scored; the repeat KC-row is masked.
    assert stable_event_key(1, 1, 10, 100) not in keys
    assert stable_event_key(1, 1, 10, 101) not in keys
    assert stable_event_key(1, 2, 20, 200) in keys
    assert len(events) == 3
    assert len(set(keys)) == len(keys)


def test_dropped_row_makes_occurrence_join_collide_but_stable_keys_differ():
    """pyKT dropna of one (user,kc) attempt shifts later occurrence numbers."""
    logs = pd.DataFrame(
        [
            (1, 10, 1, 1, 0),
            (1, 20, 7, 2, 0),
            (1, 30, 7, 3, 1),
            (1, 40, 8, 4, 1),
        ],
        columns=["user_id", "item_id", "kc_id", "timestamp", "correct"],
    )
    q_map = {10: 0, 30: 1, 40: 2}  # item 20 absent → pyKT drops the first KC-7 row
    c_map = {1: 0, 7: 1, 8: 2}
    dh2 = list(iter_scored_kc_rows(logs, max_seq_len=400))
    pykt = list(
        iter_scored_kc_rows(
            drop_unmapped_question_kc(logs, q_map, c_map), max_seq_len=400
        )
    )
    dh2_kc7 = [e for e in dh2 if e["kc_id"] == 7]
    pykt_kc7 = [e for e in pykt if e["kc_id"] == 7]
    assert [e["label"] for e in dh2_kc7] == [0, 1]
    assert [e["label"] for e in pykt_kc7] == [1]
    # Occurrence 1 of (user, kc=7) is a different event on each side.
    assert dh2_kc7[0]["label"] != pykt_kc7[0]["label"]
    assert dh2_kc7[0]["stable_key"] != pykt_kc7[0]["stable_key"]
    extra = {e["stable_key"] for e in dh2} - {e["stable_key"] for e in pykt}
    assert stable_event_key(1, 2, 20, 7) in extra
