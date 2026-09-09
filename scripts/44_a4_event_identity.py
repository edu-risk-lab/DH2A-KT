#!/usr/bin/env python3
"""A4: account the DH2/pyKT occurrence join; do not treat it as event identity.

Reads ``b6_id_join_summary.json`` and ``b6_unmatched_dh2_rows.csv`` (GPU B6).
Does not drop the 8 label mismatches. Does not train.

If parquet is present, also walks both scored sets with the pre-chunk
``user|timestamp|item|kc`` key. That walk is optional on laptops without data.

    python scripts/44_a4_event_identity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.eval.event_identity import (  # noqa: E402
    account_join_counts,
    classify_unmatched_occurrence_keys,
    drop_unmapped_question_kc,
    iter_scored_kc_rows,
)

OUT = REPO_ROOT / "results" / "tables" / "a4_event_identity.json"
B6 = REPO_ROOT / "results" / "tables" / "b6_id_join_summary.json"
UNMATCHED = REPO_ROOT / "results" / "tables" / "b6_unmatched_dh2_rows.csv"
PARQUET = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed" / "xes3g5m.parquet"


def _from_b6() -> dict:
    blob = json.loads(B6.read_text(encoding="utf-8"))
    keys: list[str] = []
    if UNMATCHED.is_file():
        frame = pd.read_csv(UNMATCHED, dtype={"join_key": str})
        keys = [k for k in frame["join_key"].tolist() if isinstance(k, str) and k]
    acc = account_join_counts(
        n_native_dh2=int(blob["n_dh2_before_c_map"]),
        n_unmapped_kc=int(blob["n_dh2_unmapped_kc"]),
        n_pykt=int(blob["n_pykt"]),
        n_only_dh2=int(blob["n_only_dh2"]),
        n_only_pykt=int(blob["n_only_pykt"]),
        n_joined=int(blob["n_joined"]),
    )
    unmatched = classify_unmatched_occurrence_keys(keys) if keys else {
        "n_keys": int(blob["n_only_dh2"]),
        "n_users": None,
        "n_first_occurrence": None,
        "n_later_occurrence": None,
    }
    n_mismatch = int(blob["label_mismatch_on_join"])
    return {
        "source": "b6_id_join_summary.json",
        "join_key": blob.get("join_key"),
        "stable_key": "user|timestamp|item|kc",
        "accounting": acc,
        "dh2_only_occurrence": unmatched,
        "n_label_mismatch_on_occurrence_join": n_mismatch,
        "eight_mismatches_dropped": False,
        "occurrence_join_is_stable_event_id": False,
        "reason_occurrence_is_not_identity": (
            "Eight joined occurrence keys disagree on the label, so the same "
            "(user, dense-kc, occurrence-in-scored-set) can name two different "
            "KC-rows after the pipelines filter different rows. pyKT export "
            "drops item/KC ids absent from train-fold dense maps before chunking; "
            "DH2 native scoring uses checkpoint maps and keeps those rows. "
            "The one-row native gap (1,093,755 vs 1,093,754) is the hashed KC "
            "that does not map into the train-fold c_map."
        ),
        "headline_model_vs_baseline": "contextual_not_event_paired",
        "stable_key_join_run": False,
        "assertions": {
            "accounting_ok": acc["accounting_ok"],
            "one_row_gap_is_unmapped_kc": acc["one_row_gap_is_unmapped_kc"],
            "do_not_drop_label_mismatches": True,
            "do_not_rebase_headline_n_onto_join": True,
        },
    }


def _stable_join_from_parquet() -> dict:
    from dh2a_kt.hyperedge.p0_inputs import (
        get_fold_splits,
        load_configs,
        load_interactions_with_ids,
    )

    sys.path.insert(0, str(REPO_ROOT / "external" / "p0_leakage_audit" / "src"))
    from pykt_export import build_dense_maps  # noqa: E402

    _, p0_cfg, _ = load_configs(REPO_ROOT / "configs" / "xes3g5m.yaml")
    splits = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, 0)
    train_df = splits["train"]
    test_df = splits["test"]
    q_map, c_map = build_dense_maps(train_df)
    dh2 = list(iter_scored_kc_rows(test_df, max_seq_len=400))
    pykt_frame = drop_unmapped_question_kc(test_df, q_map, c_map)
    pykt = list(iter_scored_kc_rows(pykt_frame, max_seq_len=400))
    dh2_keys = {e["stable_key"]: e["label"] for e in dh2}
    pykt_keys = {e["stable_key"]: e["label"] for e in pykt}
    common = set(dh2_keys) & set(pykt_keys)
    label_mismatch = sum(1 for k in common if dh2_keys[k] != pykt_keys[k])
    return {
        "stable_key_join_run": True,
        "n_dh2_stable": len(dh2),
        "n_pykt_stable": len(pykt),
        "n_common_stable": len(common),
        "n_only_dh2_stable": len(set(dh2_keys) - set(pykt_keys)),
        "n_only_pykt_stable": len(set(pykt_keys) - set(dh2_keys)),
        "n_label_mismatch_on_stable_join": int(label_mismatch),
        "same_stable_id_same_label": label_mismatch == 0,
        "example_only_dh2": sorted(set(dh2_keys) - set(pykt_keys))[:8],
    }


def main() -> int:
    if not B6.is_file():
        print(f"SKIP: missing {B6}", flush=True)
        return 2
    payload = _from_b6()
    if PARQUET.is_file():
        payload.update(_stable_join_from_parquet())
    else:
        payload["parquet_present"] = False
        payload["stable_key_join_note"] = (
            "xes3g5m.parquet not on this machine; stable-key census is the "
            "function + tests. Re-run on GPU to count common pre-chunk keys."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    printable = {k: v for k, v in payload.items() if k != "dh2_only_occurrence"}
    printable["dh2_only_n_users"] = (payload.get("dh2_only_occurrence") or {}).get(
        "n_users"
    )
    printable["dh2_only_n_first_occurrence"] = (
        payload.get("dh2_only_occurrence") or {}
    ).get("n_first_occurrence")
    print(json.dumps(printable, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    ok = (
        payload["assertions"]["accounting_ok"]
        and payload["assertions"]["one_row_gap_is_unmapped_kc"]
        and payload["n_label_mismatch_on_occurrence_join"] == 8
        and payload["eight_mismatches_dropped"] is False
        and payload.get("n_label_mismatch_on_stable_join", 0) == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
