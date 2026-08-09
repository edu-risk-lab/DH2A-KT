"""Unit tests for ID-anchored KC-Jaccard explanation faithfulness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dh2a_kt.eval.explanation_faithfulness import (
    evaluate_explanation_faithfulness,
    jaccard,
    parse_grounded_kcs,
    parse_history_kcs,
    support_set,
    write_faithfulness_report,
)


def test_parse_grounded_trailer_preferred():
    text = "Student needs practice on fractions.\nGroundedKCs: [12, 7, 12]"
    assert parse_grounded_kcs(text) == {12, 7}


def test_parse_grounded_fallback_regex():
    text = "History shows kc 3 correct then kc 9 incorrect."
    assert parse_grounded_kcs(text) == {3, 9}


def test_support_set_includes_neighbors():
    adj = {1: {2}, 2: {1, 5}, 5: {2}}
    s = support_set(
        concept_id=1,
        recent_history_summary="item 9 kc 2 correct",
        adjacency=adj,
    )
    assert s == {1, 2, 5}


def test_jaccard_and_report(tmp_path: Path):
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    records = [
        {
            "student_id": "a",
            "concept_id": "1",
            "recent_history_summary": "item 1 kc 2 correct",
            "diagnostician_explanation": "Ready.\nGroundedKCs: [1, 2]",
            "critic_flagged": False,
            "predicted_correct_prob": 0.7,
        },
        {
            "student_id": "b",
            "concept_id": "9",
            "recent_history_summary": "item 2 kc 8 incorrect",
            "diagnostician_explanation": "Ungrounded.\nGroundedKCs: [99]",
            "critic_flagged": True,
            "predicted_correct_prob": 0.2,
        },
    ]
    e_pre = pd.DataFrame({"src_kc": [1], "dst_kc": [2], "weight": [1.0]})
    report = evaluate_explanation_faithfulness(records, e_pre=e_pre, source_records="toy")
    assert report.n_samples == 2
    assert report.rows[0].kc_jaccard == 1.0
    assert report.rows[1].kc_jaccard == 0.0
    assert report.mean_kc_jaccard == 0.5
    assert report.n_flagged == 1
    out = write_faithfulness_report(report, tmp_path / "faith.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "rows" not in payload
    assert payload["mean_kc_jaccard"] == 0.5


def test_parse_history_kcs():
    assert parse_history_kcs("item 1 kc 2 correct; item 3 kc 4 incorrect") == {2, 4}
