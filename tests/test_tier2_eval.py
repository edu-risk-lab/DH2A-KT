"""M9 Tier-2 pilot evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

from dh2a_kt.tier2.eval import evaluate_tier2_pilot, write_tier2_eval_report


def test_evaluate_tier2_pilot(tmp_path: Path):
    summary = {
        "n_samples": 2,
        "n_flagged": 1,
        "flag_rate": 0.5,
        "n_session_hyperedges": 2,
        "llm_backend": "stub",
        "fold": 0,
        "dataset": "toy",
    }
    records = [
        {
            "student_id": "1",
            "concept_id": "2",
            "predicted_correct_prob": 0.4,
            "critic_flagged": False,
            "diagnostician_explanation": "OK 0.4",
            "critic_reason": "OK",
            "hint_text": "hint a",
        },
        {
            "student_id": "3",
            "concept_id": "4",
            "predicted_correct_prob": 0.8,
            "critic_flagged": True,
            "diagnostician_explanation": "bad",
            "critic_reason": "FLAG drift",
            "hint_text": "hint b",
        },
    ]
    summary_path = tmp_path / "toy_fold0_tier2_pilot_summary.json"
    records_path = tmp_path / "toy_fold0_tier2_pilot.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")

    report = evaluate_tier2_pilot(summary_path, records_path, case_study_n=1)
    assert report.n_samples == 2
    assert report.n_flagged == 1
    assert report.flag_rate == 0.5
    assert report.mean_kc_jaccard is not None
    assert len(report.case_study_flagged) == 1
    assert len(report.case_study_passed) == 1

    out = write_tier2_eval_report(report, tmp_path / "toy_fold0_tier2_eval.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "mean_kc_jaccard" in payload


def test_evaluate_tier2_pilot_with_grounded_trailer(tmp_path: Path):
    summary = {
        "n_samples": 1,
        "n_flagged": 0,
        "flag_rate": 0.0,
        "n_session_hyperedges": 1,
        "llm_backend": "stub",
        "fold": 0,
        "dataset": "toy",
    }
    records = [
        {
            "student_id": "1",
            "concept_id": "2",
            "predicted_correct_prob": 0.4,
            "recent_history_summary": "item 1 kc 2 correct",
            "critic_flagged": False,
            "diagnostician_explanation": "P(correct)=0.4\nGroundedKCs: [2]",
            "critic_reason": "OK",
            "hint_text": "hint a",
        },
    ]
    summary_path = tmp_path / "toy_summary.json"
    records_path = tmp_path / "toy.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    records_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    report = evaluate_tier2_pilot(
        summary_path,
        records_path,
        faithfulness_output=tmp_path / "faith.json",
    )
    assert report.mean_kc_jaccard == 1.0
    assert Path(report.faithfulness_report).exists()