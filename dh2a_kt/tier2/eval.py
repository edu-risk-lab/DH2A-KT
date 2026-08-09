"""M9: summarize Tier-2 pilot logs (Critic flag rate, KC-Jaccard, case studies)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from dh2a_kt.eval.explanation_faithfulness import (
    evaluate_explanation_faithfulness,
    write_faithfulness_report,
)


@dataclass
class Tier2EvalReport:
    dataset: str
    fold: int
    llm_backend: str
    n_samples: int
    n_flagged: int
    flag_rate: float
    n_session_hyperedges: int
    mean_predicted_prob: float
    mean_kc_jaccard: float | None
    median_kc_jaccard: float | None
    n_empty_grounded: int | None
    case_study_flagged: list[dict]
    case_study_passed: list[dict]
    source_summary: str
    source_records: str
    faithfulness_report: str | None = None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_tier2_pilot(
    summary_path: Path,
    records_path: Path,
    *,
    case_study_n: int = 3,
    e_pre: pd.DataFrame | None = None,
    faithfulness_output: Path | None = None,
) -> Tier2EvalReport:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    records = _read_jsonl(Path(records_path))
    if not records:
        raise ValueError(f"no records in {records_path}")

    flagged = [r for r in records if r.get("critic_flagged")]
    passed = [r for r in records if not r.get("critic_flagged")]
    probs = [float(r["predicted_correct_prob"]) for r in records if "predicted_correct_prob" in r]
    mean_prob = sum(probs) / len(probs) if probs else float("nan")

    faith = evaluate_explanation_faithfulness(
        records, e_pre=e_pre, source_records=str(records_path)
    )
    faith_path = None
    if faithfulness_output is not None:
        faith_path = str(write_faithfulness_report(faith, Path(faithfulness_output)))

    def _preview(row: dict) -> dict:
        return {
            "student_id": row.get("student_id"),
            "concept_id": row.get("concept_id"),
            "predicted_correct_prob": row.get("predicted_correct_prob"),
            "diagnostician_explanation": row.get("diagnostician_explanation"),
            "critic_reason": row.get("critic_reason"),
            "hint_text": row.get("hint_text"),
        }

    return Tier2EvalReport(
        dataset=str(summary.get("dataset", "unknown")),
        fold=int(summary.get("fold", -1)),
        llm_backend=str(summary.get("llm_backend", "unknown")),
        n_samples=int(summary.get("n_samples", len(records))),
        n_flagged=int(summary.get("n_flagged", len(flagged))),
        flag_rate=float(summary.get("flag_rate", len(flagged) / len(records))),
        n_session_hyperedges=int(summary.get("n_session_hyperedges", len(records))),
        mean_predicted_prob=mean_prob,
        mean_kc_jaccard=faith.mean_kc_jaccard,
        median_kc_jaccard=faith.median_kc_jaccard,
        n_empty_grounded=faith.n_empty_grounded,
        case_study_flagged=[_preview(r) for r in flagged[:case_study_n]],
        case_study_passed=[_preview(r) for r in passed[:case_study_n]],
        source_summary=str(summary_path),
        source_records=str(records_path),
        faithfulness_report=faith_path,
    )


def write_tier2_eval_report(report: Tier2EvalReport, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
