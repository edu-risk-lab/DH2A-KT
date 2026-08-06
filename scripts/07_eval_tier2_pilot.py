#!/usr/bin/env python3
"""M9 driver: evaluate Tier-2 pilot logs (Critic flag rate + case-study excerpts).

Usage:
    python scripts/07_eval_tier2_pilot.py results/tables/xes3g5m_fold0_tier2_pilot_summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.tier2.eval import evaluate_tier2_pilot, write_tier2_eval_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        type=Path,
        help="Pilot summary JSON, e.g. results/tables/xes3g5m_fold0_tier2_pilot_summary.json",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Pilot JSONL (default: sibling *_pilot.jsonl from summary stem)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output eval JSON (default: *_tier2_eval.json next to summary)",
    )
    parser.add_argument("--case-study-n", type=int, default=3)
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    if args.records is not None:
        records_path = args.records.resolve()
    else:
        name = summary_path.name
        if name.endswith("_summary.json"):
            records_name = name.replace("_summary.json", ".jsonl")
        elif name.endswith("_tier2_pilot_summary.json"):
            records_name = name.replace("_tier2_pilot_summary.json", "_tier2_pilot.jsonl")
        else:
            records_name = summary_path.stem + ".jsonl"
        records_path = summary_path.with_name(records_name)

    if args.output is not None:
        output_path = args.output.resolve()
    else:
        output_path = summary_path.with_name(
            summary_path.name.replace("_summary.json", "_eval.json").replace(
                "_tier2_pilot_summary.json", "_tier2_eval.json"
            )
        )

    report = evaluate_tier2_pilot(summary_path, records_path, case_study_n=args.case_study_n)
    write_tier2_eval_report(report, output_path)
    print(f"Wrote Tier-2 eval: {output_path}")
    print(f"  flag_rate={report.flag_rate:.4f} n_samples={report.n_samples} backend={report.llm_backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
