#!/usr/bin/env python3
"""Export Tier-2 pilot rows for offline human rating (C-Human).

Samples 40 explanations from the Ollama v2 pilot JSONL and writes:
  - a rater CSV with Likert columns (faithfulness / usefulness / 1-5)
  - a short rubric markdown for 2–3 independent raters

Usage:
    python scripts/10_export_tier2_human_eval.py
    python scripts/10_export_tier2_human_eval.py --n 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

RUBRIC_MD = """# Tier-2 human evaluation rubric (C-Human)

## Goal
Rate Diagnostician explanations and Tutor hints from the AgentDG-KT pilot
(`xes3g5m` fold 0, Ollama qwen2.5:7b, Critic v2).

## Sample
- **N = 40** stratified / random rows (see CSV `sample_id`).
- Raters: 2–3 independent annotators; do **not** discuss scores until after
  all ratings are submitted.

## Scales (Likert 1–5)
| Score | Faithfulness (to Tier-1 P(correct) + history) | Usefulness (for a teacher) |
|------:|-----------------------------------------------|----------------------------|
| 1 | Contradicts P(correct) or invents facts | Not actionable / misleading |
| 2 | Major mismatch or unsupported claim | Weak / vague |
| 3 | Partially aligned; some stretch | Somewhat useful |
| 4 | Aligned with minor wording issues | Clear, mostly actionable |
| 5 | Fully faithful to numbers + history | Directly usable next step |

## Procedure
1. Read `predicted_correct_prob`, `recent_history_summary`, then the
   `diagnostician_explanation` and `hint_text`.
2. Fill `faithfulness_1to5` and `usefulness_1to5` (integers only).
3. Optional: `notes` for disagreements or Critic comments.
4. Leave `rater_id` as your initials (e.g. `AB`).

## Aggregate (after collection)
Report mean ± SD per scale, and pairwise agreement (e.g. Spearman / ICC)
across raters. Do **not** re-score after seeing other raters.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "tables"
        / "xes3g5m_fold0_tier2_pilot_ollama_v2.jsonl",
    )
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "tier2_human_eval",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Missing pilot JSONL: {args.input}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if len(df) < args.n:
        print(f"Only {len(df)} rows available; requested n={args.n}", file=sys.stderr)
        return 1

    # Prefer diversity: half flagged if any, else uniform random.
    if "critic_flagged" in df.columns and df["critic_flagged"].any():
        flagged = df[df["critic_flagged"]].copy()
        clean = df[~df["critic_flagged"]].copy()
        n_flag = min(len(flagged), max(1, args.n // 5))
        n_clean = args.n - n_flag
        sample = pd.concat(
            [
                flagged.sample(n=n_flag, random_state=args.seed),
                clean.sample(n=n_clean, random_state=args.seed),
            ],
            ignore_index=True,
        )
    else:
        sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    sample = sample.reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    export_cols = [
        "sample_id",
        "student_id",
        "concept_id",
        "predicted_correct_prob",
        "recent_history_summary",
        "diagnostician_explanation",
        "critic_flagged",
        "critic_reason",
        "hint_text",
        "session_hyperedge_id",
    ]
    export_cols = [c for c in export_cols if c in sample.columns]
    out = sample[export_cols].copy()
    out["rater_id"] = ""
    out["faithfulness_1to5"] = ""
    out["usefulness_1to5"] = ""
    out["notes"] = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "human_eval_samples.csv"
    rubric_path = args.out_dir / "human_eval_rubric.md"
    meta_path = args.out_dir / "human_eval_meta.json"
    out.to_csv(csv_path, index=False)
    rubric_path.write_text(RUBRIC_MD, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(args.input.relative_to(REPO_ROOT)),
                "n": len(out),
                "seed": args.seed,
                "n_critic_flagged": int(out["critic_flagged"].sum()) if "critic_flagged" in out.columns else 0,
                "scales": ["faithfulness_1to5", "usefulness_1to5"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} ({len(out)} rows)")
    print(f"Wrote {rubric_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
