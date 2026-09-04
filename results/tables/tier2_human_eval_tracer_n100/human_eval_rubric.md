# Tier-2 human evaluation rubric — recommended tracer (B10)

## Goal
Rate Diagnostician explanations and Tutor hints produced from the
**recommended tracer** QKC-T (v4 LSTM + Q←KC + Linear Δt), not the v2
graph-only checkpoint used in the earlier n=40 pilot.

## Sample
- **N = 100** rows (`sample_id` 1–100), seed 42.
- Source: grounded Ollama `qwen2.5:7b` run on XES3G5M fold 0 (subsample of
  the n=500 pilot; includes 9 Critic-flagged rows).
- Raters: **≥3 independent** annotators. Do **not** discuss scores until
  everyone has submitted.

## Scales (Likert 1–5, integers only)
| Score | Faithfulness (to Tier-1 P(correct) + history) | Usefulness (for a teacher) |
|------:|-----------------------------------------------|----------------------------|
| 1 | Contradicts P(correct) or invents facts | Not actionable / misleading |
| 2 | Major mismatch or unsupported claim | Weak / vague |
| 3 | Partially aligned; some stretch | Somewhat useful |
| 4 | Aligned with minor wording issues | Clear, mostly actionable |
| 5 | Fully faithful to numbers + history | Directly usable next step |

## Procedure
1. Read `predicted_correct_prob`, `recent_history_summary`, then
   `diagnostician_explanation` and `hint_text`.
2. Fill `faithfulness_1to5` and `usefulness_1to5` (integers 1–5 only).
3. Optional: `notes` (e.g. Critic disagreement, invented KC facts).
4. Set `rater_id` to your initials (same value on every row).

## Blind rules
- Do not change any content columns.
- Do not re-score after seeing other raters.
- `critic_flagged` / `critic_reason` are context only — score the text
  yourself; do not copy the Critic’s binary flag as your Likert.
