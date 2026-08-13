# Tier-2 human evaluation rubric (C-Human)

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
