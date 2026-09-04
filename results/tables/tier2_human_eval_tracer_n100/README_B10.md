# B10 — Likert on recommended tracer (QKC-T)

## Goal
Human ratings on Diagnostician explanations from the **recommended tracer**
(v4 Q←KC + Linear Δt), not the v2 graph-only checkpoint used in the n=40 pilot.

## Pack
- Source: `tier2_p0_dt_on/..._grounded.jsonl` (n=500 Ollama qwen2.5:7b)
- Sample: **n=100**, seed 42 (9 Critic-flagged rows included)
- Raters: **3** independent (A/B/C)
- Scales: faithfulness / usefulness Likert 1–5

## Results (returned 2026-09-04)
| Scale | Mean ± SD | Mean pairwise ρ | Within±1 |
|-------|-----------|-----------------|----------|
| Faithfulness | **4.55 ± 0.63** | 0.43 | 95.0% |
| Usefulness | **4.01 ± 0.74** | 0.34 | 94.7% |

Summary: `human_eval_summary.json`. Paper table: `paper/tables/table_human_eval.tex`.

## Status
- [x] Export sample pack
- [x] Distribution guides (A/B/C + rubric + message template)
- [x] Collect 3 rater CSVs into `distribution/returned/`
- [x] Aggregate with `scripts/15_aggregate_human_eval.py`
- [x] Update `paper/tables/table_human_eval.tex` + §Tier2 / Limitations

Do **not** overwrite the archived v2 n=40 pack under `tier2_human_eval/`.
