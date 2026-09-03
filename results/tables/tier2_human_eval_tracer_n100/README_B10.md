# B10 — Likert on recommended tracer (QKC-T)

## Goal
Human ratings on Diagnostician explanations from the **recommended tracer**
(v4 Q←KC + Linear Δt), not the v2 graph-only checkpoint used in the n=40 pilot.

## Pack
- Source: `tier2_p0_dt_on/..._grounded.jsonl` (n=500 Ollama qwen2.5:7b)
- Sample: **n=100**, seed 42 (stratified on Critic flags when present)
- Raters: **≥3** independent (A/B/C CSVs in `distribution/`)
- Scales: faithfulness / usefulness Likert 1–5 (same rubric)

## Status
- [x] Export sample pack
- [ ] Collect 3 rater CSVs into `distribution/returned/`
- [ ] Aggregate with `scripts/15_aggregate_human_eval.py`
- [ ] Update `paper/tables/table_human_eval.tex` + §Tier2

Do **not** overwrite the archived v2 n=40 pack under `tier2_human_eval/`.
