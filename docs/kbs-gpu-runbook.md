# KBS GPU runbook — faithfulness + manipulation folds

This machine (dev checkout) had **no** `results/checkpoints/`, **no** P0
`e_pre_train_only.csv`, and **no** local Ollama at last check. Harness was
validated with a synthetic StubLLM grounded-vs-IG comparison under
`results/tables/synthetic_fold0_*`. Run the following on the RTX 3090 host
that already produced the original Tier-2 Ollama v2 pilot.

## Prerequisites

- `results/checkpoints/xes3g5m_fold0.pt` (or retrain via
  `scripts/04_run_tier2_pilot.py --save-checkpoint ... --checkpoint-only`)
- P0 exports:
  `external/p0_leakage_audit/data/processed/xes3g5m/fold_{0,1,2}/e_pre_train_only.csv`
- Ollama with `qwen2.5:7b` (and optionally `1.5b` / `14b` for scale)

## 1. Grounded + IG (n=500)

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode ollama \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt \
  --device cuda --sample-size 500
```

Outputs:

- `results/tables/xes3g5m_fold0_tier2_pilot_kbs_ollama_grounded.jsonl`
- `results/tables/xes3g5m_fold0_tier2_pilot_kbs_ollama_ig.jsonl`
- `results/tables/xes3g5m_fold0_kbs_faithfulness_comparison.json`

Then paste mean JC / flag rates into
`paper/tables/table_kc_jaccard_ig.tex` (replace `---` Ollama rows).

## 2. Optional LLM-scale sweep (n=100)

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode ollama \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt \
  --device cuda --run-scale
```

## 3. Manipulation check folds 1–2

```bash
python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 1 --device cuda
python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 2 --device cuda
```

Or via the suite:

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode eval-only \
  --grounded-records results/tables/xes3g5m_fold0_tier2_pilot_kbs_ollama_grounded.jsonl \
  --ig-records results/tables/xes3g5m_fold0_tier2_pilot_kbs_ollama_ig.jsonl \
  --manipulation-folds 1,2 --device cuda
```

## 4. Smoke without Ollama (needs E_pre + optional short train)

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode smoke --sample-size 20
```

## Notes

- StubLLM IG keeps the same `GroundedKCs` trailer construction as grounded
  (history IDs), so **JC delta is not informative under stub**; flag rate is
  (IG explanations omit `P(correct)` → Critic flags). Real JC separation is
  an Ollama result.
- Re-running with the `GroundedKCs` trailer changes the explanation
  distribution relative to the published n=40 human pack; keep human scores
  as-is or re-sample after the Ollama re-run.
