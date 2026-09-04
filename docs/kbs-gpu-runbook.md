# KBS GPU runbook — faithfulness + manipulation folds

**Status 2026-08-13:** Core GPU loop **done** and merged to `main`
(`4cda173`, `7d1d04b`):

| Item | Result |
|------|--------|
| Ollama grounded vs IG ($n{=}500$) | Flag $0.026$ / $0.768$; mean JC $0.152$ / $0.142 |
| Manipulation folds 0–2 | All pass (ΔAUC $0.038$ / $0.043$ / $0.047`) |
| Best-epoch train restore | Fixes fold-1 final-epoch collapse |
| Paired ablation + w/o $L_{aux}$ | Fold-0 AUC $0.754$, ΔAUC $0.039$, still passes |

Optional remaining: LLM-scale sweep (`--run-scale`). Author gate still open
(`docs/kbs-author-gate.md`).

## Prerequisites

- `results/checkpoints/xes3g5m_fold0.pt` (or retrain via
  `scripts/04_run_tier2_pilot.py --save-checkpoint ... --checkpoint-only`)
- P0 exports:
  `external/p0_leakage_audit/data/processed/xes3g5m/fold_{0,1,2}/e_pre_train_only.csv`
- Ollama with `qwen2.5:7b` (and optionally `1.5b` / `14b` for scale)

## 1. Grounded + IG (n=500) — done

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode ollama \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt \
  --device cuda --sample-size 500
python scripts/19_update_kbs_faithfulness_table.py
```

Outputs (gitignored JSONL; numbers synced into paper tables):

- `results/tables/xes3g5m_fold0_tier2_pilot_kbs_ollama_grounded.jsonl`
- `results/tables/xes3g5m_fold0_tier2_pilot_kbs_ollama_ig.jsonl`
- `results/tables/xes3g5m_fold0_kbs_faithfulness_comparison.json`

## 2. Optional LLM-scale sweep (n=100)

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode ollama \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt \
  --device cuda --run-scale
```

## 3. Manipulation check folds 1–2 — done

```bash
python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 1 --device cuda
python scripts/05_run_manipulation_check.py configs/xes3g5m.yaml --fold 2 --device cuda
```

## 4. Smoke without Ollama

```bash
python scripts/18_run_kbs_faithfulness_suite.py --mode smoke --sample-size 20
```

## 5. B12 structure checks (rewire + label permute) — run on GPU

Frozen-weight checks on the **graph-only v2 diagnostic encoder**, not QKC-T.
Node-drop already shows dependence; these two ask whether the encoder uses
the *correct* relation structure.

```bash
git pull
python scripts/32_b12_structure_checks.py --device cuda
# or one fold:
python scripts/32_b12_structure_checks.py --device cuda --folds 0
```

Needs `results/checkpoints/xes3g5m_fold{0,1,2}.pt`. The driver **aborts** if
a v2 full-eval clean AUC is below 0.65 (fold~0 once scored 0.535 — do not
keep that JSON as a result). Writes:

- `results/tables/xes3g5m_foldN_manipulation_check_degree_preserving_rewire.json`
- `results/tables/xes3g5m_foldN_manipulation_check_relation_label_permute.json`

On a single-kind v2 graph, relation-label permutation is expected near-null.
Paste `auc_drop` into `paper/tables/table_manipulation.tex` (rows currently
`not run`). Do not invent numbers.

## 6. Capacity-matched Q←KC attribution — required rerun

The historical `hg_qkc_off` arm removed the whole question pathway and is
not a valid attribution twin. This rerun keeps `question_embed`,
`question_gcn_linear`, and `question_in_proj` in both arms; only the
train-only incidence message is removed in the `zero` arm.

```bash
git fetch origin
git checkout qkc-capacity-matched
git pull --ff-only
python scripts/34_qkc_capacity_matched.py --device cuda
```

For a one-seed smoke run before launching all ten jobs:

```bash
python scripts/34_qkc_capacity_matched.py --device cuda --seeds 42
```

Outputs:

- `results/tables/qkc_capacity_matched_matrix.csv`
- `results/tables/qkc_capacity_matched_summary.json`
- `results/tables/qkc_capacity_matched_logs/`
- `results/checkpoints/xes3g5m_fold0_qkc_cm_{observed,zero}_sSEED.pt`

The driver refuses a result if the two arms have different checkpoint state
shapes/element counts, if the observed arm has no Q–KC links, or if the zero
arm retains any links. Do not update the manuscript until all five paired
seeds finish. The claim passes only when the measured paired validation
difference supports it; do not reuse the legacy hard-off delta.

## Notes

- StubLLM IG keeps the same `GroundedKCs` trailer construction as grounded
  (history IDs), so **JC delta is not informative under stub**; flag rate is
  (IG explanations omit `P(correct)` → Critic flags).
- Ollama grounded--IG ΔJC is small (~0.010); the main IG signal is Critic
  flag rate. Do not over-claim KC-overlap separation.
- Human n=40 pack is from the pre-`GroundedKCs` v2 run; distribution differs
  from the faithfulness re-run.
