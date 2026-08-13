# GreyKT GPU runbook (RTX 3090)

## Status (fold 0, 2026-08-14)

Two C_B signals are **NOT_SUPPORTED**. Do not `--force`. Do not wrap/train
on either of them.

| Signal | ρ(C_B, err²) | Notes | Artefact |
|--------|--------------|-------|----------|
| `frequency` | −0.032 | Saturated (~0.997–1.0) | `..._black_confidence_diagnostic.json` |
| `mc_dropout` | −0.020 | **Not saturated** (std med 0.0425, max 0.479; C_B p05–p95 0.77–0.96). Epistemic graph-dropout variance does not track next-step error. | `..._black_confidence_diagnostic_mc_dropout.json` |

MC-dropout failing with real dynamic range means a **multi-seed ensemble
is the same hypothesis at much higher cost** (two extra full DH2-KT
trains). Do not start that yet.

Next cheap test: **predictive** C_B = |2 p_B − 1| of eval-mode output
(one forward, ~40s). If this also fails, stop GreyKT on XES3G5M; the
paper stays DH2-KT.

```bash
git pull
pytest tests/test_greykt.py tests/test_greykt_inputs.py tests/test_greykt_plumbing.py tests/test_black_confidence.py -v
```

## 1. Predictive-confidence diagnostic (do this next)

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode diagnose \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive
```

Writes `results/tables/xes3g5m_fold0_black_confidence_diagnostic_predictive.json`.

This signal is output-confidence, not “training reliability”. If it is
SUPPORTED, a GreyKT write-up must say so. Overconfident errors (high
|2p−1| still wrong) can make it NOT_SUPPORTED.

| Verdict | Action |
|---------|--------|
| SUPPORTED | Wrap/train with `--signal predictive` |
| AMBIGUOUS | Inspect buckets; do not over-claim |
| NOT_SUPPORTED | **Stop GreyKT.** Ensemble is not the next step. |

## 2. Wrap / train (only if predictive is not NOT_SUPPORTED)

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive
```
