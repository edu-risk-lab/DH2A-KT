# GreyKT GPU runbook (RTX 3090)

## Status (fold 0, 2026-08-14)

Frequency C_B is **NOT_SUPPORTED**. ρ(C_B, squared error) = −0.032
(threshold |ρ| ≥ 0.05). C_B saturates at ~0.997–1.0 because XES3G5M
concepts have huge N_c^train, so the gate is noise. Artefact:

`results/tables/xes3g5m_fold0_black_confidence_diagnostic.json`

Do **not** `--mode wrap` / `train` on `--signal frequency`. Do **not**
`--force` that diagnostic. Next signal: MC-dropout (graph-encoder dropout
only — DualGatedUpdate has no dropout, so each sample must re-encode
concepts with `model.train()`).

```bash
git pull
pytest tests/test_greykt.py tests/test_greykt_inputs.py tests/test_greykt_plumbing.py tests/test_black_confidence.py -v
```

## 1. MC-dropout diagnostic (do this next)

Reuses `results/checkpoints/xes3g5m_fold0.pt`. Writes a **separate** JSON
so the frequency artefact is not overwritten. Expect ~5–10 min (8 extra
graph+sequence forwards on validation).

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode diagnose \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal mc_dropout
```

Writes `results/tables/xes3g5m_fold0_black_confidence_diagnostic_mc_dropout.json`.

| Verdict | Action |
|---------|--------|
| SUPPORTED | Wrap, then optional train, still `--signal mc_dropout` |
| AMBIGUOUS | Inspect std min/median/max in the JSON; wrap is optional |
| NOT_SUPPORTED | Stop. Next candidate is a **multi-seed ensemble**, not `--force` |

If MC std is itself tiny (graph dropout does not move next-step logits),
the diagnostic will also report saturation. That means this architecture
cannot express epistemic uncertainty without retraining (e.g. dropout on
the prediction head) or an ensemble.

## 2. Wrap (only if MC diagnostic is not NOT_SUPPORTED)

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal mc_dropout
```

`--signal` defaults to `mc_dropout` in this driver. The frequency JSON
will **not** unlock wrap.

## 3. Train (same gate)

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal mc_dropout
```

MC-dropout C_B is computed under `no_grad` each batch (8 samples). Slow
but the white-box / gate stay non-learned.
