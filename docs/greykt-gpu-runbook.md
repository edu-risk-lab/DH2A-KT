# GreyKT GPU runbook (RTX 3090)

## Status (fold 0, 2026-08-14)

| Signal | Verdict | ρ(C_B, err²) | Notes |
|--------|---------|--------------|-------|
| `frequency` | NOT_SUPPORTED | −0.032 | Saturated (~0.997–1.0) |
| `mc_dropout` | NOT_SUPPORTED | −0.020 | Real variance; does not track error |
| `predictive` | **SUPPORTED** | **−0.588** | C_B = \|2p_B−1\|; mean 0.516; Brier 0.205→0.167 |

C_B is **output-confidence**, not training-frequency reliability. Any
paper text must say that. Do not `--force` the two failed signals.

The most-confident bucket is slightly worse-calibrated than the fourth
(Brier 0.167 vs 0.164, ECE 0.073 vs 0.008) — overconfidence. Wrap v3
first; consider `--variant v4a` only if fused NLL/ECE need it.

```bash
git pull
pytest tests/test_greykt.py tests/test_greykt_inputs.py tests/test_greykt_plumbing.py tests/test_black_confidence.py -v
```

## 1. Wrap (do this next)

Frozen DH2-KT + white-box + predictive gate. No extra learned params.
White-box is a Python timestep loop — expect longer than the 37s
black-box diagnostic (likely tens of minutes on val+test).

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive
```

Writes `results/tables/xes3g5m_fold0_greykt_wrap_v3_predictive.json`.

Read fused vs black vs white AUC **and** the 2D `C_B × C_W` grid. The
design claim is that fused wins in the low-C_B / high-C_W cell.

## 2. Train (only after wrap numbers exist)

Starts from the DH2-KT checkpoint; fused BCE; C_B is detached `|2p-1|`.
Matched GKT budget (batch 4, 10 epochs). Slow: white-box loop every batch.

Each completed epoch writes:

- `results/checkpoints/greykt_xes3g5m_fold0_v3_predictive/latest.pt`
- `.../best.pt` when train loss improves

Train also **precomputes white-box once** (no learned params) and reuses
it every epoch — major wall-clock win; scientific equivalence unchanged.

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive \
  --batch-size 12
```

`--batch-size 12` is a **speed override** (not matched GKT batch 4). Report it
as observational if used in any table. Omit the flag to keep batch 4.

To actually spend leftover VRAM on throughput (hypergraph encode is ~5.5 GB
fixed; sequence activations scale with batch):

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive \
  --batch-size 64 --amp --freeze-hypergraph
```

`--freeze-hypergraph` encodes the concept graph once per epoch (no graph
backward). Dual-gate still trains. Combined with `--amp`, leftover VRAM goes
into a larger sequence batch.

After a crash / kill, resume (continues from the next epoch after `latest.pt`):

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive --resume
```

Optional after wrap, if calibration looks poor:

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive --variant v4a
```
