# GreyKT GPU runbook (RTX 3090)

## Status (fold 0)

| Signal | Verdict | ρ(C_B, err²) | Notes |
|--------|---------|--------------|-------|
| `frequency` | NOT_SUPPORTED | −0.032 | Saturated (~0.997–1.0) |
| `mc_dropout` | NOT_SUPPORTED | −0.020 | Real variance; does not track error |
| `predictive` | **SUPPORTED** | **−0.588** | C_B = \|2p_B−1\|; mean 0.516; Brier 0.205→0.167 |

C_B is **output-confidence**, not training-frequency reliability. Any
paper text must say that. Do not `--force` the two failed signals.

### Infrastructure (2026-08-15) — re-wrap before trusting numbers

Two bugs in the GreyKT driver, now fixed in tree. **Do not cite the
2026-08-14 wrap JSON** (`black_auc = 0.535`) and **do not resume** an
old GreyKT `latest.pt` that selected `best.pt` on train loss.

1. **Wrap black-box.** `wrap_trained_fold` used to `load_state_dict` into
   a freshly constructed DH2KT. Fold-0 wrap reported `black_auc = 0.535`
   (near chance) at the same n=1,078,580 where Table 4 DH²-KT is ~0.750.
   Wrap now **shares** `trained.model` as `GreyKT.black_box`. After
   collect, the driver compares GreyKT `black_auc` to
   `evaluate_auc(trained.model)` on the same loader and **exits** if
   `|Δ| > 0.01`.
2. **Checkpoint selection.** `eval_df = concat(valid, test)` is the
   Table-4 *report* loader. `best.pt` was selected on **train loss**, not
   on that loader (so it was not test-set early stopping), but train-loss
   selection is still wrong for a paper table. Selection is now
   **validation fused NLL**. JSON reports `valid`, `test`, and
   `valid+test` (Table 4 comparable). Never use test for model selection.
3. **White-box next-step.** The white-box used to query `c_t` and credit
   the DH2KT-shifted `responses[:, t]`. It now queries `c_{t+1}` and
   credits `correct[:, t]` when that tensor is on the batch. Re-wrap
   before reporting white AUC. Zero learned parameters either way.

```bash
git pull
pytest tests/test_greykt.py tests/test_greykt_inputs.py tests/test_greykt_plumbing.py tests/test_black_confidence.py -v
```

## Findings to keep (matched train, 2026-08-14)

These are from the matched-budget train JSON, whose `best.pt` was chosen
on train loss (not test). Directional only until a validation-selected
re-train exists. Do not put them in `paper/main.tex` yet.

- **Gate direction.** On the 2D grid, low-C_B cells: fused beat black by
  about **+0.011 AUC**. High-C_B cells: fused slightly worse (**−0.003**).
  That is the predicted sign. Caveat: C_B = `|2p_B−1|` means “the black
  box is unsure”, so the low-C_B win is slightly circular, but the
  direction is still the central hypothesis.
- **White-box AUC.** The frozen Beta-Binomial branch reported **0.674
  AUC with 0 learned parameters** (identical across wrap / matched /
  observational because the white-box does not train). Re-measure after
  the next-step alignment; the number is worth reporting if it holds.

## 1. Wrap (do this next)

Frozen DH2-KT v2 checkpoint + white-box + predictive gate. No extra
learned params. The run **must** print `wrap audit OK` and
`black_auc ≈ native evaluate_auc ≈ 0.75`. If black stays near 0.53, stop.

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive
```

Writes `results/tables/xes3g5m_fold0_greykt_wrap_v3_predictive.json`.

Read fused vs black vs white AUC **and** the 2D `C_B × C_W` grid (now
includes `white_auc` per cell). The design claim is that fused wins in
the low-C_B / high-C_W cell. Headline comparison to Table 4 uses the
`valid+test` split; selection never sees test.

## 2. Train (only after wrap audit passes)

Starts from the DH2-KT checkpoint; fused BCE; C_B is detached `|2p-1|`.
Matched GKT budget (batch 4, 10 epochs). `best.pt` is the epoch with
lowest **validation** fused NLL.

Each completed epoch writes:

- `results/checkpoints/greykt_xes3g5m_fold0_v3_predictive/latest.pt`
- `.../best.pt` when valid NLL improves

Do **not** `--resume` a checkpoint whose `meta.selection_metric` is
missing or `train_loss`; tracking resets, but the run is cleaner from
scratch.

Train also **precomputes white-box once** (no learned params) and reuses
it every epoch — major wall-clock win; scientific equivalence unchanged.

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --signal predictive
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
into a larger sequence batch. Observational, not matched-budget.

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
