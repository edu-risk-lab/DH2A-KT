# GreyKT GPU runbook (RTX 3090)

GreyKT is **not** a drop-in replacement for DH2-KT until the C_B diagnostic
passes. Push this branch, then on the GPU host:

```bash
git pull
pytest tests/test_greykt.py tests/test_greykt_inputs.py tests/test_greykt_plumbing.py -v
```

Reuse `results/checkpoints/xes3g5m_fold0.pt`. Do not retrain DH2-KT for the
diagnostic unless that file is missing.

## 1. Premise diagnostic (do this first)

Needs P0 processed XES3G5M + `e_pre_train_only.csv` + CUDA + `torch_geometric`.
Evaluates **plain DH2-KT** on the validation split only.

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode diagnose \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt
```

Writes `results/tables/xes3g5m_fold0_black_confidence_diagnostic.json`.

| Verdict | Action |
|---------|--------|
| SUPPORTED | Continue to wrap, then optional train |
| AMBIGUOUS | Wrap is still informative; do not over-claim |
| NOT_SUPPORTED | Stop. Do not `--mode train` (script will refuse without `--force`) |

## 2. Wrap existing checkpoint (cheap first GreyKT number)

Uses frozen DH2-KT weights; white-box + gate have no learned params.

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode wrap \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt
```

Output: `results/tables/xes3g5m_fold0_greykt_wrap_v3.json` with fused vs black vs white AUC and the 2D `C_B x C_W` grid.

Optional variants: `--variant v4a` / `v4b` / `v4ab` (v4a fits temperature on **validation only**).

## 3. Train GreyKT fold 0 (only if diagnostic is not NOT_SUPPORTED)

Starts from the DH2-KT checkpoint and fine-tunes the black-box with fused BCE.
Matched GKT budget (batch 4, 10 epochs).

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cuda --mode train \
  --load-checkpoint results/checkpoints/xes3g5m_fold0.pt
```

Smoke (no full data wait):

```bash
python scripts/22_run_greykt.py configs/xes3g5m.yaml --fold 0 --device cpu --mode wrap --max-users 64 --force
```

(`--force` skips diagnostic; smoke only.)
