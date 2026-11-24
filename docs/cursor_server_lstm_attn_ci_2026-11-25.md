# Cursor on GPU server — recompute LSTM−attention learner CI (25 Nov 2026)

Paste from **You are on the GPU VPS** to the end into Cursor on the VPS after it has pulled `origin/main`.

Laptop will push this file and the shared-weight fix in `scripts/66_nov5_dump_analyses.py`. Do **not** train new models. Do **not** edit `paper/main.tex` or `paper/sections/results.tex`. The author laptop will restore the CI sentence after this JSON/table lands.

---

You are on the GPU VPS for repo **edu-risk-lab/DH2A-KT**, branch **main**, remote **https://github.com/edu-risk-lab/DH2A-KT.git**.

## Goal

Recompute the **learner-clustered 95% interval** of the mean-over-seeds difference of package effects

\[
(\mathrm{AUC}_{\mathrm{pkg}}-\mathrm{AUC}_{\mathrm{no\text{-}time}})_{\mathrm{LSTM}}
-
(\mathrm{AUC}_{\mathrm{pkg}}-\mathrm{AUC}_{\mathrm{no\text{-}time}})_{\mathrm{attn}}
\]

on the **intersection of held-out learners** of XES3G5M fold 0 (ten seeds). An earlier interval used **independent multinomial weights per encoder and seed** and is **withdrawn**. The script on `main` now draws **one weight matrix** (\(B{=}2000\), seed `20261023`) and reuses it for both encoders, both arms, and all ten seeds; histogram AUC uses **2000 quantile bins**.

This is family E. It does not enter family C. Do not invent numbers.

## 0. Sync

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

Confirm you have the shared-weight fix:

```bash
python -c "import pathlib,re; t=pathlib.Path('scripts/66_nov5_dump_analyses.py').read_text(encoding='utf-8'); assert 'shared_weights' in t and 'N_BINS = 2000' in t; print('script 66 ok')"
```

If that assert fails, stop. Confirm GPU and checkpoints:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi
ls results/checkpoints | head
```

Need a ~24 GB GPU only if any of the 40 npz files below are missing. If all 40 dumps already exist, the recompute is CPU (NumPy) and does not need the GPU.

**Do not overwrite** `results/tables/credit_ladder_status.json`. **Do not** run `scripts/50_credit_ladder_plan.py`. **Do not** run `scripts/67_xes_attempt_table.py`. **Do not** edit existing preregistration files. **Do not** move tag `es-submit-2026-10-22b`.

## 1. Dumps needed (fold 0, ten seeds)

40 files under `results/predictions/` (gitignored):

| group | arms | seeds |
|---|---|---|
| `xes-ref` | `time-package`, `no-time` | `0 17 42 1234 2024 7 99 314 2718 3407` |
| `backbone-attn` | `time-package`, `no-time` | same |

Names look like `xes3g5m_fold0_xes-ref_time-package_s0.npz` and `xes3g5m_fold0_backbone-attn_no-time_s42.npz`. Each npz must contain `user_ids`, `targets`, `probs`.

If a previous dump already wrote these files, script 66 prints `SKIP` and does not rescore.

```bash
python scripts/66_nov5_dump_analyses.py --dump-first
```

`--dump-first` rescored **only those 40 runs**, not the rest of family C. If a checkpoint is missing, print the path, leave that arm pending, and do not invent a CI. Rescore logs go to `logs/rescore/`.

Do **not** commit `results/predictions/` or `*.pt`.

## 2. What success looks like

`results/tables/nov5_dump_analyses.json` → `lstm_minus_attn` must include:

- `"shared_weights": true`
- `"n_bins": 2000`
- `"n_shared_learners"` (expected 5420)
- `"boot_ci95": [lo, hi]`
- `"mean_delta_of_deltas"` (point should stay about \(+0.94\times 10^{-3}\); the interval may change)
- `"max_abs_binned_minus_exact_auc"`
- `"seeds_used"`: all ten seeds

`paper/tables/table_lstm_attn_delta.tex` must show the point **and** the learner 95% CI, and the caption must say one weight draw is shared across both encoders, both arms, and all ten seeds.

Alert-budget numbers may be rewritten from the same dumps; that is expected. Do not change family C JSON.

If `lstm_minus_attn` is `null`, stop and report which npz/checkpoints are missing.

## 3. Gitignore / do not stage

Never add: `results/predictions/`, `results/checkpoints/`, `*.pt`, `paper/main.pdf`, `paper/appendix.pdf`, `paper/main.tex`, `paper/sections/results.tex`, `docs/preregistration_primary_rule.md`, `KBS_reviewer/`, `TaiLieuThamKhao/`, `paper/S1_P0_*`, `.env`.

If `credit_ladder_status.json` changed, restore it:

```bash
git checkout -- results/tables/credit_ladder_status.json
```

## 4. Commit and push (only these)

```bash
git add results/tables/nov5_dump_analyses.json paper/tables/table_lstm_attn_delta.tex
git status
git diff --cached
git log -5 --oneline
```

Commit message:

```
Recompute LSTM-attention learner CI with shared bootstrap weights.
```

Then:

```bash
git pull --ff-only origin main
git push origin main
```

No `--force`, no `--amend`, no `--no-verify`. If commit or push fails, stop and report.

## Done when

Reply with:

1. how many of the 40 npz were `SKIP` vs rescored vs `MISSING`
2. `n_shared_learners`, point \(\times 10^{-3}\), `boot_ci95` \(\times 10^{-3}\), `max_abs_binned_minus_exact_auc`
3. commit SHA and `git status -sb` after push
4. do **not** rewrite Results prose; the author laptop will restore the CI sentence from this JSON
