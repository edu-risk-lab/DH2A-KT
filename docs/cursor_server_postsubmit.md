# Cursor on GPU server — post-submit queue

Pull `main`, then run only the jobs below. Do **not** invent new groups or change τ / families. Manuscript edits stay on the author laptop; server work is stamps + allowlisted CSVs.

## Device

- One NVIDIA RTX 3090 (24 GB). Refuse smaller GPUs (`--min-vram-mib 20000`).
- Repo root: clone of `edu-risk-lab/DH2A-KT`, branch `main`.

## Priority (do in order)

### 1. Finish Junyi history-only, all learners, fold 2

Missing as of author push: `junyi-hist-full-fold2` (`time-history` and `t-zero-history`, ten seeds each = 20 jobs).

```bash
git pull --ff-only
python scripts/50_credit_ladder_plan.py --group junyi-hist-full-fold2
# then launch one-at-a-time (or loop until group done):
python scripts/50_credit_ladder_plan.py --group junyi-hist-full-fold2 --launch
```

After each finished batch (or when the group is complete):

```bash
git add logs/credit_ladder/junyi-hist-full-fold2_*.ok logs/credit_ladder/junyi-hist-full-fold2_*.log \
  results/tables/dh2_kt_ladder_junyi-hist-full-fold2_*_vs_p0.csv \
  results/tables/credit_ladder_status.json
git commit -m "$(cat <<'EOF'
Record junyi-hist-full-fold2 history-only runs.

EOF
)"
git push origin HEAD
```

On Windows PowerShell use an equivalent multi-line commit message; do not use `&&`.

### 2. Optional next (only if fold 2 is done and GPU is free)

Attention encoder on Junyi under the repository plan (onset already checked → history-only arms):

```bash
python scripts/50_credit_ladder_plan.py --group attn-junyi50k-fold0 --launch
# then attn-junyi50k-fold{1,2}, then attn-junyi-fold{0,1,2} if time remains
```

Commit/push the same way (group-prefixed logs + CSVs + `credit_ladder_status.json`).

## Do not run

- Do not relaunch finished `attn-assist-*`, `tracer-grid`, `tracer-seeds`, or credited family~C XES/ASSIST map arms.
- Do not unblock family~D Junyi onset groups.
- Do not edit `paper/*.tex` on the server unless the author asks.
- Do not force-push; do not amend published commits.

## If a job fails

Rerun the same command once. If it fails twice, leave it pending and push what succeeded with a note in the commit body.

## After push

Author laptop: `git pull`, then (optional) refresh Junyi history numbers in `paper/main.tex` / `table_junyi_history.tex` if fold 2 is complete.
