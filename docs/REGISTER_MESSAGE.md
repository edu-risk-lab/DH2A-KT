# Register a message on the DH²A-KT credit ladder

The protocol is Section 4 of `paper/main.tex` (`sec:protocol`) and Algorithm Register. A new message is not added by editing the tracer alone.

## What you declare

1. **Message.** The tensor or feature the on arm receives and the control does not, or that the control zeros.
2. **Rung.**
   - `(i)` capacity-matched: same maps; only the message content changes (observed versus zero incidence).
   - `(ii)` architecture-matched: same maps; the input is zeroed or misaligned (T-zero, T-misaligned).
   - `(iii)` package: the branch is added, including new parameters. Record `Δparams`. A package pass does not credit the input alone.
3. **Designated control.** One twin per contrast. Do not compare against a different architecture.
4. **Corpus, learner fold, five training seeds.** The reference seeds are `42, 17, 1234, 0, 2024` on XES3G5M fold 0. Use the same five seeds on a transfer corpus.
5. **Outcome**, from the five paired validation deltas `Δ_s`:
   - **Credited** if every `Δ_s ≥ τ`.
   - **Consistent-positive, not credited** if every `Δ_s > 0` and at least one `Δ_s < τ`.
   - **Unsupported** otherwise.

## Threshold

On XES3G5M fold 0, `τ = +0.002`. That default sits in the stability band `(+0.00175, +0.00236]` (`paper/tables/table_threshold_sensitivity.tex`).

On any other corpus or backbone, report both:

- absolute `τ = +0.002`;
- `τ_rel = max(0.002, 10 * SD_seed)`, where `SD_seed` is the sample standard deviation of the **control** arm's validation AUC across the five seeds.

`K = 10` is the largest integer that keeps both XES3G5M packages credited (bound 10.47). It was fixed after the XES runs. If the two cuts disagree, publish both maps. `python scripts/51_transfer_credit_tables.py` recomputes every outcome and stops if `K` no longer preserves the XES map.

## Reference command shape

The tracer entry point is `scripts/03_train_tier1.py`. Package versus no-time on the reference backbone:

```bash
python scripts/03_train_tier1.py configs/xes3g5m.yaml --fold 0 --device cuda \
  --architecture v4 --use-questions --question-graph --time-gap --time-gap-mode both \
  --mask-repeats --window-mode chunked --seq-len 400 --seed 42
```

Drop `--time-gap` for the no-time control. Add `--time-gap-pad` instead when the no-time arm must carry the same `Linear(1, hidden)` (256 parameters at hidden 128) without changing the representation. Keep `--time-gap` and set `--time-gap-control zero` or `misaligned` for rung (ii). `--time-gap` and `--time-gap-pad` cannot be combined. Repeat for seeds `42 17 1234 0 2024`.

The batch driver for the transfer ladders is `scripts/50_credit_ladder_plan.py`. It prints the run list and, with `--launch`, starts the next job only when CUDA is available. It does not invent AUC numbers.

## What not to do

- Do not credit a single seed.
- Do not loosen `+0.002` after seeing the sign of `Δ`.
- Do not describe a package pass as proof that the aligned input is isolated. That claim requires rung (ii).
- Do not rank the tracer against a baseline that was scored on a different evaluator.
