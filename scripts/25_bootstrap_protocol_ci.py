#!/usr/bin/env python3
"""Learner-level bootstrap confidence interval for a difference between two AUCs.

Two models scored on the same evaluation split give two AUCs, and a raw gap of a
few ten-thousandths says nothing without an interval. Resampling *positions*
would understate the uncertainty, because positions from one learner are not
independent: one learner contributes hundreds of correlated rows. This script
resamples *learners* with replacement and recomputes both AUCs on the same
resample, so the two models always see the same learners and the interval is
paired at the level where the data is actually independent.

Each ``.npz`` must hold ``ts`` (labels), ``ps`` (probabilities) and ``us``
(learner id) with one entry per scored position. The two files need not contain
identical position sets; the bootstrap only requires that they cover the same
learners, which is why the paired unit is the learner and not the row.

Usage:
    python scripts/25_bootstrap_protocol_ci.py \
        --a "DH2-KT v4q=results/predictions/clean/v4q_clean_clean.npz" \
        --b "SimpleKT=results/pykt_work_clean/xes3g5m/fold_0/C_simplekt_clean/test_predictions.npz"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC with averaged ranks for ties, matching ``roc_auc_score``.

    Ties are common here rather than incidental: resampling learners with
    replacement duplicates whole blocks of identical scores, so tie handling has
    to be vectorised or the bootstrap becomes the bottleneck.
    """
    n = scores.size
    n_pos = float(labels.sum())
    n_neg = float(n - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    group_start = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
    counts = np.diff(np.concatenate((group_start, [n])))
    # Ranks are 1-based, so a group starting at index i spans ranks i+1 .. i+count.
    averaged = (group_start + 1.0) + (counts - 1.0) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(averaged, counts)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def load(spec: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if "=" not in spec:
        raise SystemExit(f"expected NAME=PATH, got {spec!r}")
    name, raw = spec.split("=", 1)
    data = np.load(Path(raw))
    missing = {"ts", "ps", "us"} - set(data.files)
    if missing:
        raise SystemExit(f"{raw} is missing {sorted(missing)}; it holds {data.files}")
    return name, data["ts"].astype(np.int8), data["ps"].astype(np.float64), data["us"]


def group_positions(users: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
    groups = np.split(order, boundaries)
    keys = sorted_users[np.concatenate(([0], boundaries))]
    return dict(zip(keys.tolist(), groups, strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, metavar="NAME=PATH")
    parser.add_argument("--b", required=True, metavar="NAME=PATH")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    name_a, ts_a, ps_a, us_a = load(args.a)
    name_b, ts_b, ps_b, us_b = load(args.b)

    groups_a = group_positions(us_a)
    groups_b = group_positions(us_b)
    shared = sorted(set(groups_a) & set(groups_b))
    if not shared:
        raise SystemExit("the two files share no learner ids; cannot pair")

    full_a = fast_auc(ts_a, ps_a)
    full_b = fast_auc(ts_b, ps_b)
    print(f"{name_a:<24} AUC={full_a:.6f}  n={ts_a.size:,}  learners={len(groups_a):,}")
    print(f"{name_b:<24} AUC={full_b:.6f}  n={ts_b.size:,}  learners={len(groups_b):,}")
    print(f"learners in both        : {len(shared):,}")
    print(f"observed difference     : {full_a - full_b:+.6f}  ({name_a} minus {name_b})")
    print()

    rng = np.random.default_rng(args.seed)
    learners = np.asarray(shared)
    boot_a = np.empty(args.n_boot)
    boot_b = np.empty(args.n_boot)
    for i in range(args.n_boot):
        picked = rng.choice(learners, size=learners.size, replace=True)
        idx_a = np.concatenate([groups_a[int(u)] for u in picked])
        idx_b = np.concatenate([groups_b[int(u)] for u in picked])
        boot_a[i] = fast_auc(ts_a[idx_a], ps_a[idx_a])
        boot_b[i] = fast_auc(ts_b[idx_b], ps_b[idx_b])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_boot} resamples", flush=True)

    diff = boot_a - boot_b
    lo, hi = np.percentile(diff, [2.5, 97.5])
    ci_a = np.percentile(boot_a, [2.5, 97.5])
    ci_b = np.percentile(boot_b, [2.5, 97.5])

    print()
    print(f"{name_a:<24} 95% CI [{ci_a[0]:.6f}, {ci_a[1]:.6f}]")
    print(f"{name_b:<24} 95% CI [{ci_b[0]:.6f}, {ci_b[1]:.6f}]")
    print(f"difference               95% CI [{lo:+.6f}, {hi:+.6f}]")
    crosses_zero = bool(lo <= 0.0 <= hi)
    print()
    if crosses_zero:
        print("VERDICT: the interval contains zero, so the two models are")
        print("         statistically indistinguishable on this protocol.")
        print("         Report parity, not a win.")
    else:
        side = name_a if lo > 0 else name_b
        print(f"VERDICT: the interval excludes zero; {side} is ahead.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "auc_a": full_a,
                    "auc_b": full_b,
                    "difference": full_a - full_b,
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "auc_a_ci_low": ci_a[0],
                    "auc_a_ci_high": ci_a[1],
                    "auc_b_ci_low": ci_b[0],
                    "auc_b_ci_high": ci_b[1],
                    "n_boot": args.n_boot,
                    "n_learners": len(shared),
                    "indistinguishable": crosses_zero,
                }
            ]
        ).to_csv(args.output, index=False)
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
