"""Graded positive control for the hyperedge group-membership diagnostic.

The full-log control in script 29 maps every concept member to the whole
log, so the leak rate reaches 1 almost by construction. Here the train-only
chain hyperedges are kept fixed and only the member->interaction map is
contaminated: a fraction p of held-out learners is added to the map, as if
those learners had slipped into graph construction. For each p the script
reports

  * group-membership leak rate (share of hyperedges with any tainted member);
  * held-out support share (mean over hyperedges of the share of supporting
    interactions that are held-out), which is continuous in p.

Writes results/tables/xes3g5m_fold0_partial_membership_control.json.

    python scripts/53_partial_membership_control.py configs/xes3g5m.yaml --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dh2a_kt.hyperedge.audit import compute_group_membership_leakage  # noqa: E402
from dh2a_kt.hyperedge.construction import build_concept_prerequisite_hyperedges  # noqa: E402
from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    build_concept_member_interaction_map,
    get_fold_splits,
    held_out_interaction_ids,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)

FRACTIONS = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)


def _support_share(hyperedges: list, built_from: pd.DataFrame, held_out: set[int]) -> float:
    # Each interaction row carries one kc_id, so concept supports are disjoint
    # and a hyperedge's held-out share is a ratio of summed per-concept counts.
    is_held = built_from["interaction_id"].isin(held_out)
    total = {int(k): int(v) for k, v in built_from.groupby("kc_id").size().items()}
    held = {int(k): int(v) for k, v in is_held.groupby(built_from["kc_id"]).sum().items()}
    shares = []
    for he in hyperedges:
        kcs = [int(m[1]) for m in he.members if m[0] == "concept"]
        n = sum(total.get(k, 0) for k in kcs)
        if n > 0:
            shares.append(sum(held.get(k, 0) for k in kcs) / n)
    return float(np.mean(shares)) if shares else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dh2_cfg, p0_cfg, _ = load_configs(REPO / args.config)
    dataset = dh2_cfg["dataset"]
    he_cfg = dh2_cfg.get("hyperedge", {}).get("concept_prerequisite", {})
    splits = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, args.fold)
    held_out = held_out_interaction_ids(splits)
    hyperedges = build_concept_prerequisite_hyperedges(
        load_e_pre(p0_cfg, args.fold),
        fold=args.fold,
        min_chain_len=int(he_cfg.get("min_chain_len", 3)),
        max_chain_len=int(he_cfg.get("max_chain_len", 8)),
    )
    held_df = pd.concat([splits[k] for k in ("valid", "test") if k in splits], ignore_index=True)
    held_users = np.sort(held_df["user_id"].unique())
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(held_users)

    rows = []
    for p in FRACTIONS:
        n_leak = int(round(p * len(order)))
        leaked = set(order[:n_leak].tolist())
        built_from = pd.concat(
            [splits["train"], held_df[held_df["user_id"].isin(leaked)]], ignore_index=True
        )
        member_map = build_concept_member_interaction_map(built_from)
        rate = compute_group_membership_leakage(hyperedges, held_out, member_map)
        share = _support_share(hyperedges, built_from, held_out)
        rows.append(
            {
                "fraction_heldout_learners": p,
                "n_leaked_learners": n_leak,
                "group_membership_leak_rate": rate,
                "heldout_support_share": share,
            }
        )
        print(f"p={p:<7} learners={n_leak:>6} leak_rate={rate:.4f} support_share={share:.6f}")

    out = args.out or REPO / "results" / "tables" / f"{dataset}_fold{args.fold}_partial_membership_control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "fold": args.fold,
                "seed": args.seed,
                "n_hyperedges": len(hyperedges),
                "n_heldout_learners": int(len(order)),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
