"""How much of P0's baseline AUC is computed on padding?

``_diag_pykt_smask.py`` established that P0's pyKT export scores padded
positions and that every padded target carries label 0. This script measures how
many such positions the real xes3g5m fold-0 evaluation contains.

P0 pads every learner to ``max_seq_len`` and pyKT scores ``max_seq_len - 1``
positions per learner regardless of the true log length, so a learner with 20
interactions contributes 19 real targets and 180 padded ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids

MAX_SEQ_LEN = 200


def report(name: str, df: pd.DataFrame) -> dict:
    lens = df.groupby("user_id").size()
    lens = lens[lens >= 2]  # P0's _build_rows skips learners with fewer than 2 rows
    eff = lens.clip(upper=MAX_SEQ_LEN)

    n_users = int(lens.shape[0])
    scored = n_users * (MAX_SEQ_LEN - 1)
    real = int((eff - 1).sum())
    padded = scored - real

    # Label balance among the real targets: padded targets are all 0, so the
    # padded block is a mass of negatives added to whatever the real mix is.
    real_mean = float(df.groupby("user_id").tail(MAX_SEQ_LEN)["correct"].mean())

    print("-" * 72)
    print(f"[{name}]")
    print(f"  learners exported            : {n_users:,}")
    print(f"  positions pyKT scores        : {scored:,}")
    print(f"  of which real                : {real:,} ({real / scored:.2%})")
    print(f"  of which PADDING (label 0)   : {padded:,} ({padded / scored:.2%})")
    print(f"  mean correct on real rows    : {real_mean:.4f}")
    print(f"  median log length            : {lens.median():.0f}")
    return {"scored": scored, "real": real, "padded": padded, "real_mean": real_mean}


def main() -> None:
    _dh2, p0, _ = load_configs(Path("configs/xes3g5m.yaml"))
    inter = load_interactions_with_ids(p0)
    splits = get_fold_splits(inter, p0, 0)

    print("=" * 72)
    print("PADDING SHARE OF P0's pyKT BASELINE EVALUATION (xes3g5m fold 0)")
    print("=" * 72)

    test = report("TEST  (what run_pykt_fold scores by default)", splits["test"])
    both = report(
        "VALID+TEST  (what the results CSV claims)",
        pd.concat([splits["valid"], splits["test"]], ignore_index=True),
    )

    print()
    print("=" * 72)
    print("WHY THIS INFLATES AUC")
    print("=" * 72)
    pad_share = test["padded"] / test["scored"]
    pos_rate = test["real_mean"]
    print(f"Padded targets are all label 0 and carry concept id 0 with response 0,")
    print(f"so the model sees a run of failures on one concept and predicts a low")
    print(f"probability there. Real targets are {pos_rate:.1%} positive. AUC counts")
    print(f"correctly ordered (positive, negative) pairs, and the padded block adds")
    print(f"{pad_share:.1%} of all scored positions as negatives that are trivially")
    print(f"separable from the real positives.")
    print()
    print(f"n_eval reported in baseline_fold_results.csv : 1,922,840")
    print(f"positions actually scored (test)             : {test['scored']:,}")
    print(f"positions actually scored (valid+test)       : {both['scored']:,}")
    print("The reported n_eval is len(eval_df), not the number of scored")
    print("positions, so neither figure matches the published column.")


if __name__ == "__main__":
    main()
