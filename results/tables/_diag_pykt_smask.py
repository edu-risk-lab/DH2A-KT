"""Does P0's pyKT export score padded positions?

P0's ``pykt_export._build_rows`` writes ``"0"`` for the ``selectmasks`` padding,
while pyKT builds the select mask as ``selectmasks[:, 1:] != -1``. If ``0`` is
not the pad value pyKT expects, every padded position enters the loss and the
AUC. This script builds a tiny export and reads back what pyKT actually scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

P0 = Path("external/p0_leakage_audit").resolve()
sys.path.insert(0, str(P0))

from src.pykt_export import build_dense_maps, dataframe_to_pykt_csvs  # noqa: E402

import pykt.datasets.data_loader as dl  # noqa: E402

dl.LongTensor = torch.LongTensor
dl.FloatTensor = torch.FloatTensor
from pykt.datasets.data_loader import KTDataset  # noqa: E402

MAX_SEQ_LEN = 10


def logs(user_id: int, n: int, first_kc: int) -> pd.DataFrame:
    """One learner with ``n`` interactions, all answered correctly (label 1)."""
    return pd.DataFrame(
        {
            "user_id": [user_id] * n,
            "item_id": list(range(n)),
            "kc_id": [first_kc + i for i in range(n)],
            "correct": [1] * n,
            "timestamp": list(range(n)),
        }
    )


def main() -> None:
    # Every real label is 1, so any 0 that shows up as a scored target is padding.
    train = pd.concat([logs(1, MAX_SEQ_LEN, 0), logs(2, MAX_SEQ_LEN, 0)], ignore_index=True)
    valid = logs(3, MAX_SEQ_LEN, 0)
    test_len = 4
    test = logs(4, test_len, 0)

    out = Path("temp/_diag_pykt_smask")
    for stale in ("train_valid_sequences.csv", "test_sequences.csv"):
        (out / stale).unlink(missing_ok=True)
    for stale in out.glob("*.pkl"):
        stale.unlink()
    q_map, c_map = build_dense_maps(train)
    dataframe_to_pykt_csvs(
        train_df=train,
        valid_df=valid,
        test_df=test,
        q_map=q_map,
        c_map=c_map,
        out_dir=out,
        max_seq_len=MAX_SEQ_LEN,
    )

    row = pd.read_csv(out / "test_sequences.csv").iloc[0]
    print(f"max_seq_len          = {MAX_SEQ_LEN}")
    print(f"real interactions    = {test_len}")
    print(f"selectmasks exported = {row['selectmasks']}")
    print(f"concepts exported    = {row['concepts']}")
    print(f"responses exported   = {row['responses']}")

    ds = KTDataset(str(out / "test_sequences.csv"), ["questions", "concepts"], {-1})
    item = ds[0]
    sm = item["smasks"].bool()
    rshft = item["shft_rseqs"]
    cshft = item["shft_cseqs"]

    scored = int(sm.sum())
    honest = test_len - 1
    print()
    print(f"positions pyKT scores    = {scored}")
    print(f"positions that are real  = {honest}")
    print(f"padded positions scored  = {scored - honest}")
    print()
    print(f"scored targets  = {rshft[sm].tolist()}")
    print(f"scored concepts = {cshft[sm].tolist()}")

    if scored > honest:
        extra = rshft[sm][honest:]
        share_zero = float((extra == 0).float().mean()) if extra.numel() else float("nan")
        print()
        print("VERDICT: padding IS scored.")
        print(f"  {scored - honest} of {scored} scored targets are padding "
              f"({(scored - honest) / scored:.1%})")
        print(f"  share of padded targets equal to 0 (label 'incorrect') = {share_zero:.4f}")
        print("  Every real label here is 1, so these are free negatives:")
        print("  a constant predictor already separates them from the real positives.")
    else:
        print()
        print("VERDICT: padding is correctly excluded.")


if __name__ == "__main__":
    main()
