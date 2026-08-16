"""The clean-protocol pyKT export must control exactly what pyKT scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dh2a_kt.baselines.pykt_clean import export_pykt_sequences, repeat_flags, windows


def _logs(user_id: int, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [user_id] * n,
            "item_id": list(range(n)),
            "kc_id": list(range(n)),
            "correct": [1] * n,
            "timestamp": list(range(n)),
        }
    )


def _multi_kc_logs() -> pd.DataFrame:
    """A learner whose 2nd question covers 3 concepts, as pyKT's KC-level export writes it."""
    rows = [
        {"user_id": 0, "item_id": 10, "kc_id": 0, "timestamp": 100, "correct": 1},
        {"user_id": 0, "item_id": 11, "kc_id": 1, "timestamp": 200, "correct": 0},
        {"user_id": 0, "item_id": 11, "kc_id": 2, "timestamp": 200, "correct": 0},
        {"user_id": 0, "item_id": 11, "kc_id": 3, "timestamp": 200, "correct": 0},
        {"user_id": 0, "item_id": 12, "kc_id": 4, "timestamp": 300, "correct": 1},
    ]
    return pd.DataFrame(rows)


def test_last_window_reproduces_p0_tail_slice_and_first_does_not():
    assert windows(25, 10, "last") == [(15, 25)]
    assert windows(25, 10, "first") == [(0, 10)]
    assert windows(25, 10, "chunked") == [(0, 10), (10, 20), (20, 25)]
    assert windows(4, 10, "last") == [(0, 4)], "a short learner is not clipped"
    assert windows(1, 10, "last") == [], "P0 skips learners with fewer than 2 rows"


def test_repeat_flags_only_mark_continuations_of_the_same_attempt():
    df = _multi_kc_logs()
    flags = repeat_flags(
        df["user_id"].to_numpy(), df["item_id"].to_numpy(), df["timestamp"].to_numpy()
    )
    assert flags.tolist() == [False, False, True, True, False]


def _read_selectmasks(path) -> list[list[str]]:
    return [row.split(",") for row in pd.read_csv(path)["selectmasks"]]


def test_repeat_rows_are_written_as_pykt_pad_value(tmp_path):
    df = _multi_kc_logs()
    q_map = {int(q): i for i, q in enumerate(sorted(df["item_id"].unique()))}
    c_map = {int(c): i for i, c in enumerate(sorted(df["kc_id"].unique()))}
    common = dict(
        train_df=df, valid_df=df, test_df=df, q_map=q_map, c_map=c_map, max_seq_len=5,
        window_mode="last",
    )

    leaky = export_pykt_sequences(out_dir=tmp_path / "leaky", mask_repeats=False, **common)
    clean = export_pykt_sequences(out_dir=tmp_path / "clean", mask_repeats=True, **common)

    assert _read_selectmasks(tmp_path / "leaky" / "test_sequences.csv") == [["1"] * 5]
    assert _read_selectmasks(tmp_path / "clean" / "test_sequences.csv") == [
        ["1", "1", "-1", "-1", "1"]
    ]
    assert leaky["test_positions_scored"] == 4
    assert clean["test_positions_scored"] == 2, "the two repeat targets are dropped"


def test_pykt_scores_exactly_the_unmasked_positions(tmp_path):
    """End-to-end: pyKT's own KTDataset must agree with what we intended to score."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pykt")
    import pykt.datasets.data_loader as dl

    dl.LongTensor = torch.LongTensor
    dl.FloatTensor = torch.FloatTensor
    from pykt.datasets.data_loader import KTDataset

    df = _multi_kc_logs()
    q_map = {int(q): i for i, q in enumerate(sorted(df["item_id"].unique()))}
    c_map = {int(c): i for i, c in enumerate(sorted(df["kc_id"].unique()))}
    out = tmp_path / "clean"
    export_pykt_sequences(
        train_df=df, valid_df=df, test_df=df, q_map=q_map, c_map=c_map,
        out_dir=out, max_seq_len=5, window_mode="last", mask_repeats=True,
    )

    ds = KTDataset(str(out / "test_sequences.csv"), ["questions", "concepts"], {-1})
    item = ds[0]
    sm = item["smasks"].bool()
    # selectmasks = 1,1,-1,-1,1 and pyKT scores selectmasks[1:], so of the four
    # targets only the 1st and 4th survive: concepts 1 then 4 in the source log.
    assert sm.tolist() == [True, False, False, True]
    assert item["shft_cseqs"][sm].tolist() == [c_map[1], c_map[4]]


def test_padding_is_excluded_unlike_p0s_export(tmp_path):
    """P0 pads selectmasks with "0", which pyKT reads as a scored position."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pykt")
    import pykt.datasets.data_loader as dl

    dl.LongTensor = torch.LongTensor
    dl.FloatTensor = torch.FloatTensor
    from pykt.datasets.data_loader import KTDataset

    df = _logs(0, 4)
    q_map = {int(q): i for i, q in enumerate(sorted(df["item_id"].unique()))}
    c_map = {int(c): i for i, c in enumerate(sorted(df["kc_id"].unique()))}
    out = tmp_path / "padded"
    export_pykt_sequences(
        train_df=df, valid_df=df, test_df=df, q_map=q_map, c_map=c_map,
        out_dir=out, max_seq_len=10, window_mode="last", mask_repeats=False,
    )

    ds = KTDataset(str(out / "test_sequences.csv"), ["questions", "concepts"], {-1})
    sm = ds[0]["smasks"].bool()
    assert int(sm.sum()) == 3, "4 real rows give 3 targets; the 6 padded ones must not count"


def test_chunked_export_covers_every_interaction(tmp_path):
    df = pd.concat([_logs(0, 25), _logs(1, 12)], ignore_index=True)
    q_map = {int(q): i for i, q in enumerate(sorted(df["item_id"].unique()))}
    c_map = {int(c): i for i, c in enumerate(sorted(df["kc_id"].unique()))}
    stats = {}
    for mode in ("last", "chunked"):
        stats[mode] = export_pykt_sequences(
            train_df=df, valid_df=df, test_df=df, q_map=q_map, c_map=c_map,
            out_dir=tmp_path / mode, max_seq_len=10, window_mode=mode, mask_repeats=False,
        )

    # last: one window per learner, min(25,10)-1 + min(12,10)-1 = 9 + 9
    assert stats["last"]["test_positions_scored"] == 18
    # chunked: windows (10,10,5) and (10,2) => 9+9+4 + 9+1
    assert stats["chunked"]["test_positions_scored"] == 32
    assert stats["chunked"]["test_sequences"] == 5
