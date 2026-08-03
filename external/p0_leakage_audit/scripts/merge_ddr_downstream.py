#!/usr/bin/env python3
"""Merge/resume-safe combine of DDR downstream CSV shards into the paper bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY_COLS = ["dataset", "model", "fold", "split_seed", "operator", "p"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "results/tables/ddr_downstream.csv",
        help="Existing paper CSV (kept if --append targets missing rows only).",
    )
    parser.add_argument(
        "--append",
        type=Path,
        action="append",
        default=[],
        help="One or more CSV files to merge (e.g. results/q1/ddr_downstream_gkt/ddr_downstream_gkt.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite --base).",
    )
    args = parser.parse_args()
    out = args.out or args.base

    frames: list[pd.DataFrame] = []
    if args.base.exists():
        frames.append(pd.read_csv(args.base))
    for path in args.append:
        if not path.exists():
            raise SystemExit(f"Missing shard: {path}")
        frames.append(pd.read_csv(path))

    if not frames:
        raise SystemExit("Nothing to merge; provide --base and/or --append.")

    df = pd.concat(frames, ignore_index=True)
    for col in KEY_COLS:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col!r} in merged frame.")

    if "p" in df.columns:
        df["p"] = df["p"].astype(float).round(4)
    df = df.drop_duplicates(subset=KEY_COLS, keep="last").sort_values(KEY_COLS)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    if "model" in df.columns:
        print(df.groupby(["dataset", "model"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
