#!/usr/bin/env python3
"""Pha 2-3 driver (docs/idea-D-plan.md): train DH2-KT and compare against
P0's reused baselines. NotImplementedError until dh2a_kt/models/dh2_kt.py's
forward() is implemented (Pha 2) — see that module's docstring.

Once DH2KT.forward() is real, this script's job is:
  1. train DH2KT on the configured dataset under a P0-matched epoch/batch
     budget (see baselines_reused + P0's configs/*.yaml `pykt:`/`baselines:`
     sections for the exact budgets to match, per docs/idea-D-plan.md
     section 9 "Ngan sach compute & diem kiem tra").
  2. compare_against_p0() from dh2a_kt.baselines.loader for the headline table.
  3. run dh2a_kt.eval.manipulation_check.run_manipulation_check() before
     trusting any ablation result.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.parse_args()
    raise NotImplementedError(
        "Pha 2-3 (docs/idea-D-plan.md): implement dh2a_kt/models/dh2_kt.py "
        "DH2KT.forward() first, then wire the train/eval loop here."
    )


if __name__ == "__main__":
    raise SystemExit(main())
