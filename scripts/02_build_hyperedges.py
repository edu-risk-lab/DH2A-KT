#!/usr/bin/env python3
"""Pha 1 driver: build concept-prerequisite hyperedges from P0's audited
E_pre, then run the hyperedge leakage audit (Idea D section 2/6).

Usage:
    python scripts/02_build_hyperedges.py configs/xes3g5m.yaml --fold 0

Requires P0 preprocessing to already be done (see script 00 / P0's own
README section 3) — this script does not preprocess raw data itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dh2a_kt.hyperedge.construction import build_concept_prerequisite_hyperedges
from dh2a_kt.hyperedge.audit import audit_hyperedges
from dh2a_kt.p0_bridge import infer_prerequisites_from_train, audit_dag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(f"[fold {args.fold}] dataset={cfg['dataset']} role={cfg.get('role')}")
    print(
        "NOTE: this driver expects P0-preprocessed parquet + exported "
        "e_pre_train_only.csv for this fold to already exist under "
        "external/p0_leakage_audit/data/processed/ — see that repo's README "
        "section 4/5 for how to produce them. Wiring the exact load path is "
        "Pha 1 integration work (docs/idea-D-plan.md), left as a TODO here "
        "so this scaffold does not silently fabricate a data path."
    )
    raise NotImplementedError(
        "Pha 1 (docs/idea-D-plan.md): load this fold's audited E_pre from "
        "P0's export, then call build_concept_prerequisite_hyperedges() and "
        "audit_hyperedges(). Function-level pieces are implemented in "
        "dh2a_kt/hyperedge/; only this end-to-end driver's data-loading glue "
        "is left, pending a P0 preprocessing run on real data."
    )


if __name__ == "__main__":
    raise SystemExit(main())
