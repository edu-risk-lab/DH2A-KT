#!/usr/bin/env python3
"""Build session hyperedges (with Hint members) for FoundationalASSIST.

Usage:
    python scripts/13_build_session_hyperedges.py configs/foundational_assist.yaml --fold 0

Requires ``scripts/12_ingest_foundational_assist.py`` to have written the
extended parquet + train-only E_pre.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.hyperedge.audit import audit_hyperedges
from dh2a_kt.hyperedge.construction import build_session_hyperedges
from dh2a_kt.hyperedge.p0_inputs import (
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)


def _load_extended(dh2_cfg: dict) -> pd.DataFrame:
    rel = dh2_cfg.get("extended_processed_path")
    if not rel:
        raise ValueError("config missing extended_processed_path for session HE")
    path = REPO_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun scripts/12_ingest_foundational_assist.py first."
        )
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True)
    df["interaction_id"] = df.index.astype("int64")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tables",
    )
    parser.add_argument("--max-sessions", type=int, default=None, help="Cap for smoke tests")
    args = parser.parse_args()

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    sess_cfg = dh2_cfg.get("hyperedge", {}).get("session", {})
    if not sess_cfg.get("enabled", False):
        print("session hyperedge disabled in config — nothing to do.")
        return 0

    fold = args.fold
    dataset = dh2_cfg["dataset"]
    gap = float(sess_cfg.get("session_gap_seconds", 1800))

    # Splits from canonical interactions (same user folds as training).
    canonical = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(canonical, p0_cfg, fold)
    train_users = set(splits["train"]["user_id"].astype("int64").tolist())

    extended = _load_extended(dh2_cfg)
    train_ext = extended[extended["user_id"].isin(train_users)].copy()
    print(
        f"[fold {fold}] dataset={dataset} train_rows={len(train_ext)} "
        f"hint_rate={train_ext['hint_used'].mean():.3f}"
        if "hint_used" in train_ext.columns
        else f"[fold {fold}] dataset={dataset} train_rows={len(train_ext)}"
    )

    hyperedges = build_session_hyperedges(
        train_ext,
        fold=fold,
        session_gap_seconds=gap,
        train_only=True,
    )
    if args.max_sessions is not None:
        hyperedges = hyperedges[: args.max_sessions]

    n_with_hint = sum(1 for he in hyperedges if any(t == "hint" for t, _ in he.members))
    report = audit_hyperedges(
        hyperedges,
        splits=splits,
        train_df=splits["train"],
        test_df=splits.get("test"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset,
        "fold": fold,
        "session_gap_seconds": gap,
        "n_session_hyperedges": len(hyperedges),
        "n_with_hint_member": n_with_hint,
        "hint_member_rate": (n_with_hint / len(hyperedges)) if hyperedges else 0.0,
        "preview": [
            {
                "hyperedge_id": he.hyperedge_id,
                "n_members": len(he.members),
                "member_types": sorted({t for t, _ in he.members}),
                "provenance": he.provenance,
            }
            for he in hyperedges[:5]
        ],
    }
    summary_path = args.out_dir / f"{dataset}_fold{fold}_session_hyperedges.json"
    audit_path = args.out_dir / f"{dataset}_fold{fold}_session_hyperedge_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"  n={len(hyperedges)} with_hint={n_with_hint} ({summary['hint_member_rate']:.3f})")
    print(
        f"Wrote {audit_path} ecr_flag={report.ecr_flag:.4f} "
        f"group_leak={report.group_membership_leak_rate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
