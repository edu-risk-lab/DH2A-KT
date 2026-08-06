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
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.hyperedge.audit import audit_hyperedges
from dh2a_kt.hyperedge.construction import build_concept_prerequisite_hyperedges
from dh2a_kt.hyperedge.p0_inputs import (
    build_concept_member_interaction_map,
    e_pre_export_path,
    get_fold_splits,
    held_out_interaction_ids,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)


def _hyperedge_summary(hyperedges: list, *, sample: int = 5) -> dict:
    chain_lens = [len(he.members) for he in hyperedges]
    preview = [
        {
            "hyperedge_id": he.hyperedge_id,
            "chain_len": len(he.members),
            "members": he.members,
        }
        for he in hyperedges[:sample]
    ]
    return {
        "n_hyperedges": len(hyperedges),
        "chain_len_min": min(chain_lens) if chain_lens else 0,
        "chain_len_max": max(chain_lens) if chain_lens else 0,
        "preview_first_n": preview,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="DH2A-KT config, e.g. configs/xes3g5m.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tables",
        help="Directory for audit JSON + hyperedge summary JSON",
    )
    args = parser.parse_args()

    dh2_cfg, p0_cfg, p0_config_path = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    he_cfg = dh2_cfg.get("hyperedge", {}).get("concept_prerequisite", {})
    if not he_cfg.get("enabled", True):
        print("concept_prerequisite hyperedge disabled in config — nothing to do.")
        return 0

    fold = args.fold
    e_pre_path = e_pre_export_path(p0_cfg, fold)
    print(f"[fold {fold}] dataset={dataset} role={dh2_cfg.get('role')}")
    print(f"  p0_config: {p0_config_path}")
    print(f"  e_pre:     {e_pre_path}")

    e_pre = load_e_pre(p0_cfg, fold)
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, fold)

    hyperedges = build_concept_prerequisite_hyperedges(
        e_pre,
        fold=fold,
        min_chain_len=int(he_cfg.get("min_chain_len", 3)),
        max_chain_len=int(he_cfg.get("max_chain_len", 8)),
    )
    report = audit_hyperedges(
        hyperedges,
        splits=splits,
        train_df=splits["train"],
        test_df=splits.get("test"),
        held_out_interaction_ids=held_out_interaction_ids(splits),
        member_to_interaction_id=build_concept_member_interaction_map(splits["train"]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / f"{dataset}_fold{fold}_hyperedge_audit.json"
    summary_path = args.out_dir / f"{dataset}_fold{fold}_concept_prereq_hyperedges.json"
    audit_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(_hyperedge_summary(hyperedges), indent=2), encoding="utf-8")

    print(f"  hyperedges built: {report.n_hyperedges}")
    print(f"  ecr_flag:         {report.ecr_flag}")
    print(f"  tbmr:             {report.tbmr:.6f}")
    print(f"  group_leak_rate:  {report.group_membership_leak_rate:.6f}")
    print(f"  audit written:    {audit_path}")
    print(f"  summary written:  {summary_path}")

    if report.n_hyperedges == 0:
        print("FAIL: no hyperedges produced.", file=sys.stderr)
        return 1
    if report.ecr_flag != 0.0:
        print("FAIL: structural leakage flag set (ecr_flag != 0).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
