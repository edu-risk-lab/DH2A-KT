#!/usr/bin/env python3
"""GS Hau A4: hyperedge audit positive control (train-only vs full-log E_pre).

Builds chain hyperedges from (a) P0 train-only ``e_pre_train_only.csv`` and
(b) leaky full-log ``full_log/e_pre.csv``, then audits both under the same
held-out learner protocol. The contaminated arm uses a concept→interaction map
from the full log so group-membership leakage can register non-zero signal.

Usage:
    python scripts/29_a4_audit_positive_control.py configs/xes3g5m.yaml --fold 0
    python scripts/29_a4_audit_positive_control.py configs/xes3g5m.yaml --fold 0 --sample-pairs 100000
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

from dh2a_kt.hyperedge.audit import (  # noqa: E402
    HyperedgeLeakageReport,
    _PAIRWISE_DIAGNOSTIC_MAX_ROWS,
    _flatten_to_pairwise,
    audit_hyperedges,
    compute_group_membership_leakage,
)
from dh2a_kt.hyperedge.construction import build_concept_prerequisite_hyperedges  # noqa: E402
from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    build_concept_member_interaction_map,
    e_pre_export_path,
    get_fold_splits,
    held_out_interaction_ids,
    load_configs,
    load_e_pre,
    load_interactions_with_ids,
)
from dh2a_kt.p0_bridge import compute_ecr_flag, compute_rho_edge_outcome, compute_tbvr  # noqa: E402

P0_ROOT = REPO_ROOT / "external" / "p0_leakage_audit"


def _full_log_e_pre_path(p0_cfg: dict) -> Path:
    dataset = p0_cfg["dataset"]
    return P0_ROOT / "data" / "processed" / dataset / "full_log" / "e_pre.csv"


def _load_full_log_e_pre(p0_cfg: dict) -> pd.DataFrame:
    path = _full_log_e_pre_path(p0_cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing full-log E_pre: {path}\n"
            "Run: cd external/p0_leakage_audit && "
            "python -m src.export_full_log_graph --config configs/xes3g5m.yaml"
        )
    df = pd.read_csv(path)
    for col in ("src_kc", "dst_kc"):
        if col not in df.columns:
            raise ValueError(f"{path} missing column {col!r}")
    return df


def _all_interactions_df(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [splits[k] for k in ("train", "valid", "test") if k in splits and not splits[k].empty]
    return pd.concat(parts, ignore_index=True)


def _sampled_pairwise_diagnostics(
    hyperedges: list,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    sample_pairs: int,
    seed: int,
) -> tuple[float, float | None, str]:
    """Return (tbmr, |rho|, note) on a random pairwise subsample."""
    pairwise = _flatten_to_pairwise(hyperedges)
    n = len(pairwise)
    if n == 0:
        return 0.0, None, "no pairwise pairs"
    if n > sample_pairs:
        pairwise = pairwise.sample(n=sample_pairs, random_state=seed).reset_index(drop=True)
        note = f"TBMR/|rho| on random sample of {sample_pairs:,} / {n:,} projected pairs (seed={seed})"
    else:
        note = f"TBMR/|rho| on all {n:,} projected pairs"
    tbmr = compute_tbvr(train_df, pairwise[["src_kc", "dst_kc"]])
    rho = compute_rho_edge_outcome(
        pairwise.assign(weight=1.0),
        pd.DataFrame(columns=["src_kc", "dst_kc", "weight"]),
        test_df,
    )
    return tbmr, rho, note


def _audit_arm(
    hyperedges: list,
    *,
    splits: dict[str, pd.DataFrame],
    member_map: dict,
    held_out: set[int],
    sample_pairs: int | None,
    seed: int,
    label: str,
) -> dict:
    base = audit_hyperedges(
        hyperedges,
        splits=splits,
        train_df=splits["train"],
        test_df=splits.get("test"),
        held_out_interaction_ids=held_out,
        member_to_interaction_id=member_map,
    )
    out = asdict(base)
    out["arm"] = label
    out["n_hyperedges"] = len(hyperedges)

    if sample_pairs and sample_pairs > 0:
        n_pairwise = sum(len(he.members) * (len(he.members) - 1) // 2 for he in hyperedges)
        if n_pairwise > _PAIRWISE_DIAGNOSTIC_MAX_ROWS:
            tbmr, rho, note = _sampled_pairwise_diagnostics(
                hyperedges,
                splits["train"],
                splits["test"],
                sample_pairs=sample_pairs,
                seed=seed,
            )
            out["tbmr"] = tbmr
            out["rho"] = rho
            out["notes"] = list(out.get("notes", [])) + [note]

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=100_000,
        help="Subsample size for TBMR/|rho| when projection exceeds the cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "xes3g5m_fold0_hyperedge_audit_positive_control.json",
    )
    args = parser.parse_args()

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    fold = args.fold
    he_cfg = dh2_cfg.get("hyperedge", {}).get("concept_prerequisite", {})
    min_chain = int(he_cfg.get("min_chain_len", 3))
    max_chain = int(he_cfg.get("max_chain_len", 8))

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, fold)
    held_out = held_out_interaction_ids(splits)
    all_df = _all_interactions_df(splits)

    e_pre_train = load_e_pre(p0_cfg, fold)
    e_pre_full = _load_full_log_e_pre(p0_cfg)

    he_train = build_concept_prerequisite_hyperedges(
        e_pre_train, fold=fold, min_chain_len=min_chain, max_chain_len=max_chain
    )
    he_full = build_concept_prerequisite_hyperedges(
        e_pre_full, fold=fold, min_chain_len=min_chain, max_chain_len=max_chain
    )

    train_arm = _audit_arm(
        he_train,
        splits=splits,
        member_map=build_concept_member_interaction_map(splits["train"]),
        held_out=held_out,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
        label="train_only",
    )
    contaminated_arm = _audit_arm(
        he_full,
        splits=splits,
        member_map=build_concept_member_interaction_map(all_df),
        held_out=held_out,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
        label="full_log_positive_control",
    )

    payload = {
        "dataset": dataset,
        "fold": fold,
        "e_pre_train_only": str(e_pre_export_path(p0_cfg, fold)),
        "e_pre_full_log": str(_full_log_e_pre_path(p0_cfg)),
        "n_edges_train_only": len(e_pre_train),
        "n_edges_full_log": len(e_pre_full),
        "sample_pairs": args.sample_pairs,
        "seed": args.seed,
        "train_only": train_arm,
        "full_log_positive_control": contaminated_arm,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.out}")
    for key in ("train_only", "full_log_positive_control"):
        arm = payload[key]
        print(
            f"  [{key}] n_he={arm['n_hyperedges']:,} "
            f"group_leak={arm['group_membership_leak_rate']:.6f} "
            f"tbmr={arm['tbmr']:.6f} rho={arm['rho']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
