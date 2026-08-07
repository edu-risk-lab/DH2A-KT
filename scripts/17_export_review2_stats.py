#!/usr/bin/env python3
"""Export review-2 stats: AUC mean±SD and baseline manipulation comparison.

Reads local DH2 fold AUCs + P0 reused fold AUCs + P0 GKT DDR downstream
summary (node_drop p=0.9). Writes JSON under results/tables/ and prints
LaTeX-ready numbers for paper tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "tables" / "review2_auc_and_manipulation_stats.json"
P0_DDR = (
    REPO
    / "external"
    / "p0_leakage_audit"
    / "results"
    / "tables"
    / "ddr_downstream_summary.csv"
)
P0_FOLDS = REPO / "baselines" / "p0_reused" / "baseline_fold_results.csv"
DH2 = REPO / "results" / "tables" / "dh2_kt_vs_p0.csv"
DH2_MANIP = REPO / "results" / "tables" / "xes3g5m_fold0_manipulation_check.json"


def main() -> int:
    dh2 = pd.read_csv(DH2)
    dh2_folds = (
        dh2[dh2["fold"].astype(str).isin(["0", "1", "2"])]
        .drop_duplicates("fold")
        .sort_values("fold")
    )
    dh2_aucs = dh2_folds["dh2_kt_auc"].astype(float).to_numpy()

    folds = pd.read_csv(P0_FOLDS)
    folds = folds[
        (folds["dataset"] == "xes3g5m")
        & (folds["graph_construction"] == "train_only")
        & (folds["eval_split"] == "valid+test")
    ]
    gkt = folds[folds["model"] == "gkt"].sort_values("fold")["auc"].astype(float).to_numpy()
    skt = folds[folds["model"] == "simplekt"].sort_values("fold")["auc"].astype(float).to_numpy()

    ddr = pd.read_csv(P0_DDR)
    gkt_drop = ddr[
        (ddr["dataset"] == "xes3g5m")
        & (ddr["model"] == "gkt")
        & (ddr["operator"] == "node_drop")
        & (ddr["p"] == 0.9)
    ].iloc[0]

    manip = json.loads(DH2_MANIP.read_text(encoding="utf-8")) if DH2_MANIP.exists() else {}

    # Paired fold-wise delta vs GKT (same fold index)
    n = min(len(dh2_aucs), len(gkt), len(skt))
    delta_gkt = dh2_aucs[:n] - gkt[:n]
    delta_skt = dh2_aucs[:n] - skt[:n]

    payload = {
        "dh2_kt_auc_folds": dh2_aucs.tolist(),
        "dh2_kt_auc_mean": float(dh2_aucs.mean()),
        "dh2_kt_auc_sd": float(dh2_aucs.std(ddof=1)),
        "gkt_auc_folds": gkt.tolist(),
        "gkt_auc_mean": float(gkt.mean()),
        "gkt_auc_sd": float(gkt.std(ddof=1)),
        "simplekt_auc_folds": skt.tolist(),
        "simplekt_auc_mean": float(skt.mean()),
        "simplekt_auc_sd": float(skt.std(ddof=1)),
        "delta_vs_gkt_folds": delta_gkt.tolist(),
        "delta_vs_gkt_mean": float(delta_gkt.mean()),
        "delta_vs_gkt_sd": float(delta_gkt.std(ddof=1)),
        "delta_vs_simplekt_mean": float(delta_skt.mean()),
        "delta_vs_simplekt_sd": float(delta_skt.std(ddof=1)),
        "manipulation": {
            "dh2_kt_fold0": {
                "auc_clean": manip.get("auc_clean"),
                "auc_destroyed": manip.get("auc_destroyed"),
                "auc_drop": manip.get("auc_drop"),
                "ddr": manip.get("ddr"),
                "protocol": "train_clean_eval_destroyed_hyperedges",
            },
            "gkt_p0_xes3g5m_node_drop_p0.9": {
                "auc_drop_mean": float(gkt_drop["auc_drop_mean"]),
                "auc_drop_std": float(gkt_drop["auc_drop_std"]),
                "ddr_mean": float(gkt_drop["ddr_mean"]),
                "auc_mean_destroyed": float(gkt_drop["auc_mean"]),
                "n": int(gkt_drop["n"]),
                "protocol": "p0_ddr_downstream_retrain_on_destroyed_graph",
                "source": str(P0_DDR.relative_to(REPO)),
            },
            "simplekt": {
                "auc_drop": None,
                "note": (
                    "Sequence model without concept-graph adjacency; "
                    "graph node-drop destruction is undefined (N/A by architecture)."
                ),
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
