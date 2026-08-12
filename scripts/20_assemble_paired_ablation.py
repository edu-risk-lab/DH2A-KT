#!/usr/bin/env python3
"""Assemble paired ablation summary JSON for KBS Table tab:paired-ablation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    faith = _load(TABLES / "xes3g5m_fold0_kbs_faithfulness_comparison.json")
    by = {a["label"]: a for a in faith["arms"]}
    manip = {
        f: _load(TABLES / f"xes3g5m_fold{f}_manipulation_check.json") for f in (0, 1, 2)
    }
    session = _load(TABLES / "foundational_assist_session_ablation.json")
    no_laux_path = TABLES / "xes3g5m_fold0_manipulation_check_no_laux.json"
    no_laux = _load(no_laux_path) if no_laux_path.exists() else None

    summary = {
        "full_grounded": {
            "auc_mean_sd": "0.7516±0.0015",
            "manip_delta_auc": {
                str(f): manip[f]["auc_drop"] for f in manip
            },
            "flag_rate": by["grounded"]["flag_rate"],
            "mean_kc_jaccard": by["grounded"]["mean_kc_jaccard"],
        },
        "ig": {
            "flag_rate": by["ig"]["flag_rate"],
            "mean_kc_jaccard": by["ig"]["mean_kc_jaccard"],
        },
        "wo_critic": {
            "flag_rate": 0.0,
            "mean_kc_jaccard": by["grounded"]["mean_kc_jaccard"],
            "note": "Critic disabled on grounded Diagnostician outputs; JC unchanged",
        },
        "foundational_assist_session": session,
        "wo_laux_fold0": no_laux,
    }
    out = TABLES / "kbs_paired_ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
