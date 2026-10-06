"""Descriptive statistics of the start-to-start gap on the three corpora.

No model is trained. For learner fold 0, train split, rows sorted as in
``UserSequenceDataset`` (user, timestamp, item), reports:

  * share of non-first rows with dt = 0, split into repeat KC-rows of the
    same attempt and zero gaps between distinct items;
  * median and upper quantiles of positive gaps (seconds);
  * single-feature AUC of log(1+dt) for correctness on non-repeat rows;
  * lag-1 correlation of log(1+dt) within learners (pace persistence,
    which is what T-misaligned keeps per learner and T-shuffled destroys).

Writes results/tables/gap_descriptives.json and
paper/tables/table_gap_descriptives.tex.

    python scripts/52_gap_descriptives.py
    python scripts/52_gap_descriptives.py --configs configs/xes3g5m.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dh2a_kt.data.aux_signals import infer_timestamp_seconds_scale  # noqa: E402
from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)

DEFAULT_CONFIGS = ("configs/xes3g5m.yaml", "configs/assist2012.yaml", "configs/junyi.yaml")
LABELS = {"xes3g5m": "XES3G5M", "assist2012": "ASSIST2012", "junyi": "Junyi"}
OUT_JSON = REPO / "results" / "tables" / "gap_descriptives.json"
OUT_TEX = REPO / "paper" / "tables" / "table_gap_descriptives.tex"
# Junyi time_done is unix microseconds; the magnitude rule in
# infer_timestamp_seconds_scale reads it as seconds. The tracer inputs keep that
# rule (frozen runs); only these descriptives are converted.
SECONDS_PER_UNIT = {"junyi": 1.0e-6}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label == 1
    n_pos = int(pos.sum())
    n_neg = len(label) - n_pos
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def describe(config: Path, fold: int) -> dict:
    dh2_cfg, p0_cfg, _ = load_configs(config)
    splits = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, fold)
    df = splits["train"].sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    users = df["user_id"].to_numpy()
    items = df["item_id"].to_numpy()
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    y = df["correct"].to_numpy()
    tracer_scale = infer_timestamp_seconds_scale(ts)
    scale = SECONDS_PER_UNIT.get(dh2_cfg["dataset"], tracer_scale)

    same_user = np.zeros(len(df), dtype=bool)
    same_user[1:] = users[1:] == users[:-1]
    raw = np.zeros(len(df), dtype=np.float64)
    raw[1:] = np.maximum((ts[1:] - ts[:-1]).astype(np.float64) * scale, 0.0)
    repeat = np.zeros(len(df), dtype=bool)
    repeat[1:] = same_user[1:] & (items[1:] == items[:-1]) & (ts[1:] == ts[:-1])

    body = same_user
    zero = body & (raw == 0)
    pos_gap = raw[body & (raw > 0)]
    scored = body & ~repeat
    log_gap = np.log1p(raw)

    nxt = np.zeros(len(df), dtype=bool)
    nxt[1:] = scored[1:] & scored[:-1] & same_user[1:]
    lag_corr = float(np.corrcoef(log_gap[:-1][nxt[1:]], log_gap[1:][nxt[1:]])[0, 1])

    per_user = df.groupby("user_id").size().to_numpy()
    return {
        "dataset": dh2_cfg["dataset"],
        "fold": fold,
        "n_rows": int(len(df)),
        "n_learners": int(len(per_user)),
        "rows_per_learner_median": float(np.median(per_user)),
        "share_zero_gap": float(zero.sum() / body.sum()),
        "share_zero_repeat": float((zero & repeat).sum() / body.sum()),
        "share_zero_distinct_item": float((zero & ~repeat).sum() / body.sum()),
        "positive_gap_median_s": float(np.median(pos_gap)),
        "positive_gap_p90_s": float(np.quantile(pos_gap, 0.9)),
        "share_gap_over_1day": float((raw[body] > 86400).mean()),
        "auc_log_gap_correct": _auc(log_gap[scored], y[scored]),
        "lag1_corr_log_gap": lag_corr,
        "timestamp_scale_to_seconds": scale,
        "tracer_scale_to_seconds": tracer_scale,
        "timestamp_resolution_s": float(pos_gap.min()),
        "median_stored_timestamp": float(np.median(ts)),
    }


def _fmt_s(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}\\,s"
    if seconds < 120:
        return f"{seconds:.0f}\\,s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}\\,min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}\\,h"
    return f"{seconds / 86400:.1f}\\,d"


def write_table(rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\caption{Start-to-start gaps on the training split of learner fold~0.",
        "Resolution is the smallest positive gap in the processed corpus.",
        "Zero-gap shares are over non-first rows; ``repeat'' rows are later",
        "KC-rows of one attempt. Gap AUC scores correctness by $\\log(1{+}\\Delta t)$",
        "alone on non-repeat rows (0.5 is uninformative; below 0.5, longer gaps",
        "go with errors). Lag-1 $r$ is the within-learner correlation of",
        "consecutive $\\log(1{+}\\Delta t)$.}",
        "\\label{tab:gap-descriptives}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Corpus & Resolution & $\\Delta t{=}0$ & Repeat & Median & 90th pct. & Gap AUC & Lag-1 $r$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{LABELS.get(r['dataset'], r['dataset'])} & {_fmt_s(r['timestamp_resolution_s'])} & "
            f"{100 * r['share_zero_gap']:.1f}\\% & "
            f"{100 * r['share_zero_repeat']:.1f}\\% & {_fmt_s(r['positive_gap_median_s'])} & "
            f"{_fmt_s(r['positive_gap_p90_s'])} & {r['auc_log_gap_correct']:.3f} & "
            f"{r['lag1_corr_log_gap']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for cfg in args.configs:
        row = describe(REPO / cfg, args.fold)
        print(json.dumps(row))
        rows.append(row)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_table(rows)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
