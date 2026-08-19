#!/usr/bin/env python3
"""Aggregate returned C-Human rating CSVs (2–3 raters).

Usage:
    python scripts/15_aggregate_human_eval.py
    python scripts/15_aggregate_human_eval.py --returned-dir results/tables/tier2_human_eval/distribution/returned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Rank-based Pearson without scipy dependency.
    ra = pd.Series(a).rank().to_numpy(dtype=float)
    rb = pd.Series(b).rank().to_numpy(dtype=float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _within_one(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) <= 1))


def load_returned(returned_dir: Path) -> pd.DataFrame:
    files = sorted(returned_dir.glob("human_eval_rater_*_done.csv"))
    if not files:
        files = sorted(returned_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {returned_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        for col in ("sample_id", "rater_id", "faithfulness_1to5", "usefulness_1to5"):
            if col not in df.columns:
                raise ValueError(f"{path.name} missing {col}")
        df = df.copy()
        df["faithfulness_1to5"] = pd.to_numeric(df["faithfulness_1to5"], errors="coerce")
        df["usefulness_1to5"] = pd.to_numeric(df["usefulness_1to5"], errors="coerce")
        if df[["faithfulness_1to5", "usefulness_1to5"]].isna().any().any():
            raise ValueError(f"{path.name} has blank/non-numeric scores")
        if not df["rater_id"].astype(str).str.strip().all():
            raise ValueError(f"{path.name} has blank rater_id")
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate(long_df: pd.DataFrame) -> dict:
    raters = sorted(long_df["rater_id"].astype(str).unique().tolist())
    n_samples = int(long_df["sample_id"].nunique())
    per_rater = {}
    for rid, g in long_df.groupby("rater_id"):
        per_rater[str(rid)] = {
            "n": int(len(g)),
            "faithfulness_mean": float(g["faithfulness_1to5"].mean()),
            "faithfulness_sd": float(g["faithfulness_1to5"].std(ddof=1)),
            "usefulness_mean": float(g["usefulness_1to5"].mean()),
            "usefulness_sd": float(g["usefulness_1to5"].std(ddof=1)),
        }

    overall = {
        "n_ratings": int(len(long_df)),
        "n_samples": n_samples,
        "n_raters": len(raters),
        "raters": raters,
        "faithfulness_mean": float(long_df["faithfulness_1to5"].mean()),
        "faithfulness_sd": float(long_df["faithfulness_1to5"].std(ddof=1)),
        "usefulness_mean": float(long_df["usefulness_1to5"].mean()),
        "usefulness_sd": float(long_df["usefulness_1to5"].std(ddof=1)),
        "per_rater": per_rater,
    }

    agreement: dict = {}
    if len(raters) >= 2:
        wide_f = long_df.pivot_table(
            index="sample_id", columns="rater_id", values="faithfulness_1to5", aggfunc="first"
        )
        wide_u = long_df.pivot_table(
            index="sample_id", columns="rater_id", values="usefulness_1to5", aggfunc="first"
        )
        pairs = []
        for i, r1 in enumerate(raters):
            for r2 in raters[i + 1 :]:
                if r1 not in wide_f.columns or r2 not in wide_f.columns:
                    continue
                f1 = wide_f[r1].to_numpy(dtype=float)
                f2 = wide_f[r2].to_numpy(dtype=float)
                u1 = wide_u[r1].to_numpy(dtype=float)
                u2 = wide_u[r2].to_numpy(dtype=float)
                pairs.append(
                    {
                        "pair": f"{r1}-vs-{r2}",
                        "faithfulness_spearman": _spearman(f1, f2),
                        "usefulness_spearman": _spearman(u1, u2),
                        "faithfulness_within_1": _within_one(f1, f2),
                        "usefulness_within_1": _within_one(u1, u2),
                        "n_paired": int(len(f1)),
                    }
                )
        agreement["pairwise"] = pairs
        if pairs:
            agreement["faithfulness_spearman_mean"] = float(
                np.nanmean([p["faithfulness_spearman"] for p in pairs])
            )
            agreement["usefulness_spearman_mean"] = float(
                np.nanmean([p["usefulness_spearman"] for p in pairs])
            )
            agreement["faithfulness_within_1_mean"] = float(
                np.mean([p["faithfulness_within_1"] for p in pairs])
            )
            agreement["usefulness_within_1_mean"] = float(
                np.mean([p["usefulness_within_1"] for p in pairs])
            )
    overall["agreement"] = agreement
    return overall


def write_human_eval_tex(summary: dict, path: Path) -> None:
    """Sync paper table from aggregated dual-rater summary."""
    agr = summary.get("agreement", {})
    f_mean = summary["faithfulness_mean"]
    f_sd = summary["faithfulness_sd"]
    u_mean = summary["usefulness_mean"]
    u_sd = summary["usefulness_sd"]
    rho_f = agr.get("faithfulness_spearman_mean", float("nan"))
    rho_u = agr.get("usefulness_spearman_mean", float("nan"))
    w1_f = 100.0 * float(agr.get("faithfulness_within_1_mean", float("nan")))
    w1_u = 100.0 * float(agr.get("usefulness_within_1_mean", float("nan")))
    n = summary["n_samples"]
    n_raters = summary["n_raters"]
    tex = f"""\\begin{{table}}[!t]
\\caption{{Human rating of Tier~2 outputs (XES3G5M fold~0 Ollama v2 pilot subsample,
$n{{=}}{n}$ explanations, {n_raters} independent raters, Likert 1--5). Agreement is pairwise
Spearman $\\rho$ and fraction of scores within $\\pm 1$.}}
\\label{{tab:human-eval}}
\\centering
\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
Scale & Mean $\\pm$ SD & Agreement (A1 vs.\\ B1) \\\\
\\midrule
Faithfulness & ${f_mean:.2f} \\pm {f_sd:.2f}$ & $\\rho{{=}}{rho_f:.2f}$; within$\\pm$1: {w1_f:.1f}\\% \\\\
Usefulness & ${u_mean:.2f} \\pm {u_sd:.2f}$ & $\\rho{{=}}{rho_u:.2f}$; within$\\pm$1: {w1_u:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returned-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "tables"
        / "tier2_human_eval"
        / "distribution"
        / "returned",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "tier2_human_eval",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=REPO_ROOT / "paper" / "tables" / "table_human_eval.tex",
        help="Write LaTeX table synced from the JSON summary.",
    )
    args = parser.parse_args()

    long_df = load_returned(args.returned_dir)
    summary = aggregate(long_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.out_dir / "human_eval_ratings_long.csv"
    summary_path = args.out_dir / "human_eval_summary.json"
    long_df.to_csv(long_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_human_eval_tex(summary, args.tex_out)

    print(f"Raters: {summary['raters']}  samples={summary['n_samples']}")
    print(
        f"Faithfulness: {summary['faithfulness_mean']:.3f} ± {summary['faithfulness_sd']:.3f}"
    )
    print(
        f"Usefulness:   {summary['usefulness_mean']:.3f} ± {summary['usefulness_sd']:.3f}"
    )
    agr = summary.get("agreement", {})
    if agr.get("pairwise"):
        print(
            f"Agreement Spearman F/U: "
            f"{agr.get('faithfulness_spearman_mean', float('nan')):.3f} / "
            f"{agr.get('usefulness_spearman_mean', float('nan')):.3f}"
        )
        print(
            f"Within±1 F/U: "
            f"{agr.get('faithfulness_within_1_mean', float('nan')):.3f} / "
            f"{agr.get('usefulness_within_1_mean', float('nan')):.3f}"
        )
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
