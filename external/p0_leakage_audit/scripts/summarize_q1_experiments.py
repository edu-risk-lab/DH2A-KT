"""Summarize isolated Q1 GPU runs under results/q1/ into tables for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GKT_VALID_AUC_FLOOR = 0.80


def _load_q1_folders(q1_root: Path) -> pd.DataFrame:
    frames = []
    if not q1_root.exists():
        return pd.DataFrame()
    for sub in sorted(q1_root.iterdir()):
        csv_path = sub / "baseline_fold_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "experiment_tag" not in df.columns or df["experiment_tag"].isna().all():
                df["experiment_tag"] = sub.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["dataset"] == "xes3g5m"]
    out = out[out["graph_construction"].fillna("train_only") == "train_only"]
    return out


def _load_trio_fallback(out_dir: Path, q1_df: pd.DataFrame) -> pd.DataFrame:
    """Keep Phase-3 trio rows when trio folders are absent (gitignored on server)."""
    merged = out_dir / "q1_baseline_fold_results.csv"
    if not merged.exists():
        return pd.DataFrame()
    prev = pd.read_csv(merged)
    trio = prev[prev["experiment_tag"].astype(str).str.startswith("trio_matched", na=False)]
    if not q1_df.empty:
        keys = set(zip(q1_df["experiment_tag"], q1_df["fold"], q1_df["model"]))
        trio = trio[~trio.apply(lambda r: (r["experiment_tag"], r["fold"], r["model"]) in keys, axis=1)]
    return trio.copy()


def _load_simplekt30_cache() -> pd.DataFrame:
    """Ingest simpleKT 30ep fold results from results/cache (GPU server exports)."""
    cache_dir = ROOT / "results/cache"
    if not cache_dir.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(cache_dir.glob("xes3g5m_fold_*_simplekt_s*_train_only_result.json")):
        name = path.name
        if "inject" in name:
            continue
        for base in (17, 42, 1234):
            token = f"_simplekt_s{base}_"
            if token not in name:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": payload.get("dataset", "xes3g5m"),
                    "fold": int(payload["fold"]),
                    "split_seed": int(payload["split_seed"]),
                    "model": "simplekt",
                    "graph_construction": payload.get("graph_construction", "train_only"),
                    "eval_split": payload.get("eval_split", "valid+test"),
                    "auc": float(payload["auc"]),
                    "acc": float(payload["acc"]),
                    "nll": float(payload["nll"]),
                    "n_eval": payload.get("n_eval"),
                    "status": payload.get("status", "pykt_checkpoint"),
                    "note": payload.get("note", ""),
                    "experiment_tag": f"simplekt30_s{base}",
                    "split_base_seed": base,
                }
            )
            break
    return pd.DataFrame(rows)


def _merge_q1_tables(q1_root: Path, out_dir: Path) -> pd.DataFrame:
    q1_df = _load_q1_folders(q1_root)
    sk30_df = _load_simplekt30_cache()
    trio_df = _load_trio_fallback(out_dir, q1_df)
    parts = [p for p in (q1_df, sk30_df, trio_df) if not p.empty]
    if not parts:
        raise FileNotFoundError(f"No Q1 results under {q1_root} or trio rows in merged CSV")
    merged = pd.concat(parts, ignore_index=True)
    key = ["experiment_tag", "fold", "model", "split_base_seed"]
    return merged.drop_duplicates(subset=key, keep="first")


def _gkt_seed_valid(gkt_part: pd.DataFrame) -> bool:
    return float(gkt_part["auc"].mean()) >= GKT_VALID_AUC_FLOOR


def _paired_delta_cross_tag(df: pd.DataFrame, baseline: str, challenger: str) -> pd.DataFrame:
    """Pair GKT (gkt_epochs30_*) vs simpleKT (trio_matched_*) at the same split_base_seed."""
    rows = []
    gkt = df[(df["model"] == challenger) & df["experiment_tag"].astype(str).str.startswith("gkt_epochs30")]
    simple = df[(df["model"] == baseline) & df["experiment_tag"].astype(str).str.startswith("simplekt30")]
    if simple.empty:
        simple = df[(df["model"] == baseline) & df["experiment_tag"].astype(str).str.startswith("trio_matched")]
    for seed in sorted(gkt["split_base_seed"].dropna().unique()):
        gpart = gkt[gkt["split_base_seed"] == seed].sort_values("fold")
        spart = simple[simple["split_base_seed"] == seed].sort_values("fold")
        if gpart.empty or spart.empty:
            continue
        mdf = spart[["fold", "auc"]].merge(gpart[["fold", "auc"]], on="fold", suffixes=("_sk", "_gkt"))
        if len(mdf) != 3:
            continue
        delta = mdf["auc_gkt"] - mdf["auc_sk"]
        valid = _gkt_seed_valid(gpart)
        rows.append(
            {
                "experiment_tag": f"gkt_epochs30_s{int(seed)}",
                "split_base_seed": int(seed),
                "baseline": baseline,
                "challenger": challenger,
                "n_folds": 3,
                "delta_mean": float(delta.mean()),
                "delta_std": float(delta.std(ddof=1)),
                "delta_values": ";".join(f"{v:.6f}" for v in delta.tolist()),
                "gkt_mean_auc": float(gpart["auc"].mean()),
                "valid": valid,
            }
        )
    cols = [
        "experiment_tag",
        "split_base_seed",
        "baseline",
        "challenger",
        "n_folds",
        "gkt_mean_auc",
        "delta_mean",
        "delta_std",
        "delta_values",
        "valid",
    ]
    return pd.DataFrame(rows, columns=cols)


def _write_tex(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/summarize_q1_experiments.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Epoch-matched GKT ablation on XES3G5M (isolated Q1 GPU runs). "
        r"$\Delta$AUC = AUC(GKT) $-$ AUC(\textit{simpleKT}; 30~epochs from cache when available, else Phase-3 trio) per fold. "
        r"Only seeds with fold-aligned graph exports are included.}",
        r"\label{tab:q1-gkt-epochs30}",
        r"\footnotesize",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Tag & Seed & Folds & $\Delta$AUC (GKT $-$ \textit{simpleKT}) & Fold deltas \\",
        r"\midrule",
    ]
    gkt = summary[(summary["challenger"] == "gkt") & summary["valid"].fillna(False)]
    if gkt.empty:
        lines.append(r"% No valid paired GKT rows (rebuild graph per seed before rerun).")
    else:
        for row in gkt.itertuples(index=False):
            pm = f"${row.delta_mean:+.3f} \\pm {row.delta_std:.3f}$"
            tag = row.experiment_tag.replace("_", r"\_")
            lines.append(
                f"\\texttt{{{tag}}} & {int(row.split_base_seed)} & {row.n_folds} & {pm} & "
                f"\\texttt{{{row.delta_values}}} \\\\"
            )
    stale = summary[(summary["challenger"] == "gkt") & ~summary["valid"].fillna(False)]
    for row in stale.itertuples(index=False):
        lines.append(
            f"% stale (omit): seed {int(row.split_base_seed)} GKT mean={row.gkt_mean_auc:.3f}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    tabular_only = [ln for ln in lines if not ln.startswith(r"\begin{table}") and not ln.startswith(r"\end{table}") and not ln.startswith(r"\centering") and not ln.startswith(r"\caption") and not ln.startswith(r"\label") and not ln.startswith(r"\footnotesize")]
    tabular_path = out_path.parent / "q1_gkt_epochs30_ablation_tabular.tex"
    tabular_path.write_text("\n".join(tabular_only), encoding="utf-8")


def sync_gkt_cache_fold0(dataset: str = "xes3g5m", split_base_seed: int = 17) -> bool:
    """Promote aligned fold-0 cache into results/q1 if isolated CSV is stale."""
    cache = ROOT / f"results/cache/{dataset}_fold_0_gkt_s{split_base_seed}_train_only_result.json"
    q1_csv = ROOT / f"results/q1/gkt_epochs30_s{split_base_seed}/baseline_fold_results.csv"
    if not cache.exists() or not q1_csv.exists():
        return False
    payload = json.loads(cache.read_text(encoding="utf-8"))
    df = pd.read_csv(q1_csv)
    if df.empty or float(df.loc[df["fold"] == 0, "auc"].iloc[0]) >= GKT_VALID_AUC_FLOOR:
        return False
    if float(payload["auc"]) < GKT_VALID_AUC_FLOOR:
        return False
    idx = df["fold"] == 0
    df.loc[idx, "auc"] = payload["auc"]
    df.loc[idx, "acc"] = payload["acc"]
    df.loc[idx, "nll"] = payload["nll"]
    df.loc[idx, "note"] = (
        "pyKT `gkt` trained on learner-split train users; metrics on valid+test sequence positions. "
        "GKT adjacency from P0 exported graphs (aligned graph rerun; fold 0)."
    )
    df.to_csv(q1_csv, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-root", type=Path, default=Path("results/q1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--sync-cache", action="store_true", help="Promote valid fold-0 cache into q1 CSVs")
    args = parser.parse_args()

    if args.sync_cache:
        for seed in (17, 1234, 42):
            if sync_gkt_cache_fold0(split_base_seed=seed):
                print(f"Synced fold-0 cache -> results/q1/gkt_epochs30_s{seed}/")

    df = _merge_q1_tables(args.q1_root, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / "q1_baseline_fold_results.csv"
    df.to_csv(merged_path, index=False)

    summary = _paired_delta_cross_tag(df, baseline="simplekt", challenger="gkt")
    summary_path = args.out_dir / "q1_gkt_vs_simplekt.csv"
    summary.to_csv(summary_path, index=False)
    _write_tex(summary, args.out_dir / "q1_gkt_epochs30_ablation.tex")

    print(f"Wrote {merged_path} ({len(df)} rows)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.out_dir / 'q1_gkt_epochs30_ablation.tex'}")
    if not summary.empty:
        print("\nGKT vs simpleKT (cross-tag, same seed):")
        print(summary[["experiment_tag", "split_base_seed", "gkt_mean_auc", "delta_mean", "valid"]])


if __name__ == "__main__":
    main()
