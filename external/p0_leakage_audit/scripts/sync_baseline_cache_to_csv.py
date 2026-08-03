"""Rebuild baseline_fold_results / baseline_results rows from results/cache/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.baseline_runner import _merge_csv, _summarize_fold_results


def main() -> None:
    cache = Path("results/cache")
    rows: list[dict] = []
    for path in sorted(cache.glob("*_result.json")):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit("No cache result JSON files found.")
    fold_df = pd.DataFrame(rows)
    datasets = sorted(fold_df["dataset"].unique())
    for dataset in datasets:
        sub = fold_df[fold_df["dataset"] == dataset].copy()
        summary = _summarize_fold_results(sub, seed=42, n_bootstrap=1000)
        _merge_csv(Path("results/tables/baseline_fold_results.csv"), sub, dataset)
        _merge_csv(Path("results/tables/baseline_results.csv"), summary, dataset)
        print(f"{dataset}: {len(sub)} fold rows -> {len(summary)} summary rows")
        print(summary[["model", "auc", "acc", "n_folds", "graph_construction", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
