"""Remove baseline_runner / pyKT caches for selected models (before retraining)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _fold_from_cache_name(name: str) -> int | None:
    try:
        return int(name.split("_fold_")[1].split("_")[0])
    except (IndexError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear baseline cache for one or more models.")
    parser.add_argument("--dataset", default="xes3g5m")
    parser.add_argument("--models", required=True, help="Comma-separated model names, e.g. gkt")
    parser.add_argument("--folds", default="", help="Optional comma-separated fold indices, e.g. 0,1,2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    fold_filter = {int(x) for x in args.folds.split(",") if x.strip()} if args.folds.strip() else None
    cache_dir = Path("results/cache")
    work_root = Path("results/pykt_work") / args.dataset
    removed: list[str] = []

    if cache_dir.exists():
        for path in cache_dir.iterdir():
            if not path.is_file():
                continue
            name = path.name
            if not name.startswith(f"{args.dataset}_fold_"):
                continue
            matched = any(f"_{m}_" in name for m in models)
            if not matched:
                continue
            fold = _fold_from_cache_name(name)
            if fold_filter is not None and fold not in fold_filter:
                continue
            removed.append(str(path))
            if not args.dry_run:
                path.unlink(missing_ok=True)

    if work_root.exists():
        for model in models:
            for model_dir in work_root.rglob(model):
                if not model_dir.is_dir() or model_dir.name != model:
                    continue
                fold_part = model_dir.parent.parent.name if model_dir.parent else ""
                if fold_part.startswith("fold_"):
                    fold = _fold_from_cache_name(fold_part + "_0")
                    if fold_filter is not None and fold not in fold_filter:
                        continue
                removed.append(str(model_dir))
                if not args.dry_run:
                    shutil.rmtree(model_dir, ignore_errors=True)
            for ckpt in work_root.rglob(f"{model}_*_best.ckpt"):
                removed.append(str(ckpt))
                if not args.dry_run:
                    ckpt.unlink(missing_ok=True)

    print(f"{'Would remove' if args.dry_run else 'Removed'} {len(set(removed))} path(s)")
    for p in sorted(set(removed)):
        print(f"  {p}")


if __name__ == "__main__":
    main()
