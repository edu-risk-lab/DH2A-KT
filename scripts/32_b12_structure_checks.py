#!/usr/bin/env python3
"""Hau gold B12: structure checks on the frozen graph-only diagnostic encoder.

Degree-preserving rewiring keeps hyper-degree and hyperedge size; relation-label
permutation keeps membership and shuffles ``kind``. Run on the same v2
checkpoints as node-drop (not QKC-T). Does not invent ΔAUC — writes JSON for
the paper table.

Usage (GPU machine, after ``git pull``):

    python scripts/32_b12_structure_checks.py --device cuda
    python scripts/32_b12_structure_checks.py --device cuda --folds 0

Checkpoints default to ``results/checkpoints/xes3g5m_fold{N}.pt``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPS = ("degree_preserving_rewire", "relation_label_permute")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--checkpoint-template",
        default="results/checkpoints/xes3g5m_fold{fold}.pt",
        help="Python format with {fold}.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/xes3g5m.yaml"))
    args = parser.parse_args()

    missing = []
    for fold in args.folds:
        ckpt = REPO_ROOT / args.checkpoint_template.format(fold=fold)
        if not ckpt.is_file():
            missing.append(str(ckpt))
    if missing:
        print("Missing v2 checkpoints (train or copy them first):", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    for fold in args.folds:
        ckpt = REPO_ROOT / args.checkpoint_template.format(fold=fold)
        for op in OPS:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "05_run_manipulation_check.py"),
                str(args.config),
                "--fold",
                str(fold),
                "--device",
                args.device,
                "--load-checkpoint",
                str(ckpt),
                "--operator",
                op,
            ]
            print("==>", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if proc.returncode != 0:
                return proc.returncode
    print("B12 structure checks finished. JSON under results/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
