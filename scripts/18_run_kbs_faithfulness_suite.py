#!/usr/bin/env python3
"""KBS faithfulness suite: grounded vs IG (+ optional scale) and JC eval.

Smoke (no GPU / no Ollama):
    python scripts/18_run_kbs_faithfulness_suite.py --mode smoke

Full Ollama re-run (requires checkpoint + Ollama):
    python scripts/18_run_kbs_faithfulness_suite.py --mode ollama \\
        --load-checkpoint results/checkpoints/xes3g5m_fold0.pt --device cuda

Writes comparison JSON under results/tables/kbs_faithfulness_*.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dh2a_kt.eval.explanation_faithfulness import (  # noqa: E402
    evaluate_explanation_faithfulness,
    read_pilot_jsonl,
    write_faithfulness_report,
)
from dh2a_kt.hyperedge.p0_inputs import e_pre_export_path, load_configs  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _eval_arm(
    records_path: Path,
    *,
    e_pre: pd.DataFrame | None,
    label: str,
    out_dir: Path,
) -> dict:
    records = read_pilot_jsonl(records_path)
    report = evaluate_explanation_faithfulness(
        records, e_pre=e_pre, source_records=str(records_path)
    )
    out = out_dir / f"{label}_faithfulness.json"
    write_faithfulness_report(report, out)
    return {
        "label": label,
        "records": str(records_path),
        "n_samples": report.n_samples,
        "mean_kc_jaccard": report.mean_kc_jaccard,
        "median_kc_jaccard": report.median_kc_jaccard,
        "n_empty_grounded": report.n_empty_grounded,
        "n_flagged": report.n_flagged,
        "flag_rate": report.flag_rate,
        "faithfulness_json": str(out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "xes3g5m.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", choices=["smoke", "ollama", "eval-only"], default="smoke")
    parser.add_argument("--sample-size", type=int, default=None, help="Default: 20 smoke / 500 ollama")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    parser.add_argument(
        "--grounded-records",
        type=Path,
        default=None,
        help="eval-only: existing grounded JSONL",
    )
    parser.add_argument(
        "--ig-records",
        type=Path,
        default=None,
        help="eval-only: existing IG JSONL",
    )
    parser.add_argument(
        "--run-scale",
        action="store_true",
        help="Also run n=100 scale sweep (1.5b/7b/14b); ollama mode only",
    )
    parser.add_argument(
        "--manipulation-folds",
        type=str,
        default="",
        help="Comma-separated folds for manipulation check, e.g. '1,2'",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / args.config)
    dataset = dh2_cfg["dataset"]
    e_pre_path = e_pre_export_path(p0_cfg, args.fold)
    e_pre = pd.read_csv(e_pre_path) if e_pre_path.exists() else None

    py = sys.executable
    pilot = str(REPO_ROOT / "scripts" / "04_run_tier2_pilot.py")
    sample_size = args.sample_size
    if sample_size is None:
        sample_size = 20 if args.mode == "smoke" else 500

    grounded_path: Path | None = args.grounded_records
    ig_path: Path | None = args.ig_records

    if args.mode in ("smoke", "ollama"):
        backend = "stub" if args.mode == "smoke" else "ollama"
        tag_base = "kbs_smoke" if args.mode == "smoke" else "kbs_ollama"
        common = [
            py,
            pilot,
            str(args.config),
            "--fold",
            str(args.fold),
            "--sample-size",
            str(sample_size),
            "--seed",
            "42",
            "--device",
            args.device,
            "--llm-backend",
            backend,
            "--output-dir",
            str(out_dir),
        ]
        if args.load_checkpoint is not None:
            common += ["--load-checkpoint", str(args.load_checkpoint)]
        else:
            common += ["--max-train-users", "64", "--pilot-epochs", "1"]
        if backend == "ollama":
            common += ["--ollama-model", args.ollama_model]

        _run(common + ["--output-tag", f"{tag_base}_grounded", "--ablation", "none"])
        _run(common + ["--output-tag", f"{tag_base}", "--ablation", "ig"])
        grounded_path = out_dir / f"{dataset}_fold{args.fold}_tier2_pilot_{tag_base}_grounded.jsonl"
        ig_path = out_dir / f"{dataset}_fold{args.fold}_tier2_pilot_{tag_base}_ig.jsonl"

        if args.run_scale and args.mode == "ollama":
            for model in ("qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b"):
                safe = model.replace(":", "_").replace(".", "p")
                scale_cmd = [
                    py,
                    pilot,
                    str(args.config),
                    "--fold",
                    str(args.fold),
                    "--sample-size",
                    "100",
                    "--seed",
                    "42",
                    "--device",
                    args.device,
                    "--llm-backend",
                    "ollama",
                    "--ollama-model",
                    model,
                    "--output-dir",
                    str(out_dir),
                    "--output-tag",
                    f"scale_{safe}",
                    "--ablation",
                    "none",
                ]
                if args.load_checkpoint is not None:
                    scale_cmd += ["--load-checkpoint", str(args.load_checkpoint)]
                else:
                    scale_cmd += ["--max-train-users", "64", "--pilot-epochs", "1"]
                _run(scale_cmd)

    if grounded_path is None or ig_path is None:
        raise SystemExit("Need grounded and IG JSONL (run smoke/ollama or pass --*-records)")

    comparison = {
        "dataset": dataset,
        "fold": args.fold,
        "mode": args.mode,
        "e_pre": str(e_pre_path) if e_pre is not None else None,
        "arms": [
            _eval_arm(Path(grounded_path), e_pre=e_pre, label="grounded", out_dir=out_dir),
            _eval_arm(Path(ig_path), e_pre=e_pre, label="ig", out_dir=out_dir),
        ],
    }
    g_jc = comparison["arms"][0]["mean_kc_jaccard"]
    i_jc = comparison["arms"][1]["mean_kc_jaccard"]
    comparison["delta_mean_kc_jaccard_grounded_minus_ig"] = g_jc - i_jc

    cmp_path = out_dir / f"{dataset}_fold{args.fold}_kbs_faithfulness_comparison.json"
    cmp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nWrote comparison: {cmp_path}")
    print(
        f"  grounded JC={g_jc:.4f} flag={comparison['arms'][0]['flag_rate']} | "
        f"IG JC={i_jc:.4f} flag={comparison['arms'][1]['flag_rate']} | "
        f"delta={comparison['delta_mean_kc_jaccard_grounded_minus_ig']:.4f}"
    )

    if args.manipulation_folds.strip():
        manip = str(REPO_ROOT / "scripts" / "05_run_manipulation_check.py")
        for fold_s in args.manipulation_folds.split(","):
            fold = int(fold_s.strip())
            _run(
                [
                    py,
                    manip,
                    str(args.config),
                    "--fold",
                    str(fold),
                    "--device",
                    args.device,
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
