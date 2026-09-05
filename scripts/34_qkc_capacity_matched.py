#!/usr/bin/env python3
"""Run the capacity-matched Q←KC attribution check on XES3G5M fold 0.

Both arms instantiate the same question embedding and projection pathway:

* observed — use the train-only Q–KC incidence matrix.
* zero     — replace only that incidence matrix with zeros.

The driver trains both arms at five paired seeds, verifies identical checkpoint
state shapes, and writes a matrix plus paired summary under ``results/tables``.

GPU usage:

    python scripts/34_qkc_capacity_matched.py --device cuda
    python scripts/34_qkc_capacity_matched.py --device cuda --seeds 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("DH2A_PYTHON", sys.executable))
DEFAULT_SEEDS = [42, 17, 1234, 0, 2024]
ARMS = ("observed", "zero")
MATRIX = REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_matrix.csv"
SUMMARY = REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_summary.json"
LOG_DIR = REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_logs"
VAL_GATE = 0.002

COMMON_ARGS = [
    "scripts/03_train_tier1.py",
    "configs/xes3g5m.yaml",
    "--fold",
    "0",
    "--architecture",
    "v4",
    "--use-questions",
    "--no-graph",
    "--no-session",
    "--mask-repeats",
    "--window-mode",
    "chunked",
    "--max-seq-len",
    "400",
    "--batch-size",
    "16",
    "--epochs",
    "30",
    "--val-frac",
    "0.1",
    "--early-stop-patience",
    "5",
    "--graph-dropout",
    "0",
    "--graph-sensitivity-weight",
    "0",
    "--question-graph",
]


def _preflight(device: str) -> None:
    probe = (
        "import torch; "
        f"assert torch.cuda.is_available() or {device!r} == 'cpu', 'cuda unavailable'; "
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    )
    proc = subprocess.run(
        [str(PYTHON), "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise SystemExit((proc.stderr or proc.stdout).strip())
    print(proc.stdout.strip(), flush=True)


def _paths(arm: str, seed: int) -> tuple[str, Path, Path, Path]:
    tag = f"qkc_cm_{arm}_s{seed}"
    output = REPO_ROOT / "results" / "tables" / f"{tag}.csv"
    checkpoint = REPO_ROOT / "results" / "checkpoints" / f"xes3g5m_fold0_{tag}.pt"
    log = LOG_DIR / f"{tag}.log"
    return tag, output, checkpoint, log


def _checkpoint_facts(path: Path) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    shapes = [(name, list(tensor.shape)) for name, tensor in sorted(state.items())]
    signature = hashlib.sha256(
        json.dumps(shapes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    incidence = state.get("A_qs_norm")
    return {
        "state_shape_signature": signature,
        "state_numel": int(sum(t.numel() for t in state.values())),
        "incidence_nonzero": (
            int(torch.count_nonzero(incidence).item()) if incidence is not None else -1
        ),
        "checkpoint_control": payload.get("config", {}).get(
            "question_incidence_control", "missing"
        ),
    }


def _upsert(row: dict[str, object]) -> None:
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if MATRIX.exists():
        old = pd.read_csv(MATRIX)
        duplicate = (old["arm"] == row["arm"]) & (old["seed"] == row["seed"])
        frame = pd.concat([old.loc[~duplicate], frame], ignore_index=True)
    frame.sort_values(["seed", "arm"]).to_csv(MATRIX, index=False)


def _already_done(arm: str, seed: int, force: bool) -> bool:
    if force or not MATRIX.exists():
        return False
    rows = pd.read_csv(MATRIX)
    match = (rows["arm"] == arm) & (rows["seed"] == seed)
    return bool(match.any() and rows.loc[match].iloc[-1].get("status") == "ok")


def _run(arm: str, seed: int, device: str, *, force: bool, dry_run: bool) -> int:
    if _already_done(arm, seed, force):
        print(f"SKIP arm={arm} seed={seed}", flush=True)
        return 0
    tag, output, checkpoint, log = _paths(arm, seed)
    cmd = [
        str(PYTHON),
        *COMMON_ARGS,
        "--device",
        device,
        "--seed",
        str(seed),
        "--question-incidence-control",
        arm,
        "--tag",
        tag,
        "--output",
        str(output.relative_to(REPO_ROOT)),
    ]
    print("==>", " ".join(cmd), flush=True)
    if dry_run:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode:
        print(f"FAIL arm={arm} seed={seed}; see {log}", file=sys.stderr)
        return proc.returncode
    if not output.exists() or not checkpoint.exists():
        print(f"Missing output/checkpoint for {tag}", file=sys.stderr)
        return 4

    results = pd.read_csv(output)
    fold_rows = results[results["fold"].astype(str) == "0"]
    result = fold_rows.iloc[0] if len(fold_rows) else results.iloc[0]
    facts = _checkpoint_facts(checkpoint)
    if facts["checkpoint_control"] != arm:
        print(f"Checkpoint control mismatch: expected {arm}, got {facts}", file=sys.stderr)
        return 5
    if arm == "zero" and facts["incidence_nonzero"] != 0:
        print("Zero-control checkpoint contains nonzero incidence", file=sys.stderr)
        return 6
    if arm == "observed" and int(facts["incidence_nonzero"]) <= 0:
        print("Observed checkpoint contains no incidence links", file=sys.stderr)
        return 7

    _upsert(
        {
            "arm": arm,
            "seed": seed,
            "fold": 0,
            "val_auc": float(result["val_auc"]),
            "test_auc": float(result["dh2_kt_auc"]),
            "n_predictions": int(result["n_predictions"]),
            **facts,
            "minutes": round((time.time() - started) / 60.0, 2),
            "status": "ok",
        }
    )
    print(
        f"DONE arm={arm} seed={seed} val={float(result['val_auc']):.6f} "
        f"test={float(result['dh2_kt_auc']):.6f}",
        flush=True,
    )
    return 0


def _write_summary(seeds: list[int]) -> int:
    rows = pd.read_csv(MATRIX)
    rows = rows[(rows["seed"].isin(seeds)) & (rows["status"] == "ok")]
    observed = rows[rows["arm"] == "observed"].set_index("seed")
    zero = rows[rows["arm"] == "zero"].set_index("seed")
    paired_seeds = sorted(set(observed.index) & set(zero.index))
    if not paired_seeds:
        return 0

    pairs: list[dict[str, object]] = []
    for seed in paired_seeds:
        same_shape = (
            observed.loc[seed, "state_shape_signature"]
            == zero.loc[seed, "state_shape_signature"]
        )
        same_numel = int(observed.loc[seed, "state_numel"]) == int(
            zero.loc[seed, "state_numel"]
        )
        if not same_shape or not same_numel:
            raise SystemExit(f"Capacity mismatch remains at seed {seed}")
        delta_val = float(observed.loc[seed, "val_auc"] - zero.loc[seed, "val_auc"])
        delta_test = float(observed.loc[seed, "test_auc"] - zero.loc[seed, "test_auc"])
        pairs.append(
            {
                "seed": int(seed),
                "delta_val_observed_minus_zero": delta_val,
                "delta_test_observed_minus_zero": delta_test,
                "gate_pass": delta_val >= VAL_GATE,
                "identical_state_shapes": True,
                "identical_state_numel": True,
            }
        )

    val_deltas = pd.Series(
        [float(pair["delta_val_observed_minus_zero"]) for pair in pairs]
    )
    test_deltas = pd.Series(
        [float(pair["delta_test_observed_minus_zero"]) for pair in pairs]
    )
    payload = {
        "comparison": "observed Q-KC incidence minus zero-incidence capacity twin",
        "val_gate": VAL_GATE,
        "paired_seeds": paired_seeds,
        "n_pairs": len(pairs),
        "k_gate_pass": int(sum(bool(pair["gate_pass"]) for pair in pairs)),
        "mean_delta_val": float(val_deltas.mean()),
        "sd_delta_val": float(val_deltas.std(ddof=1)) if len(pairs) > 1 else None,
        "mean_delta_test": float(test_deltas.mean()),
        "sd_delta_test": float(test_deltas.std(ddof=1)) if len(pairs) > 1 else None,
        "pairs": pairs,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {MATRIX.relative_to(REPO_ROOT)}", flush=True)
    print(f"Wrote {SUMMARY.relative_to(REPO_ROOT)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"Unknown arms: {', '.join(unknown)}")
    if not args.dry_run:
        _preflight(args.device)

    for seed in seeds:
        for arm in arms:
            rc = _run(
                arm,
                seed,
                args.device,
                force=args.force,
                dry_run=args.dry_run,
            )
            if rc:
                return rc
    if args.dry_run:
        return 0
    return _write_summary(seeds)


if __name__ == "__main__":
    raise SystemExit(main())
