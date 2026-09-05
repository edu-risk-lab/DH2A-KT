#!/usr/bin/env python3
"""Run the capacity-matched Q←KC attribution check on XES3G5M fold 0.

Both arms instantiate the same question embedding and projection pathway:

* observed  — use the train-only Q–KC incidence matrix.
* zero      — replace only that incidence matrix with zeros.
* zero_time — keep zero incidence and add Linear log(1+gap) at LSTM + query.

The driver trains the requested arms at five paired seeds, verifies checkpoint
capacity, then re-scores every checkpoint on the outer test split only. The
``combined_eval_auc`` column is retained explicitly for diagnostics; it is the
historical valid+test frame and must not be called test AUC.

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON = Path(os.environ.get("DH2A_PYTHON", sys.executable))
DEFAULT_SEEDS = [42, 17, 1234, 0, 2024]
ARMS = ("observed", "zero", "zero_time")
MATRIX = REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_matrix.csv"
SUMMARY = REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_summary.json"
LOG_DIR = REPO_ROOT / "results" / "tables" / "qkc_capacity_matched_logs"
VAL_GATE = 0.002
_TEST_CONTEXT: tuple[object, object] | None = None

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


def _control_for_arm(arm: str) -> str:
    return "observed" if arm == "observed" else "zero"


def _extra_args_for_arm(arm: str) -> list[str]:
    if arm == "zero_time":
        return ["--time-gap", "--time-gap-mode", "both"]
    return []


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


def _read_matrix() -> pd.DataFrame:
    rows = pd.read_csv(MATRIX)
    # Results produced before the test-only audit used valid+test as eval_df.
    # Preserve those values, but remove the misleading "test" label.
    return rows.rename(
        columns={
            "test_auc": "combined_eval_auc",
            "n_predictions": "n_combined_predictions",
        }
    )


def _test_only_score(path: Path, device: str) -> tuple[float, int]:
    global _TEST_CONTEXT

    from dh2a_kt.hyperedge.p0_inputs import (
        get_fold_splits,
        load_configs,
        load_interactions_with_ids,
    )
    from dh2a_kt.models.dh2_kt import empty_hyperedge_index
    from dh2a_kt.train.checkpoint import load_trained_fold
    from dh2a_kt.train.greykt import make_sequence_loader
    from dh2a_kt.train.tier1 import evaluate_auc, resolve_training_budget

    if _TEST_CONTEXT is None:
        dh2_cfg, p0_cfg, _ = load_configs(
            REPO_ROOT / "configs" / "xes3g5m.yaml"
        )
        train_cfg = dh2_cfg.get("training", {})
        budget = resolve_training_budget(
            p0_cfg,
            reference_model=train_cfg.get("budget_reference", "gkt"),
            lr=float(train_cfg.get("lr", 0.001)),
            max_seq_len=400,
        )
        budget.batch_size = 16
        interactions = load_interactions_with_ids(p0_cfg)
        test_df = get_fold_splits(interactions, p0_cfg, 0)["test"]
        _TEST_CONTEXT = (test_df, budget)
    test_df, budget = _TEST_CONTEXT
    trained = load_trained_fold(path, device=device)
    index = trained.clean_hyperedge_index
    if not trained.clean_hyperedges:
        index = empty_hyperedge_index(
            trained.device, kinds=tuple(trained.model.config.hyperedge_kinds)
        )
    loader = make_sequence_loader(
        test_df, trained, budget, shuffle=False, window_mode="chunked"
    )
    auc, n_scored = evaluate_auc(
        trained.model,
        loader,
        index,
        trained.device,
        mask_repeats=True,
    )
    return float(auc), int(n_scored)


def _upsert(row: dict[str, object]) -> None:
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if MATRIX.exists():
        old = _read_matrix()
        duplicate = (old["arm"] == row["arm"]) & (old["seed"] == row["seed"])
        frame = pd.concat([old.loc[~duplicate], frame], ignore_index=True)
    frame.sort_values(["seed", "arm"]).to_csv(MATRIX, index=False)


def _already_done(arm: str, seed: int, force: bool) -> bool:
    if force or not MATRIX.exists():
        return False
    _tag, output, checkpoint, _log = _paths(arm, seed)
    if not output.exists() or not checkpoint.exists():
        return False
    rows = _read_matrix()
    match = (rows["arm"] == arm) & (rows["seed"] == seed)
    if not match.any():
        return False
    row = rows.loc[match].iloc[-1]
    return bool(
        row.get("status") == "ok"
        and pd.notna(row.get("test_only_auc"))
        and pd.notna(row.get("n_test_predictions"))
    )


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
        _control_for_arm(arm),
        *_extra_args_for_arm(arm),
        "--tag",
        tag,
        "--output",
        str(output.relative_to(REPO_ROOT)),
    ]
    reuse_checkpoint = not force and output.exists() and checkpoint.exists()
    action = "RESCORE" if reuse_checkpoint else "TRAIN"
    print(f"{action} ==>", " ".join(cmd), flush=True)
    if dry_run:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not reuse_checkpoint:
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
    expected_control = _control_for_arm(arm)
    if facts["checkpoint_control"] != expected_control:
        print(
            f"Checkpoint control mismatch: expected {expected_control}, got {facts}",
            file=sys.stderr,
        )
        return 5
    if expected_control == "zero" and facts["incidence_nonzero"] != 0:
        print("Zero-control checkpoint contains nonzero incidence", file=sys.stderr)
        return 6
    if expected_control == "observed" and int(facts["incidence_nonzero"]) <= 0:
        print("Observed checkpoint contains no incidence links", file=sys.stderr)
        return 7
    test_only_auc, n_test_predictions = _test_only_score(checkpoint, device)

    _upsert(
        {
            "arm": arm,
            "seed": seed,
            "fold": 0,
            "val_auc": float(result["val_auc"]),
            "combined_eval_auc": float(result["dh2_kt_auc"]),
            "n_combined_predictions": int(result["n_predictions"]),
            "test_only_auc": test_only_auc,
            "n_test_predictions": n_test_predictions,
            **facts,
            "minutes": round((time.time() - started) / 60.0, 2),
            "status": "ok",
        }
    )
    print(
        f"DONE arm={arm} seed={seed} val={float(result['val_auc']):.6f} "
        f"test_only={test_only_auc:.6f} n={n_test_predictions}",
        flush=True,
    )
    return 0


def _comparison_summary(
    rows: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    require_identical_state: bool,
) -> dict[str, object] | None:
    treated = rows[rows["arm"] == treatment].set_index("seed")
    baseline = rows[rows["arm"] == control].set_index("seed")
    paired_seeds = sorted(set(treated.index) & set(baseline.index))
    if not paired_seeds:
        return None

    pairs: list[dict[str, object]] = []
    for seed in paired_seeds:
        same_shape = (
            treated.loc[seed, "state_shape_signature"]
            == baseline.loc[seed, "state_shape_signature"]
        )
        same_numel = int(treated.loc[seed, "state_numel"]) == int(
            baseline.loc[seed, "state_numel"]
        )
        if require_identical_state and (not same_shape or not same_numel):
            raise SystemExit(f"Capacity mismatch remains at seed {seed}")
        delta_val = float(
            treated.loc[seed, "val_auc"] - baseline.loc[seed, "val_auc"]
        )
        pairs.append(
            {
                "seed": int(seed),
                "delta_val": delta_val,
                "delta_combined_eval": float(
                    treated.loc[seed, "combined_eval_auc"]
                    - baseline.loc[seed, "combined_eval_auc"]
                ),
                "delta_test_only": float(
                    treated.loc[seed, "test_only_auc"]
                    - baseline.loc[seed, "test_only_auc"]
                ),
                "gate_pass": delta_val >= VAL_GATE,
                "identical_state_shapes": bool(same_shape),
                "identical_state_numel": bool(same_numel),
            }
        )

    def stats(key: str) -> tuple[float, float | None]:
        values = pd.Series([float(pair[key]) for pair in pairs])
        return (
            float(values.mean()),
            float(values.std(ddof=1)) if len(values) > 1 else None,
        )

    mean_val, sd_val = stats("delta_val")
    mean_combined, sd_combined = stats("delta_combined_eval")
    mean_test, sd_test = stats("delta_test_only")
    return {
        "comparison": f"{treatment} minus {control}",
        "paired_seeds": paired_seeds,
        "n_pairs": len(pairs),
        "k_gate_pass": int(sum(bool(pair["gate_pass"]) for pair in pairs)),
        "mean_delta_val": mean_val,
        "sd_delta_val": sd_val,
        "mean_delta_combined_eval": mean_combined,
        "sd_delta_combined_eval": sd_combined,
        "mean_delta_test_only": mean_test,
        "sd_delta_test_only": sd_test,
        "pairs": pairs,
    }


def _write_summary(seeds: list[int]) -> int:
    rows = _read_matrix()
    rows = rows[(rows["seed"].isin(seeds)) & (rows["status"] == "ok")]
    comparisons = {
        "incidence_observed_vs_zero": _comparison_summary(
            rows,
            treatment="observed",
            control="zero",
            require_identical_state=True,
        ),
        "timing_on_zero_incidence": _comparison_summary(
            rows,
            treatment="zero_time",
            control="zero",
            require_identical_state=False,
        ),
    }
    comparisons = {key: value for key, value in comparisons.items() if value}
    if not comparisons:
        return 0
    payload = {
        "protocol": {
            "fold": 0,
            "validation": "internal train-user holdout used by the credit gate",
            "combined_eval": "outer valid+test; diagnostic only",
            "test_only": "outer test split, native DH2 clean mask, chunked L=400",
            "val_gate": VAL_GATE,
        },
        "comparisons": comparisons,
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
