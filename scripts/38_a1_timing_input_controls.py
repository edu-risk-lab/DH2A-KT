#!/usr/bin/env python3
"""Hau A1: trained time-*input* controls on the zero-incidence backbone.

All three arms keep ``--time-gap --time-gap-mode both`` and
``--question-incidence-control zero``. Only the gap *input* changes:

* t_real       — aligned log(1+Δt). Reuses ``qkc_cm_zero_time``; does not retrain.
* t_zero       — all gaps 0 (branch + bias still present).
* t_misaligned — per-user permutation of non-start gaps (same seed).

Usage (GPU; do not ``--force`` script 34):

    python scripts/38_a1_timing_input_controls.py --device cuda
    python scripts/38_a1_timing_input_controls.py --discover
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
VAL_GATE = 0.002
CKPT_DIR = REPO_ROOT / "results" / "checkpoints"
OUT_DIR = REPO_ROOT / "results" / "tables"
MATRIX = OUT_DIR / "a1_timing_input_controls.csv"
SUMMARY = OUT_DIR / "a1_timing_input_controls_summary.json"
LOG_DIR = OUT_DIR / "a1_timing_logs"
LOG = OUT_DIR / "a1_timing_input_controls.log"

COMMON = [
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
    "--question-incidence-control",
    "zero",
    "--time-gap",
    "--time-gap-mode",
    "both",
]


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _paths(arm: str, seed: int) -> tuple[str, Path, Path]:
    tag = f"a1_{arm}_s{seed}"
    return (
        tag,
        CKPT_DIR / f"xes3g5m_fold0_{tag}.pt",
        OUT_DIR / f"{tag}.csv",
    )


def _real_aliases(seed: int) -> list[Path]:
    tag, primary, _ = _paths("t_real", seed)
    return [
        primary,
        CKPT_DIR / f"xes3g5m_fold0_qkc_cm_zero_time_s{seed}.pt",
    ]


def _checkpoint_facts(path: Path) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    shapes = [(name, list(tensor.shape)) for name, tensor in sorted(state.items())]
    cfg = payload.get("config", {})
    return {
        "state_shape_signature": hashlib.sha256(
            json.dumps(shapes, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "state_numel": int(sum(t.numel() for t in state.values())),
        "has_time_gap_proj": any(name.startswith("time_gap_proj") for name, _ in shapes),
        "time_gap_control": cfg.get("time_gap_control", "missing"),
        "incidence_nonzero": int(
            torch.count_nonzero(state["A_qs_norm"]).item()
        )
        if "A_qs_norm" in state
        else -1,
    }


def _test_only_score(path: Path, device: str) -> tuple[float, int]:
    from dh2a_kt.hyperedge.p0_inputs import (
        get_fold_splits,
        load_configs,
        load_interactions_with_ids,
    )
    from dh2a_kt.models.dh2_kt import empty_hyperedge_index
    from dh2a_kt.train.checkpoint import load_trained_fold
    from dh2a_kt.train.greykt import make_sequence_loader
    from dh2a_kt.train.tier1 import evaluate_auc, resolve_training_budget

    dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / "configs" / "xes3g5m.yaml")
    train_cfg = dh2_cfg.get("training", {})
    budget = resolve_training_budget(
        p0_cfg,
        reference_model=train_cfg.get("budget_reference", "gkt"),
        lr=float(train_cfg.get("lr", 0.001)),
        max_seq_len=400,
    )
    budget.batch_size = 16
    test_df = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, 0)["test"]
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
        trained.model, loader, index, trained.device, mask_repeats=True
    )
    return float(auc), int(n_scored)


def _discover() -> None:
    for seed in DEFAULT_SEEDS:
        for arm in ("t_real", "t_zero", "t_misaligned"):
            if arm == "t_real":
                found = next((p for p in _real_aliases(seed) if p.exists()), None)
            else:
                found = _paths(arm, seed)[1] if _paths(arm, seed)[1].exists() else None
            mark = "OK" if found else "MISSING"
            path = found or (_real_aliases(seed)[0] if arm == "t_real" else _paths(arm, seed)[1])
            _log(f"{mark:7} {arm:14} seed={seed:<5} {path.relative_to(REPO_ROOT)}")


def _upsert(row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    if MATRIX.exists():
        old = pd.read_csv(MATRIX)
        dup = (old["arm"] == row["arm"]) & (old["seed"] == row["seed"])
        frame = pd.concat([old.loc[~dup], frame], ignore_index=True)
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["seed", "arm"]).to_csv(MATRIX, index=False)


def _already_ok(arm: str, seed: int, force: bool) -> bool:
    if force or not MATRIX.exists():
        return False
    old = pd.read_csv(MATRIX)
    match = (old["arm"] == arm) & (old["seed"] == seed)
    if not match.any():
        return False
    row = old.loc[match].iloc[-1]
    return str(row.get("status")) == "ok" and pd.notna(row.get("test_only_auc"))


def _run(arm: str, seed: int, device: str, *, force: bool, dry_run: bool) -> int:
    if _already_ok(arm, seed, force):
        _log(f"SKIP {arm} seed={seed} (already ok)")
        return 0

    if arm == "t_real":
        found = next((p for p in _real_aliases(seed) if p.exists()), None)
        if found is None:
            _log(f"STOP t_real seed={seed}: no qkc_cm_zero_time / a1_t_real checkpoint")
            _upsert(
                {
                    "arm": arm,
                    "seed": seed,
                    "status": "missing_t_real",
                    "note": "do not retrain script 34; copy Zero+Δt checkpoint",
                }
            )
            return 3
        started = time.time()
        facts = _checkpoint_facts(found)
        auc, n_scored = (float("nan"), 0) if dry_run else _test_only_score(found, device)
        _upsert(
            {
                "arm": arm,
                "seed": seed,
                "fold": 0,
                "val_auc": float("nan"),
                "test_only_auc": auc,
                "n_test_predictions": n_scored,
                "checkpoint": str(found.relative_to(REPO_ROOT)),
                **facts,
                "minutes": round((time.time() - started) / 60.0, 2),
                "status": "ok" if not dry_run else "dry_run",
                "note": "reused Zero+Δt; not retrained",
            }
        )
        _log(f"REUSE t_real seed={seed} {found.name} test_only={auc}")
        return 0

    tag, ckpt, out_csv = _paths(arm, seed)
    control = "zero" if arm == "t_zero" else "misaligned"
    cmd = [
        str(PYTHON),
        *COMMON,
        "--device",
        device,
        "--seed",
        str(seed),
        "--time-gap-control",
        control,
        "--tag",
        tag,
        "--output",
        str(out_csv.relative_to(REPO_ROOT)),
    ]
    _log(f"TRAIN {arm} seed={seed}")
    _log("  " + " ".join(cmd))
    if dry_run:
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (LOG_DIR / f"{tag}.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode:
        _log(f"FAIL {arm} seed={seed} exit={proc.returncode}")
        return proc.returncode
    if not ckpt.exists():
        _log(f"FAIL missing checkpoint {ckpt}")
        return 4
    facts = _checkpoint_facts(ckpt)
    if not facts["has_time_gap_proj"]:
        _log("FAIL time_gap_proj missing — branch was not instantiated")
        return 5
    if int(facts["incidence_nonzero"]) != 0:
        _log("FAIL incidence not zero")
        return 6
    results = pd.read_csv(out_csv)
    fold_rows = results[results["fold"].astype(str) == "0"]
    result = fold_rows.iloc[0] if len(fold_rows) else results.iloc[0]
    auc, n_scored = _test_only_score(ckpt, device)
    _upsert(
        {
            "arm": arm,
            "seed": seed,
            "fold": 0,
            "val_auc": float(result["val_auc"]),
            "test_only_auc": auc,
            "n_test_predictions": n_scored,
            "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
            **facts,
            "minutes": round((time.time() - started) / 60.0, 2),
            "status": "ok",
            "note": f"trained time-gap-control={control}",
        }
    )
    _log(f"DONE {arm} seed={seed} val={float(result['val_auc']):.6f} test_only={auc:.6f}")
    return 0


def _write_summary(seeds: list[int]) -> None:
    if not MATRIX.exists():
        return
    rows = pd.read_csv(MATRIX)
    rows = rows[(rows["seed"].isin(seeds)) & (rows["status"] == "ok")]

    def _pair(treatment: str, control: str) -> dict[str, object] | None:
        t = rows[rows["arm"] == treatment].set_index("seed")
        c = rows[rows["arm"] == control].set_index("seed")
        shared = sorted(set(t.index) & set(c.index))
        if not shared:
            return None
        pairs = []
        for seed in shared:
            d_test = float(t.loc[seed, "test_only_auc"] - c.loc[seed, "test_only_auc"])
            d_val = float("nan")
            if pd.notna(t.loc[seed, "val_auc"]) and pd.notna(c.loc[seed, "val_auc"]):
                d_val = float(t.loc[seed, "val_auc"] - c.loc[seed, "val_auc"])
            same = t.loc[seed, "state_numel"] == c.loc[seed, "state_numel"]
            pairs.append(
                {
                    "seed": int(seed),
                    "delta_val": d_val,
                    "delta_test_only": d_test,
                    "gate_pass": bool(pd.notna(d_val) and d_val >= VAL_GATE),
                    "identical_numel": bool(same),
                }
            )
        dvals = [p["delta_val"] for p in pairs if pd.notna(p["delta_val"])]
        dtests = [p["delta_test_only"] for p in pairs]
        return {
            "comparison": f"{treatment} minus {control}",
            "paired_seeds": shared,
            "k_gate_pass": int(sum(p["gate_pass"] for p in pairs)),
            "mean_delta_val": float(pd.Series(dvals).mean()) if dvals else None,
            "mean_delta_test_only": float(pd.Series(dtests).mean()),
            "pairs": pairs,
        }

    contrasts = {
        "t_real_vs_t_zero": _pair("t_real", "t_zero"),
        "t_real_vs_t_misaligned": _pair("t_real", "t_misaligned"),
    }
    isolated = False
    tz = contrasts["t_real_vs_t_zero"]
    tm = contrasts["t_real_vs_t_misaligned"]
    if tz and tm and tz["k_gate_pass"] == 5 and tm["k_gate_pass"] == 5:
        isolated = True
    payload = {
        "protocol": {
            "backbone": "zero Q-KC incidence + Linear Δt branch (both)",
            "controls": "trained, not inference-only",
            "val_gate": VAL_GATE,
            "do_not_retrain_script_34": True,
        },
        "contrasts": {k: v for k, v in contrasts.items() if v},
        "decision": {
            "aligned_dt_isolated_from_branch": isolated,
            "wording_if_false": (
                "Adding the time branch improves performance in our ablations; "
                "the separate contribution of temporal information versus the "
                "additional branch has not yet been fully isolated."
            ),
            "wording_if_true": (
                "Under the specified backbone and evaluation protocol, correctly "
                "aligned time gaps improve prediction relative to "
                "architecture-matched time-input controls."
            ),
        },
    }
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Wrote {SUMMARY.relative_to(REPO_ROOT)} isolated={isolated}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--arms",
        default="t_real,t_zero,t_misaligned",
        help="Comma list. t_real is reuse-only.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    if args.discover:
        _discover()
        return 0
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    for seed in seeds:
        for arm in arms:
            rc = _run(arm, seed, args.device, force=args.force, dry_run=args.dry_run)
            if rc:
                return rc
    if not args.dry_run:
        _write_summary(seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
