#!/usr/bin/env python3
"""Rescore existing A2/A7 DH² checkpoints on the *test-only* split.

Does not train. Does not overwrite ``a2_multiseed_matrix.csv`` or
``qkc_capacity_matched_*``. Writes a new matrix so Table 5 / A7 can share
one metric (native clean test, mask-repeats, chunked L=400).

A2's stored ``dh2_kt_auc`` is combined val+test (n=1,640,242). Script 34
already names that field ``combined_eval_auc`` and scores ``test_only_auc``
(n=1,093,755) for capacity-matched arms. This driver does the same for the
A2 QKC-T / Q←KC / A7 checkpoints.

Usage (GPU host; CPU works if VRAM is tight):

    python scripts/36_rescore_a2_testonly.py --device cuda
    python scripts/36_rescore_a2_testonly.py --device cuda --arms p0_dt_on
    python scripts/36_rescore_a2_testonly.py --discover
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = Path(os.environ.get("DH2A_PYTHON", sys.executable))
DEFAULT_SEEDS = [42, 17, 1234, 0, 2024]
CKPT_DIR = REPO_ROOT / "results" / "checkpoints"
OUT_DIR = REPO_ROOT / "results" / "tables"
MATRIX = OUT_DIR / "a2_testonly_rescore.csv"
SUMMARY = OUT_DIR / "a2_testonly_rescore_summary.json"
LOG = OUT_DIR / "rescore_testonly_2026-09-08.log"
SEED42_QKCT_REF = 0.829592957
SEED42_TOL = 5e-5
_TEST_CONTEXT: tuple[object, object] | None = None

# tag used by scripts/03_train_tier1.py → xes3g5m_fold0_{tag}.pt
ARMS: dict[str, dict[str, str]] = {
    "p0_dt_on": {
        "label": "QKC-T",
        "tag": "p0_dt_on",
        "role": "evaluation_arm",
    },
    "hg_qkc_on": {
        "label": "Q←KC",
        "tag": "hg_qkc_on",
        "role": "no_time",
    },
    "a7_dt_lstm": {
        "label": "history-only Δt",
        "tag": "a7_dt_lstm",
        "role": "a7_lstm",
    },
    "a7_dt_query": {
        "label": "query-only Δt",
        "tag": "a7_dt_query",
        "role": "a7_query",
    },
    "a7_dt_both": {
        "label": "A7 both (alias of QKC-T if missing)",
        "tag": "a7_dt_both",
        "role": "a7_both",
    },
}


def _log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _checkpoint_for(arm: str, seed: int) -> Path:
    tag = f"{ARMS[arm]['tag']}_s{seed}"
    return CKPT_DIR / f"xes3g5m_fold0_{tag}.pt"


def _alias_checkpoints(arm: str, seed: int) -> list[Path]:
    """A7 both was often imported from A2 and may share the QKC-T file."""
    primary = _checkpoint_for(arm, seed)
    extras: list[Path] = []
    if arm == "a7_dt_both":
        extras.append(_checkpoint_for("p0_dt_on", seed))
    return [primary, *extras]


def _discover() -> list[str]:
    lines: list[str] = []
    if not CKPT_DIR.exists():
        return [f"MISSING directory {CKPT_DIR}"]
    for arm in ARMS:
        for seed in DEFAULT_SEEDS:
            found = next((p for p in _alias_checkpoints(arm, seed) if p.exists()), None)
            mark = "OK" if found else "MISSING"
            path = found if found else _checkpoint_for(arm, seed)
            lines.append(f"{mark:7} {arm:12} seed={seed:<5} {path}")
    extras = sorted(CKPT_DIR.glob("xes3g5m_fold0_*s*.pt"))
    lines.append(f"glob count xes3g5m_fold0_*s*.pt = {len(extras)}")
    return lines


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
        dh2_cfg, p0_cfg, _ = load_configs(REPO_ROOT / "configs" / "xes3g5m.yaml")
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


def _read_matrix() -> pd.DataFrame:
    if not MATRIX.exists():
        return pd.DataFrame()
    return pd.read_csv(MATRIX)


def _upsert(row: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    old = _read_matrix()
    if not old.empty:
        dup = (old["arm"] == row["arm"]) & (old["seed"] == row["seed"])
        frame = pd.concat([old.loc[~dup], frame], ignore_index=True)
    frame.sort_values(["arm", "seed"]).to_csv(MATRIX, index=False)


def _already_ok(arm: str, seed: int, force: bool) -> bool:
    if force:
        return False
    old = _read_matrix()
    if old.empty:
        return False
    match = (old["arm"] == arm) & (old["seed"] == seed)
    if not match.any():
        return False
    row = old.loc[match].iloc[-1]
    return str(row.get("status")) == "ok" and pd.notna(row.get("test_only_auc"))


def _score_one(arm: str, seed: int, device: str, *, force: bool) -> dict[str, object]:
    if _already_ok(arm, seed, force):
        _log(f"SKIP {arm} seed={seed} (already ok in {MATRIX.name})")
        old = _read_matrix()
        return old.loc[(old["arm"] == arm) & (old["seed"] == seed)].iloc[-1].to_dict()

    found = next((p for p in _alias_checkpoints(arm, seed) if p.exists()), None)
    if found is None:
        searched = ", ".join(str(p.relative_to(REPO_ROOT)) for p in _alias_checkpoints(arm, seed))
        row = {
            "arm": arm,
            "label": ARMS[arm]["label"],
            "role": ARMS[arm]["role"],
            "seed": seed,
            "checkpoint": "",
            "test_only_auc": float("nan"),
            "n_test_predictions": None,
            "minutes": 0.0,
            "status": "missing_checkpoint",
            "note": searched,
        }
        _upsert(row)
        _log(f"SKIP {arm} seed={seed} missing checkpoint ({searched})")
        return row

    started = time.time()
    _log(f"SCORE {arm} seed={seed} {found.relative_to(REPO_ROOT)}")
    auc, n_scored = _test_only_score(found, device)
    note = ""
    if arm == "p0_dt_on" and seed == 42:
        delta = abs(auc - SEED42_QKCT_REF)
        if delta > SEED42_TOL:
            note = (
                f"seed42 |auc-a6|={delta:.6f} (a6={SEED42_QKCT_REF:.6f}); "
                "same checkpoint should match a6_calibration"
            )
            _log(f"WARN {note}")
        else:
            note = "matches a6 seed-42 within 5e-5"
    row = {
        "arm": arm,
        "label": ARMS[arm]["label"],
        "role": ARMS[arm]["role"],
        "seed": seed,
        "checkpoint": str(found.relative_to(REPO_ROOT)),
        "test_only_auc": auc,
        "n_test_predictions": n_scored,
        "minutes": round((time.time() - started) / 60.0, 2),
        "status": "ok",
        "note": note,
    }
    _upsert(row)
    _log(f"DONE {arm} seed={seed} test_only={auc:.6f} n={n_scored}")
    return row


def _write_summary(seeds: list[int], arms: list[str]) -> None:
    rows = _read_matrix()
    if rows.empty:
        return
    payload: dict[str, object] = {
        "protocol": {
            "split": "outer test only",
            "mask_repeats": True,
            "window_mode": "chunked",
            "max_seq_len": 400,
            "expected_n_native": 1_093_755,
            "do_not_call_combined_val_test_test_only": True,
        },
        "arms": {},
    }
    for arm in arms:
        block = rows[(rows["arm"] == arm) & (rows["seed"].isin(seeds))]
        ok = block[block["status"] == "ok"]
        missing = block[block["status"] != "ok"]
        stats: dict[str, object] = {
            "n_ok": int(len(ok)),
            "n_missing": int(len(missing)),
            "missing_seeds": [int(s) for s in missing["seed"].tolist()],
        }
        if len(ok):
            values = ok["test_only_auc"].astype(float)
            stats["mean"] = float(values.mean())
            stats["sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            stats["n_test_predictions"] = [
                int(n) for n in ok["n_test_predictions"].tolist()
            ]
            stats["per_seed"] = [
                {
                    "seed": int(r.seed),
                    "test_only_auc": float(r.test_only_auc),
                    "n": int(r.n_test_predictions),
                }
                for r in ok.itertuples()
            ]
        payload["arms"][arm] = stats
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Wrote {MATRIX.relative_to(REPO_ROOT)}")
    _log(f"Wrote {SUMMARY.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--arms",
        default="p0_dt_on,hg_qkc_on,a7_dt_lstm,a7_dt_query,a7_dt_both",
        help="Comma list from: " + ",".join(ARMS),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Print checkpoint presence and exit (no scoring).",
    )
    args = parser.parse_args()

    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    arms = [v.strip() for v in args.arms.split(",") if v.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"Unknown arms: {', '.join(unknown)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.discover:
        for line in _discover():
            _log(line)
        return 0

    _log(f"python={PYTHON} device={args.device} arms={arms} seeds={seeds}")
    for line in _discover():
        _log(line)

    qkct_missing = [
        seed
        for seed in seeds
        if not any(p.exists() for p in _alias_checkpoints("p0_dt_on", seed))
    ]
    if "p0_dt_on" in arms and qkct_missing == seeds:
        _log(
            "STOP: no QKC-T A2 checkpoints found. Do not train. "
            f"Looked under {CKPT_DIR}"
        )
        return 4

    for seed in seeds:
        for arm in arms:
            _score_one(arm, seed, args.device, force=args.force)
    _write_summary(seeds, arms)

    rows = _read_matrix()
    qkct = rows[(rows["arm"] == "p0_dt_on") & (rows["status"] == "ok")]
    if len(qkct) == 5:
        mean = float(qkct["test_only_auc"].mean())
        sd = float(qkct["test_only_auc"].std(ddof=1))
        _log(f"QKC-T 5-seed test-only {mean:.6f}±{sd:.6f} → replace Table 5 seed-42 cell")
    elif len(qkct):
        _log(
            f"QKC-T test-only scored {len(qkct)}/5 seeds; "
            "keep Table 5 seed-42 cell until all five exist"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
