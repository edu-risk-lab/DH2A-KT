#!/usr/bin/env python3
"""Learner bootstrap and practical quantities (docs/preregistration_round3.md, Sections 5-6).

Stage ``dump`` (GPU server, needs the run checkpoints): rescores every run of
a family-C contrast from its checkpoint with the run's own command plus
``--eval-checkpoint`` and ``--dump-predictions``, writing
results/predictions/<dataset>_fold<f>_<group>_<arm>_s<seed>.npz. Finished
files are skipped.

Stage ``analyse`` (CPU): for each family-C contrast
  * checks each rescored AUC against the run's stored ``dh2_kt_auc``;
  * learner bootstrap of the mean-over-seeds AUC difference, B = 2000,
    seed 20261023, one resample of held-out learners shared by both arms
    and all seeds. Each replicate AUC is computed from per-learner counts
    in 2000 quantile bins of the run's probabilities (Section 10 of the
    preregistration); the exact AUC is reported beside it;
  * learner variance of a single seed pair (for Section 5's components);
  * mean over seeds of on minus control: log loss, Brier score, ECE
    (15 equal-width bins), accuracy at 0.5, and the share of positions
    whose 0.5 decision flips.

    python scripts/61_learner_bootstrap.py --stage dump [--only XES3G5M]
    python scripts/61_learner_bootstrap.py --stage analyse

Writes results/tables/learner_bootstrap.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
CKPT = REPO / "results" / "checkpoints"
PRED = REPO / "results" / "predictions"
LOGS = REPO / "logs" / "rescore"
OUT_JSON = TABLES / "learner_bootstrap.json"

SEEDS_10 = (0, 17, 42, 1234, 2024, 7, 99, 314, 2718, 3407)
FROZEN_SEEDS = (0, 17, 1234, 2024)
B = 2000
BOOT_SEED = 20261023
N_BINS = 2000
ECE_BINS = 15
AUC_TOL = 1e-6

# xes-ref arms first trained by scripts 34 and 38 on seeds 0, 17, 1234, 2024.
FROZEN_TAG = {
    "incidence-on": "qkc_cm_observed",
    "no-time": "qkc_cm_zero",
    "time-package": "qkc_cm_zero_time",
    "t-zero": "a1_t_zero",
}

PKG, GAP, INC = "Package, zero incidence", "Aligned gap vs.\\ T-zero", "Q$\\leftarrow$KC incidence"


def _contrasts() -> list[dict]:
    """Family C of docs/preregistration_round3.md, Section 3."""
    out = []

    def add(setting: str, fold: int, group: str, arms: dict[str, tuple[str, str]]) -> None:
        for contrast, (on, ctrl) in arms.items():
            out.append({"setting": setting, "fold": fold, "group": group, "contrast": contrast, "on": on, "control": ctrl})

    xes = {PKG: ("time-package", "no-time"), GAP: ("time-package", "t-zero")}
    assist = {PKG: ("time-on-zero", "incidence-off"), GAP: ("time-on-zero", "t-zero")}
    add("XES3G5M", 0, "xes-ref", {INC: ("incidence-on", "no-time"), **xes})
    for f in (1, 2):
        add("XES3G5M", f, f"xes-fold{f}", xes)
    add("XES3G5M attn.", 0, "backbone-attn", xes)
    for f in (1, 2):
        add("XES3G5M attn.", f, f"backbone-attn-fold{f}", xes)
    add("ASSIST2012", 0, "assist-sec", {INC: ("incidence-on", "incidence-off"), **assist})
    for f in (1, 2):
        add("ASSIST2012", f, f"assist-sec-cfg-fold{f}", {INC: ("incidence-on", "incidence-off"), **assist})
    add("ASSIST2012 ref.", 0, "assist-sec-ref", {INC: ("incidence-on", "incidence-off"), **assist})
    for f in (1, 2):
        add("ASSIST2012 ref.", f, f"assist-sec-fold{f}", assist)
    return out


def _plan_jobs() -> dict[tuple[str, str, int], list[str]]:
    spec = importlib.util.spec_from_file_location("plan50", REPO / "scripts" / "50_credit_ladder_plan.py")
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    return {(j["group"], j["arm"], j["seed"]): j["cmd"] for j in plan._jobs() if j["cmd"] is not None}


def _with_seed(cmd: list[str], seed: int) -> list[str]:
    cmd = list(cmd)
    cmd[cmd.index("--seed") + 1] = str(seed)
    return cmd


def _dataset(cmd: list[str]) -> str:
    config = next(a for a in cmd if a.endswith(".yaml"))
    return yaml.safe_load((REPO / config).read_text(encoding="utf-8"))["dataset"]


def _fold(cmd: list[str]) -> int:
    return int(cmd[cmd.index("--fold") + 1])


def _run_files(group: str, arm: str, seed: int, jobs: dict) -> dict:
    """Command, checkpoint, stored-result CSV, and prediction path of one run."""
    frozen = group == "xes-ref" and seed in FROZEN_SEEDS
    cmd = _with_seed(jobs[(group, arm, 42)], seed) if frozen else jobs[(group, arm, seed)]
    dataset, fold = _dataset(cmd), _fold(cmd)
    tag = f"{FROZEN_TAG[arm]}_s{seed}" if frozen else f"ladder_{group}_{arm}_s{seed}"
    stored = TABLES / (f"{tag}.csv" if frozen else f"dh2_kt_{tag}_vs_p0.csv")
    return {
        "cmd": cmd,
        "ckpt": CKPT / f"{dataset}_fold{fold}_{tag}.pt",
        "stored_csv": stored,
        "npz": PRED / f"{dataset}_fold{fold}_{group}_{arm}_s{seed}.npz",
    }


def _needed_runs(contrasts: list[dict]) -> list[tuple[str, str, int]]:
    seen: list[tuple[str, str, int]] = []
    for c in contrasts:
        for arm in (c["on"], c["control"]):
            for seed in SEEDS_10:
                key = (c["group"], arm, seed)
                if key not in seen:
                    seen.append(key)
    return seen


def stage_dump(only: str | None) -> int:
    jobs = _plan_jobs()
    contrasts = [c for c in _contrasts() if only is None or c["setting"] == only]
    LOGS.mkdir(parents=True, exist_ok=True)
    failed = 0
    for group, arm, seed in _needed_runs(contrasts):
        files = _run_files(group, arm, seed, jobs)
        if files["npz"].exists():
            print(f"SKIP  {files['npz'].name}")
            continue
        if not files["ckpt"].exists():
            print(f"MISSING checkpoint {files['ckpt']}")
            failed += 1
            continue
        cmd = [*files["cmd"], "--eval-checkpoint", str(files["ckpt"]), "--dump-predictions", str(files["npz"])]
        log = LOGS / f"{files['npz'].stem}.log"
        print("rescore:", files["npz"].name)
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"FAIL  {files['npz'].name} (see {log})")
            failed += 1
    print(f"done; {failed} missing or failed")
    return 1 if failed else 0


def _auc_exact(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _binned_counts(y: np.ndarray, p: np.ndarray, user_idx: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]))
    b = np.searchsorted(edges, p, side="right")
    nb = len(edges) + 1
    pos = np.zeros((n_users, nb), dtype=np.float32)
    neg = np.zeros((n_users, nb), dtype=np.float32)
    np.add.at(pos, (user_idx[y == 1], b[y == 1]), 1.0)
    np.add.at(neg, (user_idx[y == 0], b[y == 0]), 1.0)
    return pos, neg


def _auc_from_hist(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Rows are replicates, columns bins in increasing probability; ties count 1/2."""
    neg_below = np.cumsum(neg, axis=1) - neg
    num = (pos * (neg_below + 0.5 * neg)).sum(axis=1)
    return num / (pos.sum(axis=1) * neg.sum(axis=1))


def _ece(y: np.ndarray, p: np.ndarray) -> float:
    b = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    total = 0.0
    for k in range(ECE_BINS):
        sel = b == k
        if sel.any():
            total += sel.sum() * abs(y[sel].mean() - p[sel].mean())
    return float(total / len(y))


def _practical(y: np.ndarray, p: np.ndarray) -> dict:
    q = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()),
        "brier": float(((q - y) ** 2).mean()),
        "ece15": _ece(y, q),
        "accuracy": float(((q >= 0.5) == (y == 1)).mean()),
    }


def _stored_auc(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return float(next(csv.DictReader(fh))["dh2_kt_auc"])


def stage_analyse() -> int:
    jobs = _plan_jobs()
    results = []
    for c in _contrasts():
        files = {
            (arm, s): _run_files(c["group"], arm, s, jobs) for arm in (c["on"], c["control"]) for s in SEEDS_10
        }
        missing = [f["npz"].name for f in files.values() if not f["npz"].exists()]
        if missing:
            results.append({**c, "pending": missing})
            print(f"PENDING {c['setting']} fold {c['fold']} {c['contrast']}: {len(missing)} files")
            continue
        data = {}
        for k, f in files.items():
            with np.load(f["npz"]) as z:
                data[k] = {name: z[name] for name in ("user_ids", "targets", "probs")}
        ref = data[(c["control"], SEEDS_10[0])]
        users, y = ref["user_ids"], ref["targets"].astype(np.int8)
        for k, d in data.items():
            if not (np.array_equal(d["user_ids"], users) and np.array_equal(d["targets"], y)):
                raise SystemExit(f"{files[k]['npz'].name}: positions differ from the control run")
        codes, user_idx = np.unique(users, return_inverse=True)
        n_users = len(codes)
        rng = np.random.default_rng(BOOT_SEED)
        weights = rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=B).astype(np.float32)

        auc_check, exact, boot, prac, binning_error = [], {}, {}, {}, {}
        for (arm, s), d in data.items():
            p = d["probs"]
            exact[(arm, s)] = _auc_exact(y, p)
            stored = _stored_auc(files[(arm, s)]["stored_csv"])
            auc_check.append(
                {"run": files[(arm, s)]["npz"].stem, "rescored": exact[(arm, s)], "stored": stored,
                 "ok": stored is not None and abs(stored - exact[(arm, s)]) <= AUC_TOL}
            )
            pos, neg = _binned_counts(y, p, user_idx, n_users)
            binned = float(_auc_from_hist(pos.sum(axis=0, keepdims=True), neg.sum(axis=0, keepdims=True))[0])
            binning_error[(arm, s)] = abs(binned - exact[(arm, s)])
            boot[(arm, s)] = _auc_from_hist(weights @ pos, weights @ neg)
            prac[(arm, s)] = _practical(y, p)
        bad = [a["run"] for a in auc_check if not a["ok"]]
        seeds = [s for s in SEEDS_10
                 if files[(c["on"], s)]["npz"].stem not in bad and files[(c["control"], s)]["npz"].stem not in bad]
        d_exact = [exact[(c["on"], s)] - exact[(c["control"], s)] for s in seeds]
        d_boot = np.stack([boot[(c["on"], s)] - boot[(c["control"], s)] for s in seeds])
        mean_boot = d_boot.mean(axis=0)
        flips = [
            float(((data[(c["on"], s)]["probs"] >= 0.5) != (data[(c["control"], s)]["probs"] >= 0.5)).mean())
            for s in seeds
        ]
        results.append(
            {
                **c,
                "seeds_used": seeds,
                "excluded_runs": bad,
                "auc_check": auc_check,
                "n_learners": int(n_users),
                "n_positions": int(len(y)),
                "mean_delta_exact": float(np.mean(d_exact)),
                "max_abs_binned_minus_exact_auc": max(binning_error.values()),
                "boot_ci95": [float(np.quantile(mean_boot, 0.025)), float(np.quantile(mean_boot, 0.975))],
                "boot_sd_mean": float(mean_boot.std(ddof=1)),
                "learner_var_single_pair": float(np.mean(d_boot.var(axis=1, ddof=1))),
                "practical_on_minus_control": {
                    key: float(np.mean([prac[(c["on"], s)][key] - prac[(c["control"], s)][key] for s in seeds]))
                    for key in ("log_loss", "brier", "ece15", "accuracy")
                },
                "decision_flip_share": float(np.mean(flips)),
            }
        )
        print(f"{c['setting']} fold {c['fold']} {c['contrast']}: "
              f"delta={np.mean(d_exact) * 1e3:+.2f} boot95=[{results[-1]['boot_ci95'][0] * 1e3:+.2f}, "
              f"{results[-1]['boot_ci95'][1] * 1e3:+.2f}] flips={np.mean(flips):.4f}")
    OUT_JSON.write_text(
        json.dumps({"preregistration": "docs/preregistration_round3.md", "B": B, "seed": BOOT_SEED,
                    "n_bins": N_BINS, "rows": results}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT_JSON}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dump", "analyse"), required=True)
    parser.add_argument("--only", default=None, help="Restrict dump to one setting name, e.g. XES3G5M")
    args = parser.parse_args()
    return stage_dump(args.only) if args.stage == "dump" else stage_analyse()


if __name__ == "__main__":
    raise SystemExit(main())
