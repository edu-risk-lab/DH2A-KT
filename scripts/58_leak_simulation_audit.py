#!/usr/bin/env python3
"""Leakage simulation report (docs/preregistration_round2.md, Section 6).

Stage ``audit`` (no training, CPU): audit statistics for both synthetic
classes on the XES3G5M fold-0 training split, for the leaky arm, its clean
twin, and graded contamination fractions.

Stage ``auc`` (needs the ``leak-sim`` runs of scripts/50_credit_ladder_plan.py
and their checkpoints): validation AUC from the run CSVs, test-only AUC
rescored from each checkpoint, and leaky-minus-clean inflation per class.
For the intervention-timing class it also runs the data-only residual
check: mean of y_{t+1} - p_{t+1} on fit-split learners, split by the leaky
hint flag at t, with p from the clean-twin (no-leak) model and a learner
bootstrap 95% interval.

    python scripts/58_leak_simulation_audit.py --stage audit
    python scripts/58_leak_simulation_audit.py --stage auc --device cuda

Writes results/tables/leak_simulation_audit.json and
results/tables/leak_simulation_auc.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dh2a_kt.data import leak_sim  # noqa: E402
from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_configs, load_interactions_with_ids  # noqa: E402

CONFIG = REPO / "configs" / "xes3g5m.yaml"
FOLD = 0
VAL_FRAC = 0.1
MAX_SEQ_LEN = 400
SEEDS = (0, 17, 42, 1234, 2024)
T_975_DF4 = 2.7764451051977987
BOOTSTRAP = 1000
GRADED_SEED = leak_sim.HINT_SEED + 2
TABLES = REPO / "results" / "tables"


def _splits():
    from dh2a_kt.train.tier1 import internal_val_users

    dh2_cfg, p0_cfg, _ = load_configs(CONFIG)
    splits = get_fold_splits(load_interactions_with_ids(p0_cfg), p0_cfg, FOLD)
    train = splits["train"]
    fit_users = set(train["user_id"].unique()) - internal_val_users(train, VAL_FRAC)
    return dh2_cfg, splits, fit_users


def stage_audit() -> int:
    _, splits, fit_users = _splits()
    train = splits["train"]
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    both = pd.concat([train, eval_df], ignore_index=True)
    train_rows = np.arange(len(train))
    out: dict = {"fold": FOLD, "fractions": list(leak_sim.FRACTIONS), "n_train_rows": len(train)}

    cm = leak_sim.crossmodal_parts(both, fit_users)
    out["crossmodal"] = {
        "statistic": "support-weighted share of supporting interactions after t or from a held-out learner",
        "leaky": leak_sim.graded_crossmodal_share(cm, train_rows, 1.0, GRADED_SEED),
        "clean": leak_sim.graded_crossmodal_share(cm, train_rows, 0.0, GRADED_SEED),
        "graded": {
            str(f): leak_sim.graded_crossmodal_share(cm, train_rows, f, GRADED_SEED)
            for f in leak_sim.FRACTIONS
        },
        "mean_scalar_train": {
            "leaky": float(cm["leak_value"].iloc[train_rows].mean()),
            "clean": float(cm["clean_value"].iloc[train_rows].mean()),
        },
    }

    hint = leak_sim.hint_parts(both, fit_users)
    out["hint"] = {
        "statistic": "share of hint rows whose selection used an outcome at or after the scored step",
        "fit_rate": hint.attrs["fit_rate"],
        "leaky": leak_sim.graded_hint_rate(hint, train_rows, 1.0, GRADED_SEED),
        "clean": leak_sim.graded_hint_rate(hint, train_rows, 0.0, GRADED_SEED),
        "graded": {
            str(f): leak_sim.graded_hint_rate(hint, train_rows, f, GRADED_SEED)
            for f in leak_sim.FRACTIONS
        },
    }
    path = TABLES / "leak_simulation_audit.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


def _summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    half = T_975_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return {"per_seed": values, "mean": mean, "ci95": [mean - half, mean + half]}


def _dataset(frame, trained, *, aux: bool):
    from dh2a_kt.train.tier1 import UserSequenceDataset

    return UserSequenceDataset(
        frame,
        trained.kc_to_idx,
        trained.item_to_idx,
        max_seq_len=MAX_SEQ_LEN,
        window_mode="chunked",
        include_time_gaps=True,
        include_aux_scalar=aux,
    )


def _residual_check(trained, fit_df: pd.DataFrame, leak_flags: np.ndarray, device) -> dict:
    """Mean y_{t+1} - p_{t+1} by leaky flag at t; p from the clean-twin model."""
    import torch

    from dh2a_kt.models.dh2_kt import DH2KTBatch
    from dh2a_kt.train.tier1 import _collate, aux_kwargs, next_step_mask, precompute_concept_states

    model = trained.model.eval()
    ds_in = _dataset(fit_df, trained, aux=True)
    ds_flag = _dataset(fit_df.assign(aux_scalar=leak_flags), trained, aux=True)
    states = precompute_concept_states(model, trained.clean_hyperedge_index, device)
    users, flags, resid = [], [], []
    with torch.no_grad():
        for lo in range(0, len(ds_in), 64):
            idx = range(lo, min(lo + 64, len(ds_in)))
            batch = _collate([ds_in[i] for i in idx])
            flag_batch = _collate([ds_flag[i] for i in idx])
            lengths = batch["lengths"].to(device)
            logits = model(
                DH2KTBatch(
                    concept_ids=batch["concept_ids"].to(device),
                    exercise_ids=batch["exercise_ids"].to(device),
                    responses=batch["correct"].to(device),
                    hyperedge_index=trained.clean_hyperedge_index,
                    concept_states=states,
                    lengths=lengths,
                    **aux_kwargs(batch, device),
                )
            )
            pred = torch.sigmoid(logits[:, :-1, 0])
            target = batch["correct"].to(device)[:, 1:]
            mask = next_step_mask(pred.size(1), lengths, is_repeat=batch["is_repeat"].to(device))
            flag = flag_batch["aux_scalars"].to(device)[:, :-1]
            user = batch["user_ids"].to(device).unsqueeze(1).expand_as(pred)
            users.append(user[mask].cpu().numpy())
            flags.append(flag[mask].cpu().numpy() > 0.5)
            resid.append((target - pred)[mask].cpu().numpy())
    users_a, flags_a, resid_a = np.concatenate(users), np.concatenate(flags), np.concatenate(resid)
    codes, user_idx = np.unique(users_a, return_inverse=True)
    n_users = len(codes)
    sums = np.zeros((2, n_users))
    counts = np.zeros((2, n_users))
    for f in (0, 1):
        sel = flags_a == bool(f)
        np.add.at(sums[f], user_idx[sel], resid_a[sel])
        np.add.at(counts[f], user_idx[sel], 1.0)
    rng = np.random.default_rng(leak_sim.HINT_SEED + 3)
    weights = rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=BOOTSTRAP)
    out = {}
    for f, name in ((1, "hint"), (0, "no_hint")):
        point = float(sums[f].sum() / counts[f].sum())
        boot = (weights @ sums[f]) / np.maximum(weights @ counts[f], 1.0)
        out[name] = {
            "mean_residual": point,
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "n_positions": int(counts[f].sum()),
        }
    diff = (weights @ sums[1]) / np.maximum(weights @ counts[1], 1.0) - (weights @ sums[0]) / np.maximum(
        weights @ counts[0], 1.0
    )
    out["hint_minus_no_hint"] = {
        "mean": out["hint"]["mean_residual"] - out["no_hint"]["mean_residual"],
        "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
    }
    out["n_learners"] = n_users
    return out


def stage_auc(device_name: str) -> int:
    import torch

    from dh2a_kt.train.checkpoint import load_trained_fold
    from dh2a_kt.train.tier1 import _collate, evaluate_auc

    dh2_cfg, splits, fit_users = _splits()
    device = torch.device(device_name)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    n_valid = len(splits["valid"])
    rows: dict[str, dict[int, dict]] = {}
    residual: dict[int, dict] = {}
    for arm in leak_sim.ARMS:
        train_a, eval_a, _ = leak_sim.attach_leak_scalar(splits["train"], eval_df, arm, fit_users)
        test_a = eval_a.iloc[n_valid:]
        rows[arm] = {}
        for seed in SEEDS:
            tag = f"ladder_leak-sim_{arm}_s{seed}"
            csv_path = TABLES / f"dh2_kt_{tag}_vs_p0.csv"
            ckpt = REPO / "results" / "checkpoints" / f"{dh2_cfg['dataset']}_fold{FOLD}_{tag}.pt"
            if not csv_path.exists() or not ckpt.exists():
                print(f"PENDING {tag}")
                continue
            run = pd.read_csv(csv_path).iloc[0]
            trained = load_trained_fold(ckpt, device=device_name)
            loader = torch.utils.data.DataLoader(
                _dataset(test_a, trained, aux=True), batch_size=64, shuffle=False, collate_fn=_collate
            )
            test_auc, n_test = evaluate_auc(
                trained.model, loader, trained.clean_hyperedge_index, device, mask_repeats=True
            )
            rows[arm][seed] = {
                "val_auc": float(run["val_auc"]),
                "valid_test_auc": float(run["dh2_kt_auc"]),
                "test_auc": float(test_auc),
                "n_test": int(n_test),
            }
            print(f"{tag}: val={run['val_auc']:.6f} test={test_auc:.6f}")
            if arm == "hint-clean":
                fit_df = train_a[train_a["user_id"].isin(fit_users)]
                both = pd.concat([splits["train"], eval_df], ignore_index=True)
                leak_all = leak_sim.hint_parts(both, fit_users)["leak_value"].to_numpy()
                leak_fit = leak_all[: len(splits["train"])][splits["train"]["user_id"].isin(fit_users).to_numpy()]
                residual[seed] = _residual_check(trained, fit_df, leak_fit.astype(np.float32), device)
                print(f"  residual check: {residual[seed]['hint_minus_no_hint']}")

    inflation = {}
    for kind in ("crossmodal", "hint"):
        leaky, clean = rows[f"{kind}-leak"], rows[f"{kind}-clean"]
        common = [s for s in SEEDS if s in leaky and s in clean]
        if len(common) < len(SEEDS):
            inflation[kind] = {"pending_seeds": [s for s in SEEDS if s not in common]}
            continue
        inflation[kind] = {
            metric: _summary([leaky[s][metric] - clean[s][metric] for s in SEEDS])
            for metric in ("val_auc", "test_auc", "valid_test_auc")
        }
    out = {
        "fold": FOLD,
        "seeds": list(SEEDS),
        "runs": {arm: {str(s): v for s, v in by_seed.items()} for arm, by_seed in rows.items()},
        "inflation_leaky_minus_clean": inflation,
        "residual_check_hint_clean_model": {str(s): v for s, v in residual.items()},
        "note": "Synthetic leakage classes; not evidence about any public corpus.",
    }
    path = TABLES / "leak_simulation_auc.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(inflation, indent=2))
    print(f"wrote {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "auc"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "audit":
        return stage_audit()
    return stage_auc(args.device)


if __name__ == "__main__":
    raise SystemExit(main())
