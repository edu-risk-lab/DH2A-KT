#!/usr/bin/env python3
"""Hau B6: join DH² vs pyKT scored rows and list the 35-row gap.

Does not train. Rebuilds DH² scored keys from the same clean test walk
(mask-repeats, chunked L=400) and joins a pyKT ``test_predictions.npz``
that holds ``us`` / ``cs`` / ``ts`` / ``ps``.

    python scripts/39_b6_join_prediction_ids.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "results" / "tables" / "b6_id_join_summary.json"
MANIFEST = REPO_ROOT / "results" / "tables" / "b6_unmatched_dh2_rows.csv"
DEFAULT_DH2 = REPO_ROOT / "results" / "checkpoints" / "xes3g5m_fold0_p0_dt_on_s42.pt"
DEFAULT_PYKT = REPO_ROOT / "results" / "pykt_work_clean" / "xes3g5m" / "fold_0"


def _find_pykt_npz(root: Path) -> Path | None:
    candidates = sorted(root.glob("**/C_gikt_clean_L400*/test_predictions.npz"))
    for path in candidates:
        if "s42" in str(path):
            return path
    return candidates[0] if candidates else None


def _dh2_scored_keys(checkpoint: Path, device: str) -> pd.DataFrame:
    from dh2a_kt.hyperedge.p0_inputs import (
        get_fold_splits,
        load_configs,
        load_interactions_with_ids,
    )
    from dh2a_kt.models.dh2_kt import empty_hyperedge_index
    from dh2a_kt.train.checkpoint import load_trained_fold
    from dh2a_kt.train.greykt import make_sequence_loader
    from dh2a_kt.train.tier1 import next_step_mask, resolve_training_budget

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
    trained = load_trained_fold(checkpoint, device=device)
    index = trained.clean_hyperedge_index
    if not trained.clean_hyperedges:
        index = empty_hyperedge_index(
            trained.device, kinds=tuple(trained.model.config.hyperedge_kinds)
        )
    loader = make_sequence_loader(
        test_df, trained, budget, shuffle=False, window_mode="chunked"
    )
    rev_item = {idx: raw for raw, idx in trained.item_to_idx.items()}
    rev_kc = {idx: raw for raw, idx in trained.kc_to_idx.items()}
    rows: list[dict[str, object]] = []
    import torch

    trained.model.eval()
    with torch.no_grad():
        for batch in loader:
            lengths = batch["length"].to(trained.device)
            repeats = batch["is_repeat"].to(trained.device)
            users = batch["user_ids"].cpu().numpy()
            items = batch["exercise_ids"].cpu().numpy()
            kcs = batch["concept_ids"].cpu().numpy()
            labels = batch["correct"].cpu().numpy()
            mask = next_step_mask(
                labels.shape[1] - 1, lengths, is_repeat=repeats
            ).cpu().numpy()
            for b in range(mask.shape[0]):
                for t in range(mask.shape[1]):
                    if not mask[b, t]:
                        continue
                    tgt = t + 1
                    rows.append(
                        {
                            "user_id": int(users[b]),
                            "item_id": int(rev_item.get(int(items[b, tgt]), -1)),
                            "kc_id": int(rev_kc.get(int(kcs[b, tgt]), -1)),
                            "label": int(labels[b, tgt]),
                        }
                    )
    return pd.DataFrame(rows)


def _occurrence_key(frame: pd.DataFrame, user_col: str, kc_col: str) -> pd.Series:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    keys = []
    for user, kc in zip(frame[user_col].tolist(), frame[kc_col].tolist(), strict=True):
        pair = (int(user), int(kc))
        counts[pair] += 1
        keys.append(f"{pair[0]}|{pair[1]}|{counts[pair]}")
    return pd.Series(keys, index=frame.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dh2-checkpoint", type=Path, default=DEFAULT_DH2)
    parser.add_argument("--pykt-npz", type=Path, default=None)
    args = parser.parse_args()

    ckpt = REPO_ROOT / args.dh2_checkpoint if not args.dh2_checkpoint.is_absolute() else args.dh2_checkpoint
    if not ckpt.is_file():
        print(f"SKIP: missing DH2 checkpoint {ckpt}", flush=True)
        return 2
    npz_path = args.pykt_npz or _find_pykt_npz(DEFAULT_PYKT)
    if npz_path is None or not Path(npz_path).is_file():
        print("SKIP: no pyKT test_predictions.npz under results/pykt_work_clean", flush=True)
        return 2

    dh2 = _dh2_scored_keys(ckpt, args.device)
    data = np.load(npz_path)
    missing = {"ts", "ps", "us", "cs"} - set(data.files)
    if missing:
        raise SystemExit(f"{npz_path} missing {sorted(missing)}; holds {data.files}")
    pykt = pd.DataFrame(
        {
            "user_id": np.asarray(data["us"]).ravel().astype(np.int64),
            "kc_id": np.asarray(data["cs"]).ravel().astype(np.int64),
            "label": np.asarray(data["ts"]).ravel().astype(np.int64),
            "prob": np.asarray(data["ps"]).ravel().astype(np.float64),
        }
    )
    dh2["join_key"] = _occurrence_key(dh2, "user_id", "kc_id")
    pykt["join_key"] = _occurrence_key(pykt, "user_id", "kc_id")
    merged = dh2.merge(pykt, on="join_key", how="outer", suffixes=("_dh2", "_pykt"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    only_dh2 = merged[merged["_merge"] == "left_only"]
    only_pykt = merged[merged["_merge"] == "right_only"]
    label_mismatch = int((both["label_dh2"] != both["label_pykt"]).sum()) if len(both) else 0
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    only_dh2.to_csv(MANIFEST, index=False)
    payload = {
        "dh2_checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "pykt_npz": str(Path(npz_path)),
        "n_dh2": int(len(dh2)),
        "n_pykt": int(len(pykt)),
        "n_joined": int(len(both)),
        "n_only_dh2": int(len(only_dh2)),
        "n_only_pykt": int(len(only_pykt)),
        "label_mismatch_on_join": label_mismatch,
        "join_key": "user|kc|occurrence_in_scored_set",
        "expected_only_dh2": 35,
        "note": (
            "Join is occurrence-of-(user,kc) in the scored set, not a native "
            "attempt_id. Headline twins may stay on native n; baseline gaps "
            "should use n_joined after this file exists."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
