"""E5: average LOGITS of three existing v4q_clean seeds; one ensemble row."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dh2a_kt.hyperedge.p0_inputs import (  # noqa: E402
    get_fold_splits,
    load_configs,
    load_interactions_with_ids,
)
from dh2a_kt.models.dh2_kt import DH2KTBatch, empty_hyperedge_index  # noqa: E402
from dh2a_kt.train.checkpoint import load_trained_fold  # noqa: E402
from dh2a_kt.train.greykt import make_sequence_loader  # noqa: E402
from dh2a_kt.train.tier1 import (  # noqa: E402
    kc_set_kwargs,
    next_step_mask,
    precompute_concept_states,
    resolve_training_budget,
    responses_for_model,
)

CKPTS = [
    ("s42", REPO / "results/checkpoints/xes3g5m_fold0_v4q_clean.pt"),
    ("s17", REPO / "results/checkpoints/xes3g5m_fold0_v4q_clean_s17.pt"),
    ("s1234", REPO / "results/checkpoints/xes3g5m_fold0_v4q_clean_s1234.pt"),
]


@torch.no_grad()
def collect_logits(model, loader, hyperedge_index, device, *, mask_repeats: bool):
    model.eval()
    logits_out, labels_out = [], []
    concept_states = precompute_concept_states(model, hyperedge_index, device)
    for batch in loader:
        lengths = batch["lengths"].to(device)
        dh2_batch = DH2KTBatch(
            concept_ids=batch["concept_ids"].to(device),
            exercise_ids=batch["exercise_ids"].to(device),
            responses=responses_for_model(model, batch).to(device),
            hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
            concept_states=concept_states,
            lengths=lengths,
            **kc_set_kwargs(batch, device),
        )
        logits = model(dh2_batch)[:, :-1, 0]
        target = batch["correct"].to(device)[:, 1:]
        mask = next_step_mask(
            logits.size(1),
            lengths,
            is_repeat=batch["is_repeat"].to(device) if mask_repeats else None,
        )
        logits_out.append(logits[mask].detach().cpu().numpy())
        labels_out.append(target[mask].detach().cpu().numpy())
    return np.concatenate(logits_out), np.concatenate(labels_out)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, p0_cfg, _ = load_configs(REPO / "configs/xes3g5m.yaml")
    budget = resolve_training_budget(p0_cfg, reference_model="gkt", lr=0.001, max_seq_len=200)
    budget.batch_size = 64
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, 0)
    # Match clean SOTA table: test-only + chunked + mask_repeats
    eval_df = splits["test"]

    all_logits = []
    labels = None
    rows: list[dict] = []
    for name, path in CKPTS:
        print(f"loading {name} <- {path}")
        trained = load_trained_fold(path, device=device)
        index = trained.clean_hyperedge_index
        if not trained.clean_hyperedges:
            index = empty_hyperedge_index(
                trained.device, kinds=tuple(trained.model.config.hyperedge_kinds)
            )
        loader = make_sequence_loader(
            eval_df, trained, budget, shuffle=False, window_mode="chunked"
        )
        logits, labs = collect_logits(
            trained.model, loader, index, trained.device, mask_repeats=True
        )
        auc = float(roc_auc_score(labs, 1 / (1 + np.exp(-logits))))
        print(f"  [{name}] single AUC={auc:.6f} n={len(labs)}")
        rows.append(
            {
                "model": f"v4q_clean_{name}",
                "kind": "single",
                "auc": auc,
                "n_scored": len(labs),
            }
        )
        all_logits.append(logits)
        if labels is None:
            labels = labs
        else:
            assert len(labs) == len(labels)

    mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    ens_auc = float(roc_auc_score(labels, 1 / (1 + np.exp(-mean_logits))))
    print(f"[ensemble_of_3_seeds] logit-avg AUC={ens_auc:.6f} n={len(labels)}")
    rows.append(
        {
            "model": "ensemble_of_3_seeds",
            "kind": "ensemble_logit_avg",
            "auc": ens_auc,
            "n_scored": len(labels),
            "note": "mean of logits from v4q_clean seeds 42/17/1234; test-only chunked mask_repeats; L=200",
        }
    )
    out = REPO / "results/tables/sota_E5_ensemble.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
