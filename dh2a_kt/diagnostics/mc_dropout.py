"""MC-dropout black-box confidence for GreyKT's reliability gate.

DH2-KT's only dropout is in the hypergraph encoder (``F.dropout`` in
``_encode_concepts``). The temporal DualGatedUpdate path is deterministic
given concept states. MC-dropout therefore *must* re-encode concepts with
``model.train()`` and ``concept_states=None`` on every sample; reusing
eval-mode precomputed states yields std = 0.

The deployed prediction used for error remains eval-mode (dropout off).
C_B is a detached function of the MC std, never a learned parameter.
"""

from __future__ import annotations

import logging

import numpy as np

from dh2a_kt.diagnostics.black_confidence import confidence_from_mc_std

logger = logging.getLogger(__name__)


def mc_dropout_std_for_batch(
    model,
    batch: dict,
    hyperedge_index: dict,
    device,
    n_samples: int,
):
    """Std of sigmoid(logits) over ``n_samples`` stochastic graph encodes.

    Shape ``(B, T)``. Restores the model's previous train/eval mode.
    """
    import torch

    from dh2a_kt.models.dh2_kt import DH2KTBatch

    if n_samples < 2:
        raise ValueError("mc_dropout needs n_samples >= 2")

    was_training = model.training
    dh2 = DH2KTBatch(
        concept_ids=batch["concept_ids"].to(device),
        exercise_ids=batch["exercise_ids"].to(device),
        responses=batch["responses"].to(device),
        hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
        concept_states=None,
    )
    samples = []
    model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(dh2).squeeze(-1)
            samples.append(torch.sigmoid(logits))
    model.train(was_training)
    stacked = torch.stack(samples, dim=0)
    return stacked.std(dim=0, unbiased=False)


def confidence_from_mc_std_torch(std):
    import torch

    return (1.0 - 2.0 * std).clamp(0.0, 1.0)


def collect_eval_probs_and_mc_std(trained, n_samples: int = 8):
    """Eval-mode probabilities plus MC-dropout std, aligned next-step mask.

    Returns ``(probs, labels, concept_ids, stds)`` as 1-d numpy arrays.
    """
    import torch

    from dh2a_kt.models.dh2_kt import DH2KTBatch
    from dh2a_kt.train.tier1 import precompute_concept_states

    model = trained.model
    device = trained.device
    hyperedge_index = trained.clean_hyperedge_index

    probs: list[float] = []
    labels: list[float] = []
    concepts: list[int] = []
    stds: list[float] = []

    model.eval()
    with torch.no_grad():
        eval_states = precompute_concept_states(model, hyperedge_index, device)
        for batch in trained.eval_loader:
            lengths = batch["lengths"].to(device)
            dh2_eval = DH2KTBatch(
                concept_ids=batch["concept_ids"].to(device),
                exercise_ids=batch["exercise_ids"].to(device),
                responses=batch["responses"].to(device),
                hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
                concept_states=eval_states,
            )
            logits = model(dh2_eval)
            pred = torch.sigmoid(logits[:, :-1, 0])
            target = batch["correct"].to(device)[:, 1:]
            target_concepts = batch["concept_ids"].to(device)[:, 1:]
            mask = torch.arange(pred.size(1), device=device).unsqueeze(0) < (
                lengths.unsqueeze(1) - 1
            )
            std = mc_dropout_std_for_batch(
                model, batch, hyperedge_index, device, n_samples
            )[:, :-1]
            probs.extend(pred[mask].cpu().tolist())
            labels.extend(target[mask].cpu().tolist())
            concepts.extend(target_concepts[mask].cpu().tolist())
            stds.extend(std[mask].cpu().tolist())

    std_arr = np.asarray(stds, dtype=float)
    logger.info(
        "MC-dropout n_samples=%s std: min=%.5f median=%.5f max=%.5f",
        n_samples,
        float(std_arr.min()) if std_arr.size else float("nan"),
        float(np.median(std_arr)) if std_arr.size else float("nan"),
        float(std_arr.max()) if std_arr.size else float("nan"),
    )
    return (
        np.asarray(probs, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(concepts, dtype=np.int64),
        std_arr,
    )


def cb_from_mc_std(std) -> np.ndarray:
    return confidence_from_mc_std(std)
