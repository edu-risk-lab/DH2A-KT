"""Wire DH2KT sequence embeddings into PropensityScoreATE (M4).

Until Hint/Video logs exist (M2), treatment is proxied by concept-prerequisite
structure: an interaction is "treated" when its KC is the downstream end of
at least one audited ``E_pre`` edge (a concept with incoming prerequisites).
Outcome is next-step correctness on the train split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dh2a_kt.models.causal_layer import PropensityScoreATE

logger = logging.getLogger(__name__)

try:
    import torch

    from dh2a_kt.models.dh2_kt import DH2KT, DH2KTBatch, DH2KTConfig

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


@dataclass
class CausalDataset:
    confounders: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray


def build_entity_maps(
    interactions: pd.DataFrame,
    e_pre: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    kc_ids = set(interactions["kc_id"].astype(int).tolist())
    kc_ids.update(int(v) for v in e_pre["src_kc"].tolist())
    kc_ids.update(int(v) for v in e_pre["dst_kc"].tolist())
    item_ids = set(interactions["item_id"].astype(int).tolist())

    kc_to_idx = {kc: idx for idx, kc in enumerate(sorted(kc_ids))}
    item_to_idx = {item: idx for idx, item in enumerate(sorted(item_ids))}
    return kc_to_idx, item_to_idx


def hyperedge_index_from_e_pre(
    e_pre: pd.DataFrame,
    kc_to_idx: dict[int, int],
) -> dict[str, torch.Tensor]:
    """Map each audited prerequisite edge to a 2-node hyperedge for PyG."""
    if not _TORCH_AVAILABLE:
        raise ImportError("hyperedge_index_from_e_pre requires PyTorch")

    node_rows: list[int] = []
    edge_rows: list[int] = []
    for he_id, row in enumerate(e_pre.itertuples(index=False)):
        for kc in (int(row.src_kc), int(row.dst_kc)):
            node_rows.append(kc_to_idx[kc])
            edge_rows.append(he_id)
    if not node_rows:
        return {"concept_prerequisite": torch.empty((2, 0), dtype=torch.long)}
    index = torch.tensor([node_rows, edge_rows], dtype=torch.long)
    return {"concept_prerequisite": index}


def treatment_proxy_kc_set(e_pre: pd.DataFrame) -> set[int]:
    """Stand-in for hint exposure until M2: KC with incoming prerequisites."""
    return set(e_pre["dst_kc"].astype(int).tolist())


def build_causal_dataset_from_train(
    train_df: pd.DataFrame,
    e_pre: pd.DataFrame,
    model: DH2KT,
    *,
    max_users: int = 128,
    max_seq_len: int = 20,
    device: str | torch.device = "cpu",
) -> CausalDataset:
    if not _TORCH_AVAILABLE:
        raise ImportError("build_causal_dataset_from_train requires PyTorch")

    kc_to_idx, item_to_idx = build_entity_maps(train_df, e_pre)
    treated_kcs = treatment_proxy_kc_set(e_pre)
    hyperedge_index = hyperedge_index_from_e_pre(e_pre, kc_to_idx)

    user_ids = train_df["user_id"].astype(int).unique()[:max_users]
    confounder_rows: list[np.ndarray] = []
    treatment_rows: list[int] = []
    outcome_rows: list[int] = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for user_id in user_ids:
            seq = (
                train_df[train_df["user_id"] == user_id]
                .sort_values("timestamp")
                .head(max_seq_len)
            )
            if len(seq) < 2:
                continue

            concept_ids = torch.tensor(
                [[kc_to_idx[int(k)] for k in seq["kc_id"].tolist()]],
                dtype=torch.long,
                device=device,
            )
            exercise_ids = torch.tensor(
                [[item_to_idx[int(i)] for i in seq["item_id"].tolist()]],
                dtype=torch.long,
                device=device,
            )
            responses = torch.tensor(
                [[float(c) for c in seq["correct"].tolist()]],
                dtype=torch.float,
                device=device,
            )
            batch = DH2KTBatch(
                concept_ids=concept_ids,
                exercise_ids=exercise_ids,
                responses=responses,
                hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
            )
            hidden = model.encode_sequence(batch)[0].cpu().numpy()
            kc_raw = seq["kc_id"].astype(int).tolist()
            outcomes = seq["correct"].astype(int).tolist()

            for t in range(len(seq) - 1):
                confounder_rows.append(hidden[t])
                treatment_rows.append(int(kc_raw[t] in treated_kcs))
                outcome_rows.append(outcomes[t + 1])

    if not confounder_rows:
        raise ValueError("No valid user sequences found for causal dataset")

    return CausalDataset(
        confounders=np.stack(confounder_rows, axis=0),
        treatment=np.asarray(treatment_rows, dtype=int),
        outcome=np.asarray(outcome_rows, dtype=int),
    )


def estimate_ate_with_dh2kt_confounders(
    causal_data: CausalDataset,
    *,
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> tuple[float, dict]:
    est = PropensityScoreATE(clip_propensity=clip_propensity)
    ate, diagnostics = est.estimate_ate(
        causal_data.confounders,
        causal_data.treatment,
        causal_data.outcome,
    )
    logger.info(
        "ATE=%.4f propensity=[%.3f, %.3f] n_treated=%s n_control=%s",
        ate,
        diagnostics["propensity_min"],
        diagnostics["propensity_max"],
        diagnostics["n_treated"],
        diagnostics["n_control"],
    )
    return ate, diagnostics


def run_causal_pipeline(
    train_df: pd.DataFrame,
    e_pre: pd.DataFrame,
    *,
    max_users: int = 128,
    max_seq_len: int = 20,
    hidden_dim: int = 32,
    device: str | torch.device = "cpu",
) -> tuple[float, dict]:
    """End-to-end M4 helper: DH2KT confounders -> IPW ATE."""
    if not _TORCH_AVAILABLE:
        raise ImportError("run_causal_pipeline requires PyTorch")

    kc_to_idx, item_to_idx = build_entity_maps(train_df, e_pre)
    config = DH2KTConfig(
        n_concepts=len(kc_to_idx),
        n_exercises=len(item_to_idx),
        hidden_dim=hidden_dim,
        embed_dim=hidden_dim,
        n_hypergraph_layers=1,
        dropout=0.0,
        hyperedge_kinds=("concept_prerequisite",),
    )
    model = DH2KT(config)
    causal_data = build_causal_dataset_from_train(
        train_df,
        e_pre,
        model,
        max_users=max_users,
        max_seq_len=max_seq_len,
        device=device,
    )
    return estimate_ate_with_dh2kt_confounders(causal_data)
