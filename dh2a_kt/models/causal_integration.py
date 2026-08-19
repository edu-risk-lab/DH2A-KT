"""Wire DH2KT sequence embeddings into PropensityScoreATE (M4).

Default treatment proxy (XES3G5M / no hint logs): an interaction is "treated"
when its KC is the downstream end of at least one audited ``E_pre`` edge.
On FoundationalASSIST (and similar), use ``build_hint_causal_dataset`` with
real ``hint_used`` / ``hint_count`` as treatment and *next-step* correctness
as outcome (same-step ``discrete_score`` is mechanically tied to hint use).
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


def build_hint_causal_dataset(
    interactions: pd.DataFrame,
    *,
    max_users: int | None = None,
    max_seq_len: int | None = 200,
    seed: int = 42,
) -> CausalDataset:
    """Observational hint ATE rows: treatment=hint at t, outcome=correct at t+1.

    Required columns: ``user_id``, ``item_id``, ``kc_id``, ``timestamp``,
    ``correct``, and either ``hint_used`` or ``hint_count``.

    Confounders are *pre-treatment* summaries from steps ``< t`` only:
    prior accuracy, log prior attempts, same-KC prior accuracy, position.
    """
    required = {"user_id", "item_id", "kc_id", "timestamp", "correct"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")
    df = interactions.copy()
    if "hint_used" not in df.columns:
        if "hint_count" not in df.columns:
            raise ValueError("need hint_used or hint_count")
        df["hint_used"] = (df["hint_count"].fillna(0).astype(int) > 0).astype(int)

    user_ids = df["user_id"].astype(int).unique()
    if max_users is not None and len(user_ids) > max_users:
        rng = np.random.default_rng(seed)
        user_ids = rng.choice(user_ids, size=max_users, replace=False)

    confounder_rows: list[list[float]] = []
    treatment_rows: list[int] = []
    outcome_rows: list[int] = []

    for user_id in user_ids:
        seq = df[df["user_id"] == user_id].sort_values("timestamp")
        if max_seq_len is not None:
            seq = seq.head(max_seq_len)
        if len(seq) < 2:
            continue

        correct = seq["correct"].astype(int).to_numpy()
        hints = seq["hint_used"].astype(int).to_numpy()
        kcs = seq["kc_id"].astype(int).to_numpy()

        for t in range(len(seq) - 1):
            n_prior = t
            prior_acc = float(correct[:t].mean()) if t > 0 else 0.5
            same_kc_mask = kcs[:t] == kcs[t]
            same_kc_acc = (
                float(correct[:t][same_kc_mask].mean()) if same_kc_mask.any() else prior_acc
            )
            confounder_rows.append(
                [
                    prior_acc,
                    float(np.log1p(n_prior)),
                    same_kc_acc,
                    float(t) / max(len(seq) - 1, 1),
                    float(np.log1p(int(seq["item_id"].iloc[t]))),
                ]
            )
            treatment_rows.append(int(hints[t]))
            outcome_rows.append(int(correct[t + 1]))

    if not confounder_rows:
        raise ValueError("No valid user sequences for hint causal dataset")

    return CausalDataset(
        confounders=np.asarray(confounder_rows, dtype=float),
        treatment=np.asarray(treatment_rows, dtype=int),
        outcome=np.asarray(outcome_rows, dtype=int),
    )


def estimate_hint_ate(
    causal_data: CausalDataset,
    *,
    clip_propensity: tuple[float, float] = (0.05, 0.95),
    trim_quantiles: tuple[float, float] | None = (0.1, 0.9),
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[float, dict]:
    """IPW ATE of hint_used_t on correct_{t+1}, plus naive / trimmed / bootstrap CI."""
    ate, diagnostics = estimate_ate_with_dh2kt_confounders(
        causal_data, clip_propensity=clip_propensity
    )
    treated = causal_data.treatment == 1
    control = causal_data.treatment == 0
    naive = float(
        causal_data.outcome[treated].mean() - causal_data.outcome[control].mean()
    ) if treated.any() and control.any() else float("nan")

    trimmed_ate = float("nan")
    n_trimmed = 0
    if trim_quantiles is not None and treated.any() and control.any():
        est = PropensityScoreATE(clip_propensity=clip_propensity)
        p = est.fit(causal_data.confounders, causal_data.treatment).propensity_scores(
            causal_data.confounders
        )
        lo, hi = np.quantile(p, trim_quantiles)
        keep = (p >= lo) & (p <= hi)
        n_trimmed = int(keep.sum())
        if keep.sum() > 10 and causal_data.treatment[keep].sum() > 0 and (
            (1 - causal_data.treatment[keep]).sum() > 0
        ):
            trimmed = CausalDataset(
                confounders=causal_data.confounders[keep],
                treatment=causal_data.treatment[keep],
                outcome=causal_data.outcome[keep],
            )
            trimmed_ate, _ = estimate_ate_with_dh2kt_confounders(
                trimmed, clip_propensity=clip_propensity
            )

    boot_ats: list[float] = []
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        n = len(causal_data.outcome)
        # Subsample large corpora so bootstrap stays tractable.
        boot_n = min(n, 50_000)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=boot_n, replace=True)
            sample = CausalDataset(
                confounders=causal_data.confounders[idx],
                treatment=causal_data.treatment[idx],
                outcome=causal_data.outcome[idx],
            )
            if sample.treatment.sum() == 0 or (1 - sample.treatment).sum() == 0:
                continue
            try:
                b_ate, _ = estimate_ate_with_dh2kt_confounders(
                    sample, clip_propensity=clip_propensity
                )
            except Exception:  # pragma: no cover — rare singular fits
                continue
            if np.isfinite(b_ate):
                boot_ats.append(float(b_ate))
    ci_lo = float(np.quantile(boot_ats, 0.025)) if boot_ats else float("nan")
    ci_hi = float(np.quantile(boot_ats, 0.975)) if boot_ats else float("nan")

    diagnostics = {
        **diagnostics,
        "naive_mean_diff": naive,
        "trimmed_ate_ipw": trimmed_ate,
        "trim_quantiles": list(trim_quantiles) if trim_quantiles else None,
        "n_trimmed_rows": n_trimmed,
        "bootstrap_n": len(boot_ats),
        "bootstrap_ci95": [ci_lo, ci_hi],
        "bootstrap_subsample": min(len(causal_data.outcome), 50_000) if n_bootstrap else 0,
        "treatment": "hint_used_t",
        "outcome": "correct_t_plus_1",
        "n_rows": int(len(causal_data.outcome)),
        "identification": (
            "IPW under no unmeasured confounding given pre-t summaries; "
            "observational KT — report as limited evidence, not proven causal effect"
        ),
    }
    logger.info(
        "hint ATE=%.4f naive_diff=%.4f trimmed=%.4f CI95=[%.4f, %.4f]",
        ate,
        naive,
        trimmed_ate,
        ci_lo,
        ci_hi,
    )
    return ate, diagnostics


def build_entity_maps(
    interactions: pd.DataFrame,
    e_pre: pd.DataFrame,
    extra_kc_ids: set[int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    kc_ids = set(interactions["kc_id"].astype(int).tolist())
    kc_ids.update(int(v) for v in e_pre["src_kc"].tolist())
    kc_ids.update(int(v) for v in e_pre["dst_kc"].tolist())
    if extra_kc_ids:
        kc_ids.update(int(v) for v in extra_kc_ids)
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
