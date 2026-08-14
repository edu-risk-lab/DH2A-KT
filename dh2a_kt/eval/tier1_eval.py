"""Tier 1 evaluation helpers for manipulation check (M6)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from dh2a_kt.eval.manipulation_check import ManipulationCheckResult, run_manipulation_check
from dh2a_kt.hyperedge.construction import Hyperedge
from dh2a_kt.hyperedge.indexing import hyperedge_index_from_list

logger = logging.getLogger(__name__)

try:
    import torch

    from dh2a_kt.train.tier1 import (
        ConceptPrerequisiteSpec,
        TrainedFold,
        evaluate_auc,
        train_fold,
    )

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def make_eval_auc_fn(trained: TrainedFold) -> Callable[[list[Hyperedge]], float]:
    """Return a callback that scores DH2-KT with a (possibly destroyed) hyperedge set."""

    def eval_auc_fn(hyperedges: list[Hyperedge]) -> float:
        hyperedge_index = hyperedge_index_from_list(hyperedges, trained.kc_to_idx)
        hyperedge_index = {k: v.to(trained.device) for k, v in hyperedge_index.items()}
        auc, _ = evaluate_auc(trained.model, trained.eval_loader, hyperedge_index, trained.device)
        return float(auc) if np.isfinite(auc) else float("nan")

    return eval_auc_fn


def run_manipulation_check_for_fold(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    e_pre: pd.DataFrame,
    *,
    fold: int,
    budget,
    device: str | torch.device = "cpu",
    hidden_dim: int = 128,
    n_hypergraph_layers: int = 2,
    graph_dropout: float = 0.0,
    graph_sensitivity_weight: float = 0.0,
    graph_sensitivity_margin: float = 0.05,
    graph_sensitivity_p: float = 0.9,
    hyperedge_spec: ConceptPrerequisiteSpec | None = None,
    session_hyperedges: list[Hyperedge] | None = None,
    architecture: str = "v2",
    diffusion_alpha: float = 0.5,
    max_users: int | None = None,
    p: float = 0.90,
    operator: str = "node_drop",
    seed: int = 42,
    trained: TrainedFold | None = None,
) -> ManipulationCheckResult:
    if not _TORCH_AVAILABLE:
        raise ImportError("run_manipulation_check_for_fold requires PyTorch")

    spec = hyperedge_spec or ConceptPrerequisiteSpec(fold=fold)
    if trained is None:
        trained = train_fold(
            train_df,
            eval_df,
            e_pre,
            budget,
            device=device,
            hidden_dim=hidden_dim,
            n_hypergraph_layers=n_hypergraph_layers,
            graph_dropout=graph_dropout,
            graph_sensitivity_weight=graph_sensitivity_weight,
            graph_sensitivity_margin=graph_sensitivity_margin,
            graph_sensitivity_p=graph_sensitivity_p,
            hyperedge_spec=spec,
            session_hyperedges=session_hyperedges,
            architecture=architecture,
            diffusion_alpha=diffusion_alpha,
            max_users=max_users,
        )
    eval_fn = make_eval_auc_fn(trained)
    return run_manipulation_check(
        trained.clean_hyperedges,
        eval_auc_fn=eval_fn,
        p=p,
        operator=operator,
        seed=seed,
    )


def write_manipulation_check_result(
    result: ManipulationCheckResult,
    *,
    dataset: str,
    fold: int,
    output_path: Path,
    extra: dict | None = None,
) -> Path:
    payload = {
        "dataset": dataset,
        "fold": fold,
        **asdict(result),
        **(extra or {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "manipulation_check passes=%s auc_drop=%.4f verdict=%s",
        result.passes_manipulation_check,
        result.auc_drop,
        result.verdict,
    )
    return output_path
