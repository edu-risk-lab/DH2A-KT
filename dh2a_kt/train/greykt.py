"""Train / evaluate GreyKT on top of the existing DH2-KT fold machinery."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dh2a_kt.train.tier1 import TrainingBudget

if TYPE_CHECKING:
    from dh2a_kt.train.tier1 import TrainedFold

logger = logging.getLogger(__name__)


@dataclass
class GreyKTEvalReport:
    fold: int
    variant: str
    n_predictions: int
    fused_auc: float
    black_auc: float
    white_auc: float
    fused_nll: float
    black_nll: float
    fused_brier: float
    black_brier: float
    fused_ece: float
    black_ece: float
    mean_gate: float
    prior_mean: float
    max_hops: int
    temperature: float
    use_absolute_backoff: bool
    grid: dict
    diagnostic_verdict: str | None = None


def greykt_config_from_dh2(
    *,
    n_concepts: int,
    n_exercises: int,
    hidden_dim: int,
    n_hypergraph_layers: int,
    dropout: float,
    prior_mean: float,
    max_hops: int,
    greykt_cfg: dict | None = None,
    temperature: float = 1.0,
    use_absolute_backoff: bool = False,
    embed_dim: int | None = None,
    hyperedge_kinds: tuple[str, ...] = ("concept_prerequisite",),
):
    from dh2a_kt.models.greykt import GreyKTConfig

    g = greykt_cfg or {}
    return GreyKTConfig(
        n_concepts=n_concepts,
        n_exercises=n_exercises,
        embed_dim=int(embed_dim if embed_dim is not None else hidden_dim),
        hidden_dim=hidden_dim,
        n_hypergraph_layers=n_hypergraph_layers,
        dropout=dropout,
        hyperedge_kinds=tuple(hyperedge_kinds),
        prior_mean=prior_mean,
        prior_strength=float(g.get("prior_strength", 4.0)),
        kappa_w=g.get("kappa_w"),
        max_hops=int(g.get("max_hops", max_hops)),
        hop_decay=float(g.get("hop_decay", 0.5)),
        recency_decay=float(g.get("recency_decay", 0.98)),
        kappa_b=float(g.get("kappa_b", 4.0)),
        temperature=float(temperature),
        use_absolute_backoff=bool(use_absolute_backoff),
        kappa_r=float(g.get("kappa_r", 1.0)),
        backoff_prob=g.get("backoff_prob"),
    )


def make_sequence_loader(df: pd.DataFrame, trained: "TrainedFold", budget: TrainingBudget, *, shuffle: bool):
    from torch.utils.data import DataLoader

    from dh2a_kt.train.tier1 import UserSequenceDataset, _collate

    return DataLoader(
        UserSequenceDataset(
            df, trained.kc_to_idx, trained.item_to_idx, max_seq_len=budget.max_seq_len
        ),
        batch_size=budget.batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )


def wrap_trained_fold(
    trained: "TrainedFold",
    *,
    prior_mean: float,
    max_hops: int,
    greykt_cfg: dict | None = None,
    temperature: float = 1.0,
    use_absolute_backoff: bool = False,
):
    from dh2a_kt.models.greykt import GreyKT

    bb = trained.model
    cfg = greykt_config_from_dh2(
        n_concepts=bb.config.n_concepts,
        n_exercises=bb.config.n_exercises,
        hidden_dim=bb.config.hidden_dim,
        n_hypergraph_layers=bb.config.n_hypergraph_layers,
        dropout=bb.config.dropout,
        prior_mean=prior_mean,
        max_hops=max_hops,
        greykt_cfg=greykt_cfg,
        temperature=temperature,
        use_absolute_backoff=use_absolute_backoff,
        embed_dim=bb.config.embed_dim,
        hyperedge_kinds=tuple(bb.config.hyperedge_kinds),
    )
    model = GreyKT(cfg)
    model.black_box.load_state_dict(bb.state_dict())
    model = model.to(trained.device)
    return model


def enforce_cb_diagnostic(path: Path, *, force: bool = False) -> str | None:
    """Refuse wrap/train when the C_B diagnostic is NOT_SUPPORTED.

    Returns the verdict string, or None when ``force`` skips a missing file.
    """
    import json

    path = Path(path)
    if not path.exists():
        if force:
            logger.warning("No diagnostic JSON at %s; --force continues", path)
            return None
        raise FileNotFoundError(
            f"Missing C_B diagnostic: {path}. Run --mode diagnose first, or pass --force."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = str(payload.get("verdict", "UNKNOWN"))
    logger.info("diagnostic verdict=%s (%s)", verdict, path)
    if verdict == "NOT_SUPPORTED" and not force:
        raise RuntimeError(
            "C_B diagnostic is NOT_SUPPORTED. Do not train GreyKT on this "
            "signal; change the black-box confidence (MC-dropout / ensemble) "
            "instead. Pass --force to override."
        )
    return verdict


def fit_temperature_on_loader(model, trained: "TrainedFold", loader, prereq_edge_index, concept_train_freq) -> float:
    """Fit v4a T on a held-out loader (validation only — never train or test)."""
    import torch

    from dh2a_kt.models.greykt import fit_temperature
    from dh2a_kt.train.tier1 import precompute_concept_states

    model.eval()
    logits_acc: list = []
    labels_acc: list = []
    with torch.no_grad():
        states = precompute_concept_states(
            model.black_box, trained.clean_hyperedge_index, trained.device
        )
        for batch in loader:
            lengths = batch["lengths"].to(trained.device)
            out = model(_to_grey_batch(batch, trained, prereq_edge_index, concept_train_freq, states))
            mask = torch.arange(out.black_logits.size(1) - 1, device=trained.device).unsqueeze(0) < (
                lengths.unsqueeze(1) - 1
            )
            logits_acc.append(out.black_logits[:, :-1][mask].cpu())
            labels_acc.append(batch["correct"][:, 1:][mask.cpu()].cpu())
    if not logits_acc:
        return 1.0
    t = fit_temperature(torch.cat(logits_acc), torch.cat(labels_acc))
    model.config.temperature = t
    logger.info("fitted temperature T=%.4f (v4a)", t)
    return t


def _to_grey_batch(batch: dict, trained: "TrainedFold", prereq_edge_index, concept_train_freq, states):
    from dh2a_kt.models.greykt import GreyKTBatch

    device = trained.device
    return GreyKTBatch(
        concept_ids=batch["concept_ids"].to(device),
        exercise_ids=batch["exercise_ids"].to(device),
        responses=batch["responses"].to(device),
        hyperedge_index={k: v.to(device) for k, v in trained.clean_hyperedge_index.items()},
        concept_states=states,
        prereq_edge_index=prereq_edge_index.to(device),
        concept_train_freq=concept_train_freq.to(device),
    )


def next_step_bce(probs, targets, lengths):
    import torch
    from torch.nn import functional as F

    if probs.size(1) < 2:
        return probs.sum() * 0.0
    pred = probs[:, :-1]
    label = targets[:, 1:]
    mask = torch.arange(pred.size(1), device=pred.device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
    if not mask.any():
        return pred.sum() * 0.0
    return F.binary_cross_entropy(pred[mask].clamp(1e-6, 1.0 - 1e-6), label[mask])


def collect_greykt_outputs(model, trained: "TrainedFold", prereq_edge_index, concept_train_freq):
    import torch

    from dh2a_kt.models.greykt import brier_score, expected_calibration_error, negative_log_likelihood, stratified_metrics_2d
    from dh2a_kt.train.tier1 import precompute_concept_states

    model.eval()
    fused: list[float] = []
    black: list[float] = []
    white: list[float] = []
    gates: list[float] = []
    cb: list[float] = []
    cw: list[float] = []
    labels: list[float] = []
    device = trained.device
    with torch.no_grad():
        states = precompute_concept_states(model.black_box, trained.clean_hyperedge_index, device)
        for batch in trained.eval_loader:
            lengths = batch["lengths"].to(device)
            out = model(_to_grey_batch(batch, trained, prereq_edge_index, concept_train_freq, states))
            mask = torch.arange(out.probs.size(1) - 1, device=device).unsqueeze(0) < (
                lengths.unsqueeze(1) - 1
            )
            target = batch["correct"].to(device)[:, 1:]
            fused.extend(out.probs[:, :-1][mask].cpu().tolist())
            black.extend(out.black_probs[:, :-1][mask].cpu().tolist())
            white.extend(out.white_prob[:, :-1][mask].cpu().tolist())
            gates.extend(out.gate[:, :-1][mask].cpu().tolist())
            cb.extend(out.black_confidence[:, :-1][mask].cpu().tolist())
            cw.extend(out.white_confidence[:, :-1][mask].cpu().tolist())
            labels.extend(target[mask].cpu().tolist())

    ft = torch.tensor(fused)
    bt = torch.tensor(black)
    wt = torch.tensor(white)
    yt = torch.tensor(labels)
    from sklearn.metrics import roc_auc_score

    def _auc(p):
        arr = np.asarray(p)
        y = np.asarray(labels)
        if len(arr) == 0 or len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, arr))

    grid = stratified_metrics_2d(
        ft,
        bt,
        torch.tensor(cb),
        torch.tensor(cw),
        yt,
        n_buckets=3,
    )
    return GreyKTEvalReport(
        fold=-1,
        variant="v3",
        n_predictions=len(labels),
        fused_auc=_auc(fused),
        black_auc=_auc(black),
        white_auc=_auc(white),
        fused_nll=negative_log_likelihood(ft, yt),
        black_nll=negative_log_likelihood(bt, yt),
        fused_brier=brier_score(ft, yt),
        black_brier=brier_score(bt, yt),
        fused_ece=expected_calibration_error(ft, yt),
        black_ece=expected_calibration_error(bt, yt),
        mean_gate=float(np.mean(gates)) if gates else float("nan"),
        prior_mean=float(model.config.prior_mean),
        max_hops=int(model.config.max_hops),
        temperature=float(model.config.temperature),
        use_absolute_backoff=bool(model.config.use_absolute_backoff),
        grid=grid,
    )


def train_greykt_one_fold(
    model,
    trained: "TrainedFold",
    train_loader,
    prereq_edge_index,
    concept_train_freq,
    budget: TrainingBudget,
    *,
    graph_dropout: float = 0.0,
    graph_sensitivity_weight: float = 0.0,
    graph_sensitivity_margin: float = 0.05,
    graph_sensitivity_p: float = 0.9,
) -> None:
    import torch

    from dh2a_kt.hyperedge.indexing import destroyed_hyperedge_index
    from dh2a_kt.models.dh2_kt import DH2KTBatch, empty_hyperedge_index
    from dh2a_kt.train.tier1 import graph_sensitivity_loss, precompute_concept_states

    device = trained.device
    optimizer = torch.optim.Adam(model.black_box.parameters(), lr=budget.lr)
    model.train()
    full_index = {k: v.to(device) for k, v in trained.clean_hyperedge_index.items()}
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(budget.epochs):
        empty_states = precompute_concept_states(
            model.black_box, empty_hyperedge_index(device), device, enable_grad=False
        )
        ablated_index = None
        if graph_sensitivity_weight > 0.0 and trained.clean_hyperedges:
            ablation_seed = int(torch.randint(0, 2**31, (1,)).item())
            ablated_index = destroyed_hyperedge_index(
                trained.clean_hyperedges,
                trained.kc_to_idx,
                p=graph_sensitivity_p,
                seed=ablation_seed,
                device=device,
            )
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            lengths = batch["lengths"].to(device)
            use_empty = graph_dropout > 0.0 and torch.rand(1).item() < graph_dropout
            states = empty_states if use_empty else precompute_concept_states(
                model.black_box, full_index, device, enable_grad=True
            )
            grey_batch = _to_grey_batch(batch, trained, prereq_edge_index, concept_train_freq, states)
            optimizer.zero_grad()
            out = model(grey_batch)
            loss = next_step_bce(out.probs, batch["correct"].to(device), lengths)
            if graph_sensitivity_weight > 0.0 and ablated_index is not None and n_batches % 5 == 0:
                logits_full = model.black_box(
                    DH2KTBatch(
                        concept_ids=grey_batch.concept_ids,
                        exercise_ids=grey_batch.exercise_ids,
                        responses=grey_batch.responses,
                        hyperedge_index=full_index,
                        concept_states=states,
                    )
                )
                logits_ablated = model.black_box(
                    DH2KTBatch(
                        concept_ids=grey_batch.concept_ids,
                        exercise_ids=grey_batch.exercise_ids,
                        responses=grey_batch.responses,
                        hyperedge_index=ablated_index,
                    )
                )
                loss = loss + graph_sensitivity_weight * graph_sensitivity_loss(
                    logits_full, logits_ablated, lengths, margin=graph_sensitivity_margin
                )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.black_box.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        logger.info("greykt epoch=%s/%s loss=%.4f", epoch + 1, budget.epochs, mean_loss)
        if n_batches > 0 and mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("greykt restored best epoch=%s train_loss=%.4f", best_epoch, best_loss)


def write_greykt_report(report: GreyKTEvalReport, path: Path) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path
