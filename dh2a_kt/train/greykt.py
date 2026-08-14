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
    black_confidence_mode: str = "frequency",
    architecture: str = "v2",
    diffusion_alpha: float = 0.5,
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
        black_confidence_mode=str(black_confidence_mode),
        architecture=str(architecture),
        diffusion_alpha=float(diffusion_alpha),
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


def _collate_with_white(batch_items: list) -> dict:
    from dh2a_kt.train.tier1 import _collate

    out = _collate(batch_items)
    if batch_items and "white_prob" in batch_items[0]:
        import torch

        out["white_prob"] = torch.stack([item["white_prob"] for item in batch_items])
        out["white_confidence"] = torch.stack([item["white_confidence"] for item in batch_items])
    return out


class _WhiteboxCachedDataset:
    """Wraps UserSequenceDataset with precomputed white-box tensors."""

    def __init__(self, base, white_probs, white_confs):
        self.base = base
        self.white_probs = white_probs
        self.white_confs = white_confs

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base[idx])
        item["white_prob"] = self.white_probs[idx]
        item["white_confidence"] = self.white_confs[idx]
        return item


def precompute_whitebox_cache(
    dataset,
    *,
    model,
    prereq_edge_index,
    batch_size: int = 256,
    device=None,
) -> tuple[list, list]:
    """Run white-box once per sequence (no grad). Safe to reuse across epochs."""
    import torch
    from torch.utils.data import DataLoader

    from dh2a_kt.models.greykt import (
        _build_multihop_prereq_lookup,
        _padded_ancestor_tables,
        whitebox_branch,
    )
    from dh2a_kt.train.tier1 import _collate

    dev = device if device is not None else next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    white_probs: list = [None] * len(dataset)
    white_confs: list = [None] * len(dataset)
    cfg = model.config
    lookup = _build_multihop_prereq_lookup(
        prereq_edge_index.to(dev), cfg.n_concepts, cfg.max_hops, cfg.hop_decay
    )
    tables = _padded_ancestor_tables(lookup, cfg.n_concepts, dev)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            concept_ids = batch["concept_ids"].to(dev)
            responses = batch["responses"].to(dev)
            wp, wc = whitebox_branch(
                concept_ids=concept_ids,
                responses=responses,
                prereq_edge_index=prereq_edge_index.to(dev),
                n_concepts=cfg.n_concepts,
                prior_mean=cfg.prior_mean,
                prior_strength=cfg.prior_strength,
                max_hops=cfg.max_hops,
                hop_decay=cfg.hop_decay,
                recency_decay=cfg.recency_decay,
                kappa_w=cfg.resolved_kappa_w(),
                ancestor_tables=tables,
            )
            bsz = concept_ids.size(0)
            for i in range(bsz):
                white_probs[offset + i] = wp[i].detach().cpu()
                white_confs[offset + i] = wc[i].detach().cpu()
            offset += bsz
    assert offset == len(dataset)
    return white_probs, white_confs


def make_whitebox_cached_loader(
    df: pd.DataFrame,
    trained: "TrainedFold",
    budget: TrainingBudget,
    model,
    prereq_edge_index,
    *,
    shuffle: bool,
    cache_batch_size: int = 256,
):
    """Build a train loader with white-box outputs cached on CPU once."""
    from torch.utils.data import DataLoader

    from dh2a_kt.train.tier1 import UserSequenceDataset

    base = UserSequenceDataset(
        df, trained.kc_to_idx, trained.item_to_idx, max_seq_len=budget.max_seq_len
    )
    logger.info("precomputing white-box cache for %d sequences (batch=%s)...", len(base), cache_batch_size)
    white_probs, white_confs = precompute_whitebox_cache(
        base,
        model=model,
        prereq_edge_index=prereq_edge_index,
        batch_size=cache_batch_size,
        device=trained.device,
    )
    logger.info("white-box cache ready (%d sequences)", len(base))
    return DataLoader(
        _WhiteboxCachedDataset(base, white_probs, white_confs),
        batch_size=budget.batch_size,
        shuffle=shuffle,
        collate_fn=_collate_with_white,
    )


def wrap_trained_fold(
    trained: "TrainedFold",
    *,
    prior_mean: float,
    max_hops: int,
    greykt_cfg: dict | None = None,
    temperature: float = 1.0,
    use_absolute_backoff: bool = False,
    black_confidence_mode: str = "frequency",
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
        black_confidence_mode=black_confidence_mode,
        architecture=getattr(bb.config, "architecture", "v2"),
        diffusion_alpha=float(getattr(bb.config, "diffusion_alpha", 0.5)),
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


def _to_grey_batch(
    batch: dict,
    trained: "TrainedFold",
    prereq_edge_index,
    concept_train_freq,
    states,
    black_confidence=None,
):
    from dh2a_kt.models.greykt import GreyKTBatch

    device = trained.device
    white_prob = batch.get("white_prob")
    white_confidence = batch.get("white_confidence")
    return GreyKTBatch(
        concept_ids=batch["concept_ids"].to(device),
        exercise_ids=batch["exercise_ids"].to(device),
        responses=batch["responses"].to(device),
        hyperedge_index={k: v.to(device) for k, v in trained.clean_hyperedge_index.items()},
        concept_states=states,
        prereq_edge_index=prereq_edge_index.to(device),
        concept_train_freq=concept_train_freq.to(device),
        black_confidence=None if black_confidence is None else black_confidence.to(device),
        lengths=None if "lengths" not in batch else batch["lengths"].to(device),
        white_prob=None if white_prob is None else white_prob.to(device),
        white_confidence=None if white_confidence is None else white_confidence.to(device),
    )


def _maybe_mc_confidence(model, batch, trained: "TrainedFold", mc_samples: int):
    if mc_samples < 2:
        return None
    from dh2a_kt.diagnostics.mc_dropout import confidence_from_mc_std_torch, mc_dropout_std_for_batch

    std = mc_dropout_std_for_batch(
        model.black_box, batch, trained.clean_hyperedge_index, trained.device, mc_samples
    )
    return confidence_from_mc_std_torch(std)


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


def collect_greykt_outputs(
    model, trained: "TrainedFold", prereq_edge_index, concept_train_freq, *, mc_samples: int = 0
):
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
            cb_t = _maybe_mc_confidence(model, batch, trained, mc_samples)
            out = model(
                _to_grey_batch(
                    batch,
                    trained,
                    prereq_edge_index,
                    concept_train_freq,
                    states,
                    black_confidence=cb_t,
                )
            )
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


def greykt_checkpoint_dir(
    *,
    dataset: str,
    fold: int,
    variant: str,
    signal: str,
    root: Path | None = None,
) -> Path:
    """Default directory for GreyKT train resume artefacts (latest.pt / best.pt)."""
    base = Path(root) if root is not None else Path("results") / "checkpoints"
    return base / f"greykt_{dataset}_fold{fold}_{variant}_{signal}"


def save_greykt_train_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    best_epoch: int,
    best_loss: float,
    best_state: dict | None,
    meta: dict | None = None,
) -> Path:
    """Persist model + optimizer + best-so-far so train can resume after a crash."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_state": best_state,
        "meta": dict(meta or {}),
    }
    torch.save(payload, path)
    logger.info("wrote GreyKT checkpoint epoch=%s -> %s", epoch, path)
    return path


def load_greykt_train_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    device: str = "cpu",
) -> dict:
    """Load a GreyKT train checkpoint into ``model`` / ``optimizer``.

    Returns the payload (epoch = last *completed* 1-based epoch).
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    logger.info(
        "resumed GreyKT from %s (completed epoch=%s best_epoch=%s best_loss=%.4f)",
        path,
        payload.get("epoch"),
        payload.get("best_epoch"),
        float(payload.get("best_loss", float("nan"))),
    )
    return payload


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
    mc_samples: int = 0,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    checkpoint_meta: dict | None = None,
    use_amp: bool = False,
    freeze_hypergraph: bool = False,
) -> None:
    import torch

    from dh2a_kt.hyperedge.indexing import destroyed_hyperedge_index
    from dh2a_kt.models.dh2_kt import DH2KTBatch, empty_hyperedge_index
    from dh2a_kt.train.tier1 import graph_sensitivity_loss, precompute_concept_states

    device = trained.device
    optimizer = torch.optim.Adam(model.black_box.parameters(), lr=budget.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and str(device).startswith("cuda"))
    model.train()
    full_index = {k: v.to(device) for k, v in trained.clean_hyperedge_index.items()}
    if freeze_hypergraph:
        logger.info("GreyKT freeze-hypergraph: concept graph encoded once per epoch (no graph backward)")
    if use_amp:
        logger.info("GreyKT AMP enabled")
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    start_epoch = 0  # 0-based index into range(budget.epochs)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(resume_from) if resume_from is not None else None
    if resume_path is None and ckpt_dir is not None:
        candidate = ckpt_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate
    if resume_path is not None:
        payload = load_greykt_train_checkpoint(
            resume_path, model, optimizer, device=str(device)
        )
        completed = int(payload.get("epoch", 0))
        start_epoch = completed  # next epoch index = completed count
        best_epoch = int(payload.get("best_epoch", -1))
        best_loss = float(payload.get("best_loss", float("inf")))
        best_state = payload.get("best_state")
        if start_epoch >= budget.epochs:
            logger.info(
                "resume checkpoint already finished all %s epochs; restoring best",
                budget.epochs,
            )
            if best_state is not None:
                model.load_state_dict(best_state)
            return

    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    for epoch in range(start_epoch, budget.epochs):
        empty_states = precompute_concept_states(
            model.black_box,
            empty_hyperedge_index(
                device, kinds=tuple(model.black_box.config.hyperedge_kinds)
            ),
            device,
            enable_grad=False,
        )
        cached_full_states = None
        if freeze_hypergraph:
            cached_full_states = precompute_concept_states(
                model.black_box, full_index, device, enable_grad=False
            )
        ablated_index = None
        if graph_sensitivity_weight > 0.0 and trained.clean_hyperedges and not freeze_hypergraph:
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
            if freeze_hypergraph:
                states = empty_states if use_empty else cached_full_states
            else:
                states = empty_states if use_empty else precompute_concept_states(
                    model.black_box, full_index, device, enable_grad=True
                )
            cb_t = _maybe_mc_confidence(model, batch, trained, mc_samples)
            grey_batch = _to_grey_batch(
                batch,
                trained,
                prereq_edge_index,
                concept_train_freq,
                states,
                black_confidence=cb_t,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_amp):
                out = model(grey_batch)
            # BCE on probabilities is not AMP-safe; compute in fp32.
            loss = next_step_bce(out.probs.float(), batch["correct"].to(device), lengths)
            if graph_sensitivity_weight > 0.0 and ablated_index is not None and n_batches % 5 == 0:
                with torch.amp.autocast(amp_device, enabled=use_amp):
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
                    logits_full.float(), logits_ablated.float(), lengths, margin=graph_sensitivity_margin
                )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.black_box.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        completed_epoch = epoch + 1
        logger.info("greykt epoch=%s/%s loss=%.4f", completed_epoch, budget.epochs, mean_loss)
        if n_batches > 0 and mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = completed_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ckpt_dir is not None:
            meta = dict(checkpoint_meta or {})
            meta.update({"completed_epoch": completed_epoch, "train_loss": mean_loss})
            save_greykt_train_checkpoint(
                ckpt_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=completed_epoch,
                best_epoch=best_epoch,
                best_loss=best_loss,
                best_state=best_state,
                meta=meta,
            )
            if best_epoch == completed_epoch and best_state is not None:
                save_greykt_train_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=completed_epoch,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    best_state=best_state,
                    meta=meta,
                )
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("greykt restored best epoch=%s train_loss=%.4f", best_epoch, best_loss)


def write_greykt_report(report: GreyKTEvalReport, path: Path) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path
