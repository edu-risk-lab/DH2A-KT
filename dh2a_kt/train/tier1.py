"""Tier 1 training loop for DH2-KT (M5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from dh2a_kt.baselines.loader import compare_against_p0, load_p0_baseline_summary
    from dh2a_kt.hyperedge.construction import Hyperedge, resolve_concept_prerequisite_hyperedges
    from dh2a_kt.hyperedge.indexing import destroyed_hyperedge_index, hyperedge_index_from_list
    from dh2a_kt.models.causal_integration import build_entity_maps
    from dh2a_kt.models.dh2_kt import DH2KT, DH2KTBatch, DH2KTConfig, empty_hyperedge_index

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


@dataclass
class TrainingBudget:
    reference_model: str
    batch_size: int
    epochs: int
    lr: float
    max_seq_len: int
    matched_p0: bool = True
    note: str = ""


@dataclass
class ConceptPrerequisiteSpec:
    fold: int = 0
    source: str = "chain"
    min_chain_len: int = 3
    max_chain_len: int = 8


@dataclass
class FoldResult:
    fold: int
    auc: float
    n_predictions: int


if _TORCH_AVAILABLE:

    @dataclass
    class TrainedFold:
        model: DH2KT
        eval_loader: DataLoader
        kc_to_idx: dict[int, int]
        item_to_idx: dict[int, int]
        clean_hyperedge_index: dict[str, torch.Tensor]
        clean_hyperedges: list[Hyperedge]
        device: torch.device


def resolve_training_budget(
    p0_cfg: dict,
    *,
    reference_model: str = "gkt",
    lr: float = 0.001,
    max_seq_len: int | None = None,
) -> TrainingBudget:
    batch_size = 4
    epochs = 10
    matched = False
    for entry in p0_cfg.get("baselines", []):
        if entry.get("name") == reference_model:
            hp = entry.get("hyperparams", {})
            batch_size = int(hp.get("batch_size", batch_size))
            epochs = int(hp.get("epochs", epochs))
            matched = True
            break
    if max_seq_len is None:
        max_seq_len = int(p0_cfg.get("pykt", {}).get("max_seq_len", 200))
    note = "" if matched else f"budget reference {reference_model!r} not found in P0 config"
    return TrainingBudget(
        reference_model=reference_model,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        max_seq_len=max_seq_len,
        matched_p0=matched,
        note=note,
    )


def shift_responses_for_next_step(correct: torch.Tensor) -> torch.Tensor:
    responses = torch.zeros_like(correct)
    if correct.size(1) > 1:
        responses[:, 1:] = correct[:, :-1]
    return responses


def next_step_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """``logits[:, t]`` predicts ``targets[:, t + 1]`` for valid positions."""
    if logits.size(1) < 2:
        return logits.sum() * 0.0
    pred = logits[:, :-1, 0]
    label = targets[:, 1:]
    mask = torch.arange(pred.size(1), device=pred.device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
    if not mask.any():
        return pred.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(pred[mask], label[mask])


def graph_sensitivity_loss(
    logits_full: torch.Tensor,
    logits_ablated: torch.Tensor,
    lengths: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Penalize identical predictions under full vs ablated hypergraph."""
    if logits_full.size(1) < 2:
        return logits_full.sum() * 0.0
    pred_full = logits_full[:, :-1, 0]
    pred_ablated = logits_ablated[:, :-1, 0]
    mask = torch.arange(pred_full.size(1), device=pred_full.device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
    if not mask.any():
        return pred_full.sum() * 0.0
    diff = (pred_full[mask] - pred_ablated[mask]).abs().mean()
    return torch.relu(torch.tensor(margin, device=diff.device) - diff)


if _TORCH_AVAILABLE:

    class UserSequenceDataset(Dataset):
        def __init__(
            self,
            df: pd.DataFrame,
            kc_to_idx: dict[int, int],
            item_to_idx: dict[int, int],
            *,
            max_seq_len: int,
        ):
            self.max_seq_len = max_seq_len
            sorted_df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
            kc_col = sorted_df["kc_id"].map(kc_to_idx)
            item_col = sorted_df["item_id"].map(item_to_idx)
            if kc_col.isna().any() or item_col.isna().any():
                missing = int(kc_col.isna().sum() + item_col.isna().sum())
                raise ValueError(f"UserSequenceDataset: {missing} rows have unmapped kc_id/item_id")
            self._kc_ids = kc_col.to_numpy(dtype=np.int64)
            self._item_ids = item_col.to_numpy(dtype=np.int64)
            self._correct = sorted_df["correct"].to_numpy(dtype=np.float32)
            user_ids = sorted_df["user_id"].to_numpy()
            if len(user_ids) == 0:
                self._starts = []
                self._ends = []
            else:
                boundaries = np.where(user_ids[1:] != user_ids[:-1])[0] + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(user_ids)]))
                valid = (ends - starts) >= 2
                self._starts = starts[valid].tolist()
                self._ends = ends[valid].tolist()
            logger.info(
                "UserSequenceDataset: %d users (from %d interactions)",
                len(self._starts),
                len(sorted_df),
            )

        def __len__(self) -> int:
            return len(self._starts)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            start = self._starts[idx]
            end = min(self._ends[idx], start + self.max_seq_len)
            length = end - start
            concept_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            exercise_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            correct = torch.zeros(self.max_seq_len, dtype=torch.float)
            concept_ids[:length] = torch.from_numpy(self._kc_ids[start:end])
            exercise_ids[:length] = torch.from_numpy(self._item_ids[start:end])
            correct[:length] = torch.from_numpy(self._correct[start:end])
            responses = shift_responses_for_next_step(correct.unsqueeze(0)).squeeze(0)
            return {
                "concept_ids": concept_ids,
                "exercise_ids": exercise_ids,
                "responses": responses,
                "correct": correct,
                "length": torch.tensor(length, dtype=torch.long),
            }

    def _collate(batch_items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "concept_ids": torch.stack([item["concept_ids"] for item in batch_items]),
            "exercise_ids": torch.stack([item["exercise_ids"] for item in batch_items]),
            "responses": torch.stack([item["responses"] for item in batch_items]),
            "correct": torch.stack([item["correct"] for item in batch_items]),
            "lengths": torch.stack([item["length"] for item in batch_items]),
        }

    @torch.no_grad()
    def collect_predictions(
        model: DH2KT,
        loader: DataLoader,
        hyperedge_index: dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs: list[float] = []
        labels: list[float] = []
        concept_states = precompute_concept_states(model, hyperedge_index, device)
        for batch in loader:
            lengths = batch["lengths"].to(device)
            dh2_batch = DH2KTBatch(
                concept_ids=batch["concept_ids"].to(device),
                exercise_ids=batch["exercise_ids"].to(device),
                responses=batch["responses"].to(device),
                hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
                concept_states=concept_states,
            )
            logits = model(dh2_batch)
            pred = logits[:, :-1, 0]
            target = batch["correct"].to(device)[:, 1:]
            mask = torch.arange(pred.size(1), device=device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
            probs.extend(torch.sigmoid(pred[mask]).cpu().tolist())
            labels.extend(target[mask].cpu().tolist())
        return np.asarray(probs), np.asarray(labels)

    def evaluate_auc(
        model: DH2KT,
        loader: DataLoader,
        hyperedge_index: dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple[float, int]:
        from sklearn.metrics import roc_auc_score

        probs, labels = collect_predictions(model, loader, hyperedge_index, device)
        if len(probs) == 0 or len(np.unique(labels)) < 2:
            return float("nan"), 0
        return float(roc_auc_score(labels, probs)), len(probs)

    def precompute_concept_states(
        model: DH2KT,
        hyperedge_index: dict[str, torch.Tensor],
        device: torch.device,
        *,
        enable_grad: bool = False,
    ) -> torch.Tensor:
        index = {k: v.to(device) for k, v in hyperedge_index.items()}
        if enable_grad:
            return model._encode_concepts(index)
        with torch.no_grad():
            return model._encode_concepts(index)

    def train_one_fold(
        model: DH2KT,
        train_loader: DataLoader,
        hyperedge_index: dict[str, torch.Tensor],
        budget: TrainingBudget,
        device: torch.device,
        *,
        clean_hyperedges: list[Hyperedge],
        kc_to_idx: dict[int, int] | None = None,
        graph_dropout: float = 0.0,
        graph_sensitivity_weight: float = 0.0,
        graph_sensitivity_margin: float = 0.05,
        graph_sensitivity_p: float = 0.9,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=budget.lr)
        model.train()
        full_hyperedge_index = {k: v.to(device) for k, v in hyperedge_index.items()}
        p0_log = logging.getLogger("src.dag_disruption")
        prev_p0_level = p0_log.level
        p0_log.setLevel(logging.WARNING)
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch_loss = float("inf")
        best_epoch = -1
        try:
            for epoch in range(budget.epochs):
                empty_states = precompute_concept_states(
                    model, empty_hyperedge_index(device), device, enable_grad=False
                )
                ablated_index = None
                if graph_sensitivity_weight > 0.0 and clean_hyperedges:
                    ablation_seed = int(torch.randint(0, 2**31, (1,)).item())
                    ablated_index = destroyed_hyperedge_index(
                        clean_hyperedges,
                        kc_to_idx or {},
                        p=graph_sensitivity_p,
                        seed=ablation_seed,
                        device=device,
                    )

                epoch_loss = 0.0
                n_batches = 0
                n_skipped = 0
                for batch in train_loader:
                    lengths = batch["lengths"].to(device)
                    use_empty = graph_dropout > 0.0 and torch.rand(1).item() < graph_dropout
                    if use_empty:
                        states = empty_states
                    else:
                        states = precompute_concept_states(
                            model, full_hyperedge_index, device, enable_grad=True
                        )
                    dh2_batch = DH2KTBatch(
                        concept_ids=batch["concept_ids"].to(device),
                        exercise_ids=batch["exercise_ids"].to(device),
                        responses=batch["responses"].to(device),
                        hyperedge_index=full_hyperedge_index,
                        concept_states=states,
                    )
                    optimizer.zero_grad()
                    logits = model(dh2_batch)
                    loss = next_step_loss(logits, batch["correct"].to(device), lengths)
                    if (
                        graph_sensitivity_weight > 0.0
                        and ablated_index is not None
                        and n_batches % 5 == 0
                    ):
                        full_batch = DH2KTBatch(
                            concept_ids=dh2_batch.concept_ids,
                            exercise_ids=dh2_batch.exercise_ids,
                            responses=dh2_batch.responses,
                            hyperedge_index=full_hyperedge_index,
                            concept_states=states if not use_empty else precompute_concept_states(
                                model, full_hyperedge_index, device, enable_grad=True
                            ),
                        )
                        ablated_batch = DH2KTBatch(
                            concept_ids=dh2_batch.concept_ids,
                            exercise_ids=dh2_batch.exercise_ids,
                            responses=dh2_batch.responses,
                            hyperedge_index=ablated_index,
                        )
                        logits_full = model(full_batch)
                        logits_ablated = model(ablated_batch)
                        loss = loss + graph_sensitivity_weight * graph_sensitivity_loss(
                            logits_full,
                            logits_ablated,
                            lengths,
                            margin=graph_sensitivity_margin,
                        )
                    if not torch.isfinite(loss):
                        n_skipped += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                    if n_batches % 500 == 0:
                        logger.info(
                            "epoch=%s/%s batch=%s/%s loss=%.4f",
                            epoch + 1,
                            budget.epochs,
                            n_batches,
                            len(train_loader),
                            epoch_loss / n_batches,
                        )
                mean_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    "epoch=%s/%s loss=%.4f batches=%s skipped_nonfinite=%s",
                    epoch + 1,
                    budget.epochs,
                    mean_loss,
                    n_batches,
                    n_skipped,
                )
                # Keep the best epoch: fold-1 previously collapsed on the final
                # epoch (loss 0.46 -> 0.76) while earlier epochs were healthy.
                if n_batches > 0 and mean_loss < best_epoch_loss:
                    best_epoch_loss = mean_loss
                    best_epoch = epoch + 1
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
            if best_state is not None:
                model.load_state_dict(best_state)
                logger.info(
                    "restored best epoch=%s train_loss=%.4f (of %s)",
                    best_epoch,
                    best_epoch_loss,
                    budget.epochs,
                )
        finally:
            p0_log.setLevel(prev_p0_level)

    def build_model(
        n_concepts: int,
        n_exercises: int,
        *,
        hidden_dim: int = 128,
        n_hypergraph_layers: int = 2,
        dropout: float = 0.2,
        hyperedge_kinds: tuple[str, ...] = ("concept_prerequisite",),
    ) -> DH2KT:
        config = DH2KTConfig(
            n_concepts=n_concepts,
            n_exercises=n_exercises,
            hidden_dim=hidden_dim,
            embed_dim=hidden_dim,
            n_hypergraph_layers=n_hypergraph_layers,
            dropout=dropout,
            hyperedge_kinds=hyperedge_kinds,
        )
        return DH2KT(config)


def train_fold(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    e_pre: pd.DataFrame,
    budget: TrainingBudget,
    *,
    device: str | torch.device = "cpu",
    hidden_dim: int = 128,
    n_hypergraph_layers: int = 2,
    graph_dropout: float = 0.0,
    graph_sensitivity_weight: float = 0.0,
    graph_sensitivity_margin: float = 0.05,
    graph_sensitivity_p: float = 0.9,
    hyperedge_spec: ConceptPrerequisiteSpec | None = None,
    session_hyperedges: list[Hyperedge] | None = None,
    max_users: int | None = None,
) -> TrainedFold:
    if not _TORCH_AVAILABLE:
        raise ImportError("train_fold requires PyTorch")

    from dh2a_kt.hyperedge.indexing import select_session_hyperedges_for_training

    device = torch.device(device)
    if max_users is not None:
        train_users = train_df["user_id"].unique()[:max_users]
        train_df = train_df[train_df["user_id"].isin(train_users)]
        eval_users = eval_df["user_id"].unique()[:max_users]
        eval_df = eval_df[eval_df["user_id"].isin(eval_users)]

    spec = hyperedge_spec or ConceptPrerequisiteSpec()
    kc_to_idx, item_to_idx = build_entity_maps(
        pd.concat([train_df, eval_df], ignore_index=True),
        e_pre,
    )
    logger.info(
        "Building concept-prerequisite hyperedges source=%s fold=%s",
        spec.source,
        spec.fold,
    )
    clean_hyperedges = resolve_concept_prerequisite_hyperedges(
        e_pre,
        fold=spec.fold,
        source=spec.source,
        min_chain_len=spec.min_chain_len,
        max_chain_len=spec.max_chain_len,
    )
    kinds: list[str] = ["concept_prerequisite"]
    if session_hyperedges:
        selected = select_session_hyperedges_for_training(session_hyperedges)
        clean_hyperedges = list(clean_hyperedges) + list(selected)
        kinds.append("session")
        logger.info(
            "Added %d session hyperedges (concept projection) for training",
            len(selected),
        )
    hyperedge_index = hyperedge_index_from_list(clean_hyperedges, kc_to_idx)
    model = build_model(
        len(kc_to_idx),
        len(item_to_idx),
        hidden_dim=hidden_dim,
        n_hypergraph_layers=n_hypergraph_layers,
        hyperedge_kinds=tuple(kinds),
    ).to(device)
    logger.info(
        "train_fold: concepts=%d exercises=%d hyperedges=%d kinds=%s device=%s",
        len(kc_to_idx),
        len(item_to_idx),
        len(clean_hyperedges),
        kinds,
        device,
    )

    train_loader = DataLoader(
        UserSequenceDataset(train_df, kc_to_idx, item_to_idx, max_seq_len=budget.max_seq_len),
        batch_size=budget.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    eval_loader = DataLoader(
        UserSequenceDataset(eval_df, kc_to_idx, item_to_idx, max_seq_len=budget.max_seq_len),
        batch_size=budget.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    train_one_fold(
        model,
        train_loader,
        hyperedge_index,
        budget,
        device,
        clean_hyperedges=clean_hyperedges,
        kc_to_idx=kc_to_idx,
        graph_dropout=graph_dropout,
        graph_sensitivity_weight=graph_sensitivity_weight,
        graph_sensitivity_margin=graph_sensitivity_margin,
        graph_sensitivity_p=graph_sensitivity_p,
    )
    return TrainedFold(
        model=model,
        eval_loader=eval_loader,
        kc_to_idx=kc_to_idx,
        item_to_idx=item_to_idx,
        clean_hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
        clean_hyperedges=clean_hyperedges,
        device=device,
    )


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    e_pre: pd.DataFrame,
    budget: TrainingBudget,
    *,
    device: str | torch.device = "cpu",
    hidden_dim: int = 128,
    n_hypergraph_layers: int = 2,
    graph_dropout: float = 0.0,
    graph_sensitivity_weight: float = 0.0,
    graph_sensitivity_margin: float = 0.05,
    graph_sensitivity_p: float = 0.9,
    hyperedge_spec: ConceptPrerequisiteSpec | None = None,
    session_hyperedges: list[Hyperedge] | None = None,
    max_users: int | None = None,
) -> FoldResult:
    if not _TORCH_AVAILABLE:
        raise ImportError("train_and_evaluate_fold requires PyTorch")

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
        hyperedge_spec=hyperedge_spec,
        session_hyperedges=session_hyperedges,
        max_users=max_users,
    )
    auc, n_predictions = evaluate_auc(
        trained.model,
        trained.eval_loader,
        trained.clean_hyperedge_index,
        trained.device,
    )
    return FoldResult(fold=-1, auc=auc, n_predictions=n_predictions)


def write_comparison_table(
    results: list[FoldResult],
    *,
    dataset: str,
    budget: TrainingBudget,
    comparison_models: list[str],
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    dh2_aucs = [r.auc for r in results if np.isfinite(r.auc)]
    dh2_mean = float(np.mean(dh2_aucs)) if dh2_aucs else float("nan")
    comparison_type = "matched" if budget.matched_p0 else "observational"

    if not comparison_models:
        # Secondary corpora (e.g. FoundationalASSIST) have no P0-reused baselines.
        for fold_result in results:
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold_result.fold,
                    "reference_model": "none",
                    "p0_auc": float("nan"),
                    "dh2_kt_auc": fold_result.auc,
                    "delta_auc": float("nan"),
                    "budget_reference": budget.reference_model,
                    "batch_size": budget.batch_size,
                    "epochs": budget.epochs,
                    "comparison_type": comparison_type,
                    "budget_note": budget.note or "no_p0_baseline",
                    "n_predictions": fold_result.n_predictions,
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "fold": "mean",
                "reference_model": "none",
                "p0_auc": float("nan"),
                "dh2_kt_auc": dh2_mean,
                "delta_auc": float("nan"),
                "budget_reference": budget.reference_model,
                "batch_size": budget.batch_size,
                "epochs": budget.epochs,
                "comparison_type": comparison_type,
                "budget_note": budget.note or "no_p0_baseline",
                "n_predictions": sum(r.n_predictions for r in results),
            }
        )
    else:
        for fold_result in results:
            for model in comparison_models:
                p0_row = load_p0_baseline_summary(dataset, model=model, graph_construction="train_only")
                p0_auc = float(p0_row.iloc[0]["auc"]) if not p0_row.empty else float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold_result.fold,
                        "reference_model": model,
                        "p0_auc": p0_auc,
                        "dh2_kt_auc": fold_result.auc,
                        "delta_auc": fold_result.auc - p0_auc if np.isfinite(fold_result.auc) else float("nan"),
                        "budget_reference": budget.reference_model,
                        "batch_size": budget.batch_size,
                        "epochs": budget.epochs,
                        "comparison_type": comparison_type,
                        "budget_note": budget.note,
                        "n_predictions": fold_result.n_predictions,
                    }
                )

        for model in comparison_models:
            cmp = compare_against_p0(dh2_mean, dataset, model=model)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": "mean",
                    "reference_model": model,
                    "p0_auc": cmp["p0_auc"],
                    "dh2_kt_auc": dh2_mean,
                    "delta_auc": cmp["delta_auc"],
                    "budget_reference": budget.reference_model,
                    "batch_size": budget.batch_size,
                    "epochs": budget.epochs,
                    "comparison_type": comparison_type,
                    "budget_note": budget.note,
                    "n_predictions": sum(r.n_predictions for r in results),
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
