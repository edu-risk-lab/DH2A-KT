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
    val_auc: float | None = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs so a run can be reproduced by seed."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


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
        val_auc: float | None = None


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


def responses_for_model(model: "DH2KT", batch: dict) -> torch.Tensor:
    """Pick the response channel matching the architecture's alignment.

    v2/v3 read the response shifted by one step, which leaves them predicting
    ``t+1`` without ever seeing the outcome of ``t``. v4/v5 read the response of
    the same step (standard DKT alignment), so no outcome is wasted.
    """
    if getattr(model.config, "architecture", "v2") in ("v4", "v5"):
        return batch["correct"]
    return batch["responses"]


def aux_kwargs(batch: dict, device: torch.device | str) -> dict:
    """Forward optional v4 sidecar tensors (time gap / saw / group) when present."""
    out: dict = {}
    if "time_gaps" in batch:
        out["time_gaps"] = batch["time_gaps"].to(device)
    if "saw_flags" in batch:
        out["saw_flags"] = batch["saw_flags"].to(device)
    if "group_ids" in batch:
        out["group_ids"] = batch["group_ids"].to(device)
    if "durations" in batch:
        out["durations"] = batch["durations"].to(device)
    if "idles" in batch:
        out["idles"] = batch["idles"].to(device)
    return out


def kc_set_kwargs(batch: dict, device: torch.device | str) -> dict:
    """Forward the v5 KC-set tensors when the loader produced them."""
    if "kc_set_ids" not in batch:
        return {}
    return {
        "kc_set_ids": batch["kc_set_ids"].to(device),
        "kc_set_mask": batch["kc_set_mask"].to(device),
    }


def shift_responses_for_next_step(correct: torch.Tensor) -> torch.Tensor:
    responses = torch.zeros_like(correct)
    if correct.size(1) > 1:
        responses[:, 1:] = correct[:, :-1]
    return responses


def next_step_mask(
    n_positions: int,
    lengths: torch.Tensor,
    *,
    is_repeat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Which positions ``t`` (predicting ``t + 1``) count towards loss and AUC.

    Single source of truth so the training loss, validation AUC and test AUC can
    never disagree about which positions are scored.

    ``is_repeat`` is the per-row flag of shape ``(B, T)``; passing it drops every
    target that merely continues the previous row's attempt. pyKT's KC-level
    export splits one multi-concept question into consecutive rows sharing the
    answer, so such a target is already visible in the model's input and is
    predictable without any knowledge tracing.
    """
    mask = torch.arange(n_positions, device=lengths.device).unsqueeze(0) < (
        lengths.unsqueeze(1) - 1
    )
    if is_repeat is not None:
        mask = mask & ~is_repeat[:, 1 : n_positions + 1].bool()
    return mask


def next_step_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    *,
    is_repeat: torch.Tensor | None = None,
) -> torch.Tensor:
    """``logits[:, t]`` predicts ``targets[:, t + 1]`` for valid positions."""
    if logits.size(1) < 2:
        return logits.sum() * 0.0
    pred = logits[:, :-1, 0]
    label = targets[:, 1:]
    mask = next_step_mask(pred.size(1), lengths, is_repeat=is_repeat)
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


def _repeat_flags(sorted_df: pd.DataFrame) -> np.ndarray:
    """Rows that continue the previous row's attempt on the same question.

    pyKT's KC-level export emits one row per concept of a multi-concept question,
    all sharing user, item, timestamp and answer. Such a row is not a new
    opportunity to predict: its label is the answer already fed to the model.
    """
    n = len(sorted_df)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags
    user_ids = sorted_df["user_id"].to_numpy()
    item_ids = sorted_df["item_id"].to_numpy()
    timestamps = sorted_df["timestamp"].to_numpy()
    flags[1:] = (
        (user_ids[1:] == user_ids[:-1])
        & (item_ids[1:] == item_ids[:-1])
        & (timestamps[1:] == timestamps[:-1])
    )
    return flags


if _TORCH_AVAILABLE:

    class UserSequenceDataset(Dataset):
        def __init__(
            self,
            df: pd.DataFrame,
            kc_to_idx: dict[int, int],
            item_to_idx: dict[int, int],
            *,
            max_seq_len: int,
            window_mode: str = "first",
            collapse_events: bool = False,
            max_kcs: int = 6,
            include_time_gaps: bool = False,
            include_saw: bool = False,
            include_group: bool = False,
            include_time_split: bool = False,
        ):
            if window_mode not in ("first", "last", "chunked"):
                raise ValueError(
                    f"window_mode must be 'first', 'last' or 'chunked', got {window_mode!r}"
                )
            self.max_seq_len = max_seq_len
            self.window_mode = window_mode
            self.collapse_events = collapse_events
            self.max_kcs = max_kcs
            self.include_time_gaps = include_time_gaps
            self.include_saw = include_saw
            self.include_group = include_group
            self.include_time_split = include_time_split
            sorted_df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
            self._kc_set_ids: np.ndarray | None = None
            self._kc_set_mask: np.ndarray | None = None
            if collapse_events:
                # v5 models one question attempt per position, carrying its KC set,
                # so the duplicated KC rows disappear instead of being masked.
                from dh2a_kt.data.events import collapse_to_events, pad_kc_matrix

                events = collapse_to_events(sorted_df, max_kcs=max_kcs)
                mapped = [
                    [kc_to_idx[kc] for kc in kcs if kc in kc_to_idx]
                    for kcs in events["kc_ids"]
                ]
                self._kc_set_ids, self._kc_set_mask = pad_kc_matrix(mapped, width=max_kcs)
                sorted_df = events
            kc_col = sorted_df["kc_id"].map(kc_to_idx)
            item_col = sorted_df["item_id"].map(item_to_idx)
            if kc_col.isna().any() or item_col.isna().any():
                missing = int(kc_col.isna().sum() + item_col.isna().sum())
                raise ValueError(f"UserSequenceDataset: {missing} rows have unmapped kc_id/item_id")
            self._kc_ids = kc_col.to_numpy(dtype=np.int64)
            self._item_ids = item_col.to_numpy(dtype=np.int64)
            self._correct = sorted_df["correct"].to_numpy(dtype=np.float32)
            user_ids = sorted_df["user_id"].to_numpy()
            self._is_repeat = _repeat_flags(sorted_df)
            timestamps = (
                sorted_df["timestamp"].to_numpy(dtype=np.int64)
                if "timestamp" in sorted_df.columns
                else np.zeros(len(sorted_df), dtype=np.int64)
            )
            if include_time_gaps:
                from dh2a_kt.data.aux_signals import log_time_gaps

                self._time_gaps = log_time_gaps(timestamps, user_ids)
            else:
                self._time_gaps = None
            if include_saw:
                if "saw_answer" in sorted_df.columns:
                    self._saw = sorted_df["saw_answer"].fillna(0).to_numpy(dtype=np.int64)
                else:
                    self._saw = np.zeros(len(sorted_df), dtype=np.int64)
            else:
                self._saw = None
            if include_group:
                if "group_id" in sorted_df.columns:
                    self._group = sorted_df["group_id"].fillna(0).to_numpy(dtype=np.int64)
                else:
                    self._group = np.zeros(len(sorted_df), dtype=np.int64)
            else:
                self._group = None
            if include_time_split:
                from dh2a_kt.data.aux_signals import log_duration_idle

                if "duration_sec" in sorted_df.columns:
                    duration_sec = (
                        sorted_df["duration_sec"].fillna(0).to_numpy(dtype=np.float32)
                    )
                else:
                    duration_sec = np.zeros(len(sorted_df), dtype=np.float32)
                self._durations, self._idles = log_duration_idle(
                    timestamps, user_ids, duration_sec
                )
            else:
                self._durations = None
                self._idles = None
            self._n_users = 0
            self._window_users: list[int] = []
            if len(user_ids) == 0:
                self._starts = []
                self._ends = []
            else:
                boundaries = np.where(user_ids[1:] != user_ids[:-1])[0] + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(user_ids)]))
                valid = (ends - starts) >= 2
                starts = starts[valid]
                ends = ends[valid]
                if window_mode == "chunked":
                    # Tile each learner so every interaction is used. "first" and
                    # "last" keep a single max_seq_len window, and on XES3G5M every
                    # learner is longer than that, so they discard most of the log.
                    window_starts: list[int] = []
                    window_ends: list[int] = []
                    for user_start, user_end in zip(starts.tolist(), ends.tolist(), strict=True):
                        for offset in range(user_start, user_end, max_seq_len):
                            window_end = min(offset + max_seq_len, user_end)
                            if window_end - offset >= 2:
                                window_starts.append(offset)
                                window_ends.append(window_end)
                    self._starts = window_starts
                    self._ends = window_ends
                elif window_mode == "last":
                    # P0's pyKT export slices ``q_list[-max_seq_len:]``, so scoring a
                    # DH2-KT checkpoint against its published table needs the tail,
                    # not the head: same position count, different rows.
                    self._starts = np.maximum(starts, ends - max_seq_len).tolist()
                    self._ends = ends.tolist()
                else:
                    self._starts = starts.tolist()
                    self._ends = ends.tolist()
                self._n_users = int(len(starts))
                # Learner id per window, so predictions can be grouped by learner
                # for a bootstrap that resamples learners rather than positions.
                self._window_users = [int(user_ids[s]) for s in self._starts]
            logger.info(
                "UserSequenceDataset: %d windows (mode=%s) from %d interactions",
                len(self._starts),
                window_mode,
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
            is_repeat = torch.zeros(self.max_seq_len, dtype=torch.bool)
            concept_ids[:length] = torch.from_numpy(self._kc_ids[start:end])
            exercise_ids[:length] = torch.from_numpy(self._item_ids[start:end])
            correct[:length] = torch.from_numpy(self._correct[start:end])
            is_repeat[:length] = torch.from_numpy(self._is_repeat[start:end])
            responses = shift_responses_for_next_step(correct.unsqueeze(0)).squeeze(0)
            item = {
                "concept_ids": concept_ids,
                "exercise_ids": exercise_ids,
                "responses": responses,
                "correct": correct,
                "is_repeat": is_repeat,
                "length": torch.tensor(length, dtype=torch.long),
                "user_id": torch.tensor(self._window_users[idx], dtype=torch.long),
            }
            if self._time_gaps is not None:
                time_gaps = torch.zeros(self.max_seq_len, dtype=torch.float)
                time_gaps[:length] = torch.from_numpy(self._time_gaps[start:end])
                item["time_gaps"] = time_gaps
            if self._saw is not None:
                saw_flags = torch.zeros(self.max_seq_len, dtype=torch.long)
                saw_flags[:length] = torch.from_numpy(self._saw[start:end])
                item["saw_flags"] = saw_flags
            if self._group is not None:
                group_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
                group_ids[:length] = torch.from_numpy(self._group[start:end])
                item["group_ids"] = group_ids
            if self._durations is not None:
                durations = torch.zeros(self.max_seq_len, dtype=torch.float)
                durations[:length] = torch.from_numpy(self._durations[start:end])
                item["durations"] = durations
            if self._idles is not None:
                idles = torch.zeros(self.max_seq_len, dtype=torch.float)
                idles[:length] = torch.from_numpy(self._idles[start:end])
                item["idles"] = idles
            if self._kc_set_ids is not None and self._kc_set_mask is not None:
                kc_set_ids = torch.zeros(self.max_seq_len, self.max_kcs, dtype=torch.long)
                kc_set_mask = torch.zeros(self.max_seq_len, self.max_kcs, dtype=torch.bool)
                kc_set_ids[:length] = torch.from_numpy(self._kc_set_ids[start:end])
                kc_set_mask[:length] = torch.from_numpy(self._kc_set_mask[start:end])
                item["kc_set_ids"] = kc_set_ids
                item["kc_set_mask"] = kc_set_mask
            return item

    def _collate(batch_items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        collated = {
            "concept_ids": torch.stack([item["concept_ids"] for item in batch_items]),
            "exercise_ids": torch.stack([item["exercise_ids"] for item in batch_items]),
            "responses": torch.stack([item["responses"] for item in batch_items]),
            "correct": torch.stack([item["correct"] for item in batch_items]),
            "is_repeat": torch.stack([item["is_repeat"] for item in batch_items]),
            "lengths": torch.stack([item["length"] for item in batch_items]),
            "user_ids": torch.stack([item["user_id"] for item in batch_items]),
        }
        if "kc_set_ids" in batch_items[0]:
            collated["kc_set_ids"] = torch.stack(
                [item["kc_set_ids"] for item in batch_items]
            )
            collated["kc_set_mask"] = torch.stack(
                [item["kc_set_mask"] for item in batch_items]
            )
        for aux_key in ("time_gaps", "saw_flags", "group_ids", "durations", "idles"):
            if aux_key in batch_items[0]:
                collated[aux_key] = torch.stack([item[aux_key] for item in batch_items])
        return collated

    @torch.no_grad()
    def collect_predictions(
        model: DH2KT,
        loader: DataLoader,
        hyperedge_index: dict[str, torch.Tensor],
        device: torch.device,
        *,
        mask_repeats: bool = False,
        return_users: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        probs: list[float] = []
        labels: list[float] = []
        users: list[int] = []
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
                **aux_kwargs(batch, device),
            )
            logits = model(dh2_batch)
            pred = logits[:, :-1, 0]
            target = batch["correct"].to(device)[:, 1:]
            mask = next_step_mask(
                pred.size(1),
                lengths,
                is_repeat=batch["is_repeat"].to(device) if mask_repeats else None,
            )
            probs.extend(torch.sigmoid(pred[mask]).cpu().tolist())
            labels.extend(target[mask].cpu().tolist())
            if return_users:
                per_row = batch["user_ids"].to(device).unsqueeze(1).expand_as(mask)
                users.extend(per_row[mask].cpu().tolist())
        if return_users:
            return np.asarray(probs), np.asarray(labels), np.asarray(users)
        return np.asarray(probs), np.asarray(labels)

    def evaluate_auc(
        model: DH2KT,
        loader: DataLoader,
        hyperedge_index: dict[str, torch.Tensor],
        device: torch.device,
        *,
        mask_repeats: bool = False,
    ) -> tuple[float, int]:
        from sklearn.metrics import roc_auc_score

        probs, labels = collect_predictions(
            model, loader, hyperedge_index, device, mask_repeats=mask_repeats
        )
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
        val_loader: DataLoader | None = None,
        early_stop_patience: int | None = None,
        mask_repeats: bool = False,
    ) -> float | None:
        if mask_repeats:
            logger.info(
                "clean protocol: repeat-row targets excluded from the loss and from val AUC"
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=budget.lr)
        model.train()
        full_hyperedge_index = {k: v.to(device) for k, v in hyperedge_index.items()}
        p0_log = logging.getLogger("src.dag_disruption")
        prev_p0_level = p0_log.level
        p0_log.setLevel(logging.WARNING)
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch_loss = float("inf")
        best_val_auc = -float("inf")
        best_epoch = -1
        epochs_without_gain = 0
        try:
            for epoch in range(budget.epochs):
                empty_states = precompute_concept_states(
                    model,
                    empty_hyperedge_index(device, kinds=tuple(model.config.hyperedge_kinds)),
                    device,
                    enable_grad=False,
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
                empty_index = empty_hyperedge_index(
                    device, kinds=tuple(model.config.hyperedge_kinds)
                )
                for batch in train_loader:
                    lengths = batch["lengths"].to(device)
                    use_empty = graph_dropout > 0.0 and torch.rand(1).item() < graph_dropout
                    if use_empty:
                        states = empty_states
                        index_for_batch = empty_index
                    else:
                        states = precompute_concept_states(
                            model, full_hyperedge_index, device, enable_grad=True
                        )
                        index_for_batch = full_hyperedge_index
                    kc_sets = kc_set_kwargs(batch, device)
                    aux = aux_kwargs(batch, device)
                    dh2_batch = DH2KTBatch(
                        concept_ids=batch["concept_ids"].to(device),
                        exercise_ids=batch["exercise_ids"].to(device),
                        responses=responses_for_model(model, batch).to(device),
                        hyperedge_index=index_for_batch,
                        concept_states=states,
                        lengths=lengths,
                        **kc_sets,
                        **aux,
                    )
                    optimizer.zero_grad()
                    logits = model(dh2_batch)
                    loss = next_step_loss(
                        logits,
                        batch["correct"].to(device),
                        lengths,
                        is_repeat=batch["is_repeat"].to(device) if mask_repeats else None,
                    )
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
                            lengths=lengths,
                            time_gaps=dh2_batch.time_gaps,
                            saw_flags=dh2_batch.saw_flags,
                            group_ids=dh2_batch.group_ids,
                            durations=dh2_batch.durations,
                            idles=dh2_batch.idles,
                            **kc_sets,
                        )
                        ablated_batch = DH2KTBatch(
                            concept_ids=dh2_batch.concept_ids,
                            exercise_ids=dh2_batch.exercise_ids,
                            responses=dh2_batch.responses,
                            hyperedge_index=ablated_index,
                            lengths=lengths,
                            time_gaps=dh2_batch.time_gaps,
                            saw_flags=dh2_batch.saw_flags,
                            group_ids=dh2_batch.group_ids,
                            durations=dh2_batch.durations,
                            idles=dh2_batch.idles,
                            **kc_sets,
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
                # Train loss is only the fallback criterion — when a held-out
                # split is available, select on its AUC instead.
                if val_loader is not None:
                    val_auc, val_n = evaluate_auc(
                        model,
                        val_loader,
                        full_hyperedge_index,
                        device,
                        mask_repeats=mask_repeats,
                    )
                    model.train()
                    logger.info(
                        "epoch=%s/%s val_auc=%.4f val_n=%s",
                        epoch + 1,
                        budget.epochs,
                        val_auc,
                        val_n,
                    )
                    improved = np.isfinite(val_auc) and val_auc > best_val_auc
                    if improved:
                        best_val_auc = float(val_auc)
                        best_epoch_loss = mean_loss
                        best_epoch = epoch + 1
                        epochs_without_gain = 0
                    else:
                        epochs_without_gain += 1
                elif n_batches > 0 and mean_loss < best_epoch_loss:
                    improved = True
                    best_epoch_loss = mean_loss
                    best_epoch = epoch + 1
                else:
                    improved = False
                if improved:
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                if (
                    early_stop_patience is not None
                    and val_loader is not None
                    and epochs_without_gain >= early_stop_patience
                ):
                    logger.info(
                        "early stop at epoch=%s (no val AUC gain for %s epochs)",
                        epoch + 1,
                        epochs_without_gain,
                    )
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
                logger.info(
                    "restored best epoch=%s train_loss=%.4f val_auc=%s (of %s)",
                    best_epoch,
                    best_epoch_loss,
                    f"{best_val_auc:.4f}" if np.isfinite(best_val_auc) else "n/a",
                    budget.epochs,
                )
        finally:
            p0_log.setLevel(prev_p0_level)
        if val_loader is not None and np.isfinite(best_val_auc) and best_val_auc > -float("inf"):
            return float(best_val_auc)
        return None

    def build_model(
        n_concepts: int,
        n_exercises: int,
        *,
        hidden_dim: int = 128,
        n_hypergraph_layers: int = 2,
        dropout: float = 0.2,
        hyperedge_kinds: tuple[str, ...] = ("concept_prerequisite",),
        architecture: str = "v2",
        diffusion_alpha: float = 0.5,
        use_questions: bool = False,
        n_lstm_layers: int = 1,
        memory_dim: int = 16,
        max_degree: int = 16,
        graph_transport: float = 0.5,
        event_pool: str = "mean",
        transport: str = "clique",
        use_hyperedge_embed: bool = False,
        kind_conditioned: bool = False,
        recap_attention: bool = False,
        question_kc_agg: bool = False,
        question_graph: bool = False,
        question_incidence_control: str = "observed",
        question_hypergraph: bool = False,
        hint_hypergraph: bool = False,
        time_gap: bool = False,
        time_gap_mode: str = "both",
        time_split: bool = False,
        concept_forget: bool = False,
        saw_input: bool = False,
        group_embed: bool = False,
        n_groups: int = 1,
        expert_graph: bool = False,
    ) -> DH2KT:
        config = DH2KTConfig(
            n_concepts=n_concepts,
            n_exercises=n_exercises,
            hidden_dim=hidden_dim,
            embed_dim=hidden_dim,
            n_hypergraph_layers=n_hypergraph_layers,
            dropout=dropout,
            hyperedge_kinds=hyperedge_kinds,
            architecture=architecture,
            diffusion_alpha=diffusion_alpha,
            use_questions=use_questions,
            n_lstm_layers=n_lstm_layers,
            memory_dim=memory_dim,
            max_degree=max_degree,
            graph_transport=graph_transport,
            event_pool=event_pool,
            transport=transport,
            use_hyperedge_embed=use_hyperedge_embed,
            kind_conditioned=kind_conditioned,
            recap_attention=recap_attention,
            question_kc_agg=question_kc_agg,
            question_graph=question_graph,
            question_incidence_control=question_incidence_control,
            question_hypergraph=question_hypergraph,
            hint_hypergraph=hint_hypergraph,
            time_gap=time_gap,
            time_gap_mode=time_gap_mode,
            time_split=time_split,
            concept_forget=concept_forget,
            saw_input=saw_input,
            group_embed=group_embed,
            n_groups=n_groups,
            expert_graph=expert_graph,
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
    dropout: float = 0.2,
    graph_dropout: float = 0.0,
    graph_sensitivity_weight: float = 0.0,
    graph_sensitivity_margin: float = 0.05,
    graph_sensitivity_p: float = 0.9,
    hyperedge_spec: ConceptPrerequisiteSpec | None = None,
    session_hyperedges: list[Hyperedge] | None = None,
    architecture: str = "v2",
    diffusion_alpha: float = 0.5,
    max_users: int | None = None,
    use_questions: bool = False,
    n_lstm_layers: int = 1,
    window_mode: str = "first",
    val_frac: float = 0.0,
    early_stop_patience: int | None = None,
    use_graph: bool = True,
    mask_repeats: bool = False,
    seed: int | None = None,
    memory_dim: int = 16,
    max_degree: int = 16,
    graph_transport: float = 0.5,
    max_kcs: int = 6,
    event_pool: str = "mean",
    transport: str = "clique",
    use_hyperedge_embed: bool = False,
    kind_conditioned: bool = False,
    recap_attention: bool = False,
    question_kc_agg: bool = False,
    question_graph: bool = False,
    question_incidence_control: str = "observed",
    question_hypergraph: bool = False,
    hint_hypergraph: bool = False,
    hint_hyperedges: list[Hyperedge] | None = None,
    time_gap: bool = False,
    time_gap_mode: str = "both",
    time_split: bool = False,
    concept_forget: bool = False,
    saw_input: bool = False,
    group_embed: bool = False,
    expert_graph: bool = False,
    skill_item_kc_map: dict[int, list[int]] | None = None,
    full_qmatrix: bool = False,
    expert_edges: pd.DataFrame | None = None,
) -> TrainedFold:
    if not _TORCH_AVAILABLE:
        raise ImportError("train_fold requires PyTorch")
    if question_incidence_control not in ("observed", "zero"):
        raise ValueError(
            "question_incidence_control must be 'observed' or 'zero', "
            f"got {question_incidence_control!r}"
        )
    if question_incidence_control != "observed" and not question_graph:
        raise ValueError(
            "question_incidence_control requires question_graph=True so the "
            "question pathway and parameter count stay matched"
        )

    from dh2a_kt.hyperedge.indexing import select_session_hyperedges_for_training

    if seed is not None:
        seed_everything(seed)
        logger.info("seed=%d (weight init and batch order)", seed)

    device = torch.device(device)
    if max_users is not None:
        train_users = train_df["user_id"].unique()[:max_users]
        train_df = train_df[train_df["user_id"].isin(train_users)]
        eval_users = eval_df["user_id"].unique()[:max_users]
        eval_df = eval_df[eval_df["user_id"].isin(eval_users)]

    spec = hyperedge_spec or ConceptPrerequisiteSpec()
    extra_kc_ids: set[int] | None = None
    if skill_item_kc_map:
        extra_kc_ids = {int(k) for kcs in skill_item_kc_map.values() for k in kcs}
        if expert_edges is not None and len(expert_edges):
            extra_kc_ids.update(int(v) for v in expert_edges["src_kc"].tolist())
            extra_kc_ids.update(int(v) for v in expert_edges["dst_kc"].tolist())
    elif expert_edges is not None and len(expert_edges):
        extra_kc_ids = set()
        extra_kc_ids.update(int(v) for v in expert_edges["src_kc"].tolist())
        extra_kc_ids.update(int(v) for v in expert_edges["dst_kc"].tolist())
    kc_to_idx, item_to_idx = build_entity_maps(
        pd.concat([train_df, eval_df], ignore_index=True),
        e_pre,
        extra_kc_ids=extra_kc_ids,
    )
    if question_hypergraph and architecture == "v4":
        # Isolated multi-KC question hyperedges — not E_pre chains, not v5
        # memory transport. Incidence is observed item metadata.
        from dh2a_kt.data.events import question_kc_sets
        from dh2a_kt.hyperedge.construction import build_question_hyperedges

        kinds = ["question_concepts"]
        kc_sets = question_kc_sets(train_df)
        clean_hyperedges = build_question_hyperedges(kc_sets, fold=spec.fold)
        logger.info(
            "question_hypergraph: %d multi-KC question hyperedges (train items)",
            len(clean_hyperedges),
        )
    else:
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
        kinds = ["concept_prerequisite"]
        if architecture == "v5":
            # The KC set of a question is observed metadata, so it needs no
            # train-only gating; build it from every split the run can see.
            from dh2a_kt.data.events import question_kc_sets
            from dh2a_kt.hyperedge.construction import build_question_hyperedges

            kc_sets = question_kc_sets(pd.concat([train_df, eval_df], ignore_index=True))
            question_hyperedges = build_question_hyperedges(kc_sets, fold=spec.fold)
            clean_hyperedges = list(clean_hyperedges) + question_hyperedges
            kinds.append("question_concepts")
            logger.info(
                "v5: added %d question hyperedges over %d questions",
                len(question_hyperedges),
                len(kc_sets),
            )
        if session_hyperedges:
            selected = select_session_hyperedges_for_training(session_hyperedges)
            clean_hyperedges = list(clean_hyperedges) + list(selected)
            kinds.append("session")
            logger.info(
                "Added %d session hyperedges (concept projection) for training",
                len(selected),
            )
    hyperedge_index = hyperedge_index_from_list(clean_hyperedges, kc_to_idx)
    if not use_graph:
        # Graph-contribution ablation: keep the architecture and the kind list so
        # state dicts stay comparable, but feed zero hyperedges throughout.
        hyperedge_index = empty_hyperedge_index(device, kinds=tuple(kinds))
        clean_hyperedges = []
        logger.info("use_graph=False: training with zero hyperedges (graph ablation)")
    n_groups = 1
    if group_embed:
        from dh2a_kt.data.aux_signals import dense_group_ids

        train_df = train_df.copy()
        eval_df = eval_df.copy()
        if "teacher_id" in train_df.columns:
            train_codes, group_map, n_groups = dense_group_ids(train_df["teacher_id"])
            train_df["group_id"] = train_codes
            eval_df["group_id"] = (
                eval_df["teacher_id"].map(group_map).fillna(0).astype("int64")
                if "teacher_id" in eval_df.columns
                else 0
            )
            n_matched = int((train_df["group_id"] > 0).sum())
            logger.info(
                "group_embed: %d train groups, %d/%d train rows matched",
                n_groups - 1,
                n_matched,
                len(train_df),
            )
        else:
            train_df["group_id"] = 0
            eval_df["group_id"] = 0
            logger.warning("group_embed=True but teacher_id missing; all group_id=0")
    model = build_model(
        len(kc_to_idx),
        len(item_to_idx),
        hidden_dim=hidden_dim,
        n_hypergraph_layers=n_hypergraph_layers,
        dropout=dropout,
        hyperedge_kinds=tuple(kinds),
        architecture=architecture,
        diffusion_alpha=diffusion_alpha,
        use_questions=use_questions,
        n_lstm_layers=n_lstm_layers,
        memory_dim=memory_dim,
        max_degree=max_degree,
        graph_transport=graph_transport,
        event_pool=event_pool,
        transport=transport,
        use_hyperedge_embed=use_hyperedge_embed,
        kind_conditioned=kind_conditioned,
        recap_attention=recap_attention,
        question_kc_agg=question_kc_agg,
        question_graph=question_graph,
        question_incidence_control=question_incidence_control,
        question_hypergraph=question_hypergraph,
        hint_hypergraph=hint_hypergraph,
        time_gap=time_gap,
        time_gap_mode=time_gap_mode,
        time_split=time_split,
        concept_forget=concept_forget,
        saw_input=saw_input,
        group_embed=group_embed,
        n_groups=n_groups,
        expert_graph=expert_graph,
    ).to(device)
    if (question_kc_agg or question_graph) and architecture == "v4":
        # Observed Q–KC incidence from TRAIN only (not E_pre / not eval).
        from dh2a_kt.data.events import pad_kc_matrix, question_kc_sets

        raw_sets = question_kc_sets(train_df)
        rev_item = {idx: raw for raw, idx in item_to_idx.items()}
        if question_kc_agg:
            kc_lists: list[list[int]] = []
            for item_idx in range(len(item_to_idx)):
                raw_item = rev_item[item_idx]
                dense = [kc_to_idx[k] for k in raw_sets.get(raw_item, []) if k in kc_to_idx]
                kc_lists.append(dense)
            ids_np, mask_np = pad_kc_matrix(kc_lists, width=max_kcs)
            model.set_question_kc_table(
                torch.as_tensor(ids_np, device=device),
                torch.as_tensor(mask_np, device=device),
            )
            logger.info(
                "question_kc_agg: registered KC table for %d items (width=%d)",
                len(item_to_idx),
                max_kcs,
            )
        if question_graph:
            A_qs = torch.zeros(len(item_to_idx), len(kc_to_idx), device=device)
            for item_idx in range(len(item_to_idx)):
                raw_item = rev_item[item_idx]
                kcs = list(raw_sets.get(raw_item, []))
                if full_qmatrix and skill_item_kc_map is not None:
                    kcs = list(skill_item_kc_map.get(raw_item, kcs))
                for k in kcs:
                    if k in kc_to_idx:
                        A_qs[item_idx, kc_to_idx[k]] = 1.0
            observed_links = int(A_qs.sum().item())
            observed_linked_items = int((A_qs.sum(dim=1) > 0).sum().item())
            if question_incidence_control == "zero":
                # Capacity-matched Q←KC twin: retain question_embed,
                # question_gcn_linear and question_in_proj, but remove only
                # the incidence message A_norm @ concept_embed.
                A_qs.zero_()
            model.set_question_kc_incidence(A_qs)
            n_linked = int((A_qs.sum(dim=1) > 0).sum().item())
            logger.info(
                "question_graph: registered Q–KC incidence for %d/%d items "
                "(mean degree=%.2f, full_qmatrix=%s, control=%s; "
                "observed links=%d across %d items)",
                n_linked,
                len(item_to_idx),
                float(A_qs.sum().item()) / max(len(item_to_idx), 1),
                full_qmatrix,
                question_incidence_control,
                observed_links,
                observed_linked_items,
            )
    if hint_hypergraph and architecture == "v4":
        from dh2a_kt.hyperedge.indexing import (
            item_hyperedge_index_from_list,
            select_hint_item_hyperedges,
        )

        selected_hint = select_hint_item_hyperedges(hint_hyperedges or [])
        hint_index = item_hyperedge_index_from_list(selected_hint, item_to_idx)
        model.set_hint_hyperedge_index(hint_index.to(device))
        n_inc = int(hint_index.size(1)) if hint_index.numel() else 0
        logger.info(
            "hint_hypergraph: %d item hyperedges (incidence=%d) — not projected to concepts",
            len(selected_hint),
            n_inc,
        )
    if expert_graph and architecture == "v4":
        A_expert = torch.zeros(len(kc_to_idx), len(kc_to_idx), device=device)
        n_kept = 0
        if expert_edges is not None and len(expert_edges):
            for row in expert_edges.itertuples(index=False):
                src, dst = int(row.src_kc), int(row.dst_kc)
                if src in kc_to_idx and dst in kc_to_idx:
                    A_expert[kc_to_idx[dst], kc_to_idx[src]] = 1.0
                    n_kept += 1
        model.set_expert_adjacency(A_expert)
        logger.info(
            "expert_graph: %d directed pairs on %d concepts (non-absorbable)",
            n_kept,
            len(kc_to_idx),
        )
    logger.info(
        "train_fold: architecture=%s concepts=%d exercises=%d hyperedges=%d kinds=%s device=%s",
        architecture,
        len(kc_to_idx),
        len(item_to_idx),
        len(clean_hyperedges),
        kinds,
        device,
    )

    # Model selection must not touch eval_df: P0 reports AUC on valid+test, so
    # the selection split is carved out of the training users instead.
    val_df: pd.DataFrame | None = None
    if val_frac > 0.0:
        users = np.sort(train_df["user_id"].unique())
        # Fixed regardless of `seed`: multi-seed runs should vary initialisation
        # and batch order, not which learners are held out for selection.
        rng = np.random.default_rng(42)
        rng.shuffle(users)
        n_val = max(1, int(round(len(users) * val_frac)))
        val_users = set(users[:n_val].tolist())
        val_df = train_df[train_df["user_id"].isin(val_users)]
        train_df = train_df[~train_df["user_id"].isin(val_users)]
        logger.info(
            "internal selection split: %d train users / %d val users (val_frac=%.2f)",
            train_df["user_id"].nunique(),
            len(val_users),
            val_frac,
        )

    def _loader(frame: pd.DataFrame, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            UserSequenceDataset(
                frame,
                kc_to_idx,
                item_to_idx,
                max_seq_len=budget.max_seq_len,
                window_mode=window_mode,
                collapse_events=architecture == "v5",
                max_kcs=max_kcs,
                include_time_gaps=time_gap or concept_forget,
                include_saw=saw_input,
                include_group=group_embed,
                include_time_split=time_split,
            ),
            batch_size=budget.batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
        )

    train_loader = _loader(train_df, shuffle=True)
    eval_loader = _loader(eval_df, shuffle=False)
    val_loader = _loader(val_df, shuffle=False) if val_df is not None else None
    val_auc = train_one_fold(
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
        val_loader=val_loader,
        early_stop_patience=early_stop_patience,
        mask_repeats=mask_repeats,
    )
    return TrainedFold(
        model=model,
        eval_loader=eval_loader,
        kc_to_idx=kc_to_idx,
        item_to_idx=item_to_idx,
        clean_hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
        clean_hyperedges=clean_hyperedges,
        device=device,
        val_auc=val_auc,
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
    dropout: float = 0.2,
    graph_dropout: float = 0.0,
    graph_sensitivity_weight: float = 0.0,
    graph_sensitivity_margin: float = 0.05,
    graph_sensitivity_p: float = 0.9,
    hyperedge_spec: ConceptPrerequisiteSpec | None = None,
    session_hyperedges: list[Hyperedge] | None = None,
    architecture: str = "v2",
    diffusion_alpha: float = 0.5,
    max_users: int | None = None,
    checkpoint_path: Path | None = None,
    use_questions: bool = False,
    n_lstm_layers: int = 1,
    window_mode: str = "first",
    val_frac: float = 0.0,
    early_stop_patience: int | None = None,
    use_graph: bool = True,
    mask_repeats: bool = False,
    seed: int | None = None,
    memory_dim: int = 16,
    max_degree: int = 16,
    graph_transport: float = 0.5,
    max_kcs: int = 6,
    event_pool: str = "mean",
    transport: str = "clique",
    use_hyperedge_embed: bool = False,
    kind_conditioned: bool = False,
    recap_attention: bool = False,
    question_kc_agg: bool = False,
    question_graph: bool = False,
    question_incidence_control: str = "observed",
    question_hypergraph: bool = False,
    hint_hypergraph: bool = False,
    hint_hyperedges: list[Hyperedge] | None = None,
    time_gap: bool = False,
    time_gap_mode: str = "both",
    time_split: bool = False,
    concept_forget: bool = False,
    saw_input: bool = False,
    group_embed: bool = False,
    expert_graph: bool = False,
    skill_item_kc_map: dict[int, list[int]] | None = None,
    full_qmatrix: bool = False,
    expert_edges: pd.DataFrame | None = None,
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
        dropout=dropout,
        graph_dropout=graph_dropout,
        graph_sensitivity_weight=graph_sensitivity_weight,
        graph_sensitivity_margin=graph_sensitivity_margin,
        graph_sensitivity_p=graph_sensitivity_p,
        hyperedge_spec=hyperedge_spec,
        session_hyperedges=session_hyperedges,
        architecture=architecture,
        diffusion_alpha=diffusion_alpha,
        max_users=max_users,
        use_questions=use_questions,
        n_lstm_layers=n_lstm_layers,
        window_mode=window_mode,
        val_frac=val_frac,
        early_stop_patience=early_stop_patience,
        use_graph=use_graph,
        mask_repeats=mask_repeats,
        seed=seed,
        memory_dim=memory_dim,
        max_degree=max_degree,
        graph_transport=graph_transport,
        max_kcs=max_kcs,
        event_pool=event_pool,
        transport=transport,
        use_hyperedge_embed=use_hyperedge_embed,
        kind_conditioned=kind_conditioned,
        recap_attention=recap_attention,
        question_kc_agg=question_kc_agg,
        question_graph=question_graph,
        question_incidence_control=question_incidence_control,
        question_hypergraph=question_hypergraph,
        hint_hypergraph=hint_hypergraph,
        hint_hyperedges=hint_hyperedges,
        time_gap=time_gap,
        time_gap_mode=time_gap_mode,
        time_split=time_split,
        concept_forget=concept_forget,
        saw_input=saw_input,
        group_embed=group_embed,
        expert_graph=expert_graph,
        skill_item_kc_map=skill_item_kc_map,
        full_qmatrix=full_qmatrix,
        expert_edges=expert_edges,
    )
    auc, n_predictions = evaluate_auc(
        trained.model,
        trained.eval_loader,
        trained.clean_hyperedge_index,
        trained.device,
        mask_repeats=mask_repeats,
    )
    if checkpoint_path is not None:
        from dh2a_kt.train.checkpoint import save_trained_fold

        save_trained_fold(Path(checkpoint_path), trained)
        logger.info("wrote DH2-KT checkpoint -> %s", checkpoint_path)
    return FoldResult(
        fold=-1,
        auc=auc,
        n_predictions=n_predictions,
        val_auc=trained.val_auc,
    )


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

    def _val_auc(result: FoldResult) -> float | None:
        return getattr(result, "val_auc", None)

    val_mean = [v for v in (_val_auc(r) for r in results) if v is not None]

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
                    "val_auc": _val_auc(fold_result),
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
                "val_auc": float(np.nanmean(val_mean)) if val_mean else None,
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
                        "val_auc": _val_auc(fold_result),
                    }
                )

        for model in comparison_models:
            p0_row = load_p0_baseline_summary(dataset, model=model, graph_construction="train_only")
            p0_auc = float(p0_row.iloc[0]["auc"]) if not p0_row.empty else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "fold": "mean",
                    "reference_model": model,
                    "p0_auc": p0_auc,
                    "dh2_kt_auc": dh2_mean,
                    "delta_auc": dh2_mean - p0_auc if np.isfinite(p0_auc) else float("nan"),
                    "budget_reference": budget.reference_model,
                    "batch_size": budget.batch_size,
                    "epochs": budget.epochs,
                    "comparison_type": comparison_type,
                    "budget_note": budget.note,
                    "n_predictions": sum(r.n_predictions for r in results),
                    "val_auc": float(np.nanmean(val_mean)) if val_mean else None,
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
