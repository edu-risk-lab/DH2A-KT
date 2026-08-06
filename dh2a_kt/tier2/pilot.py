"""Tier 2 agent pilot: sample Tier-1 predictions, run Diagnostician → Critic → Tutor."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch

    from dh2a_kt.agents.critic import CriticAgent
    from dh2a_kt.agents.diagnostician import DiagnosticianAgent, Tier1Output
    from dh2a_kt.agents.tutor_hint import TutorHintAgent
    from dh2a_kt.hyperedge.construction import Hyperedge
    from dh2a_kt.models.dh2_kt import DH2KTBatch
    from dh2a_kt.train.tier1 import TrainedFold, precompute_concept_states, shift_responses_for_next_step

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


@dataclass
class PilotRecord:
    student_id: str
    concept_id: str
    predicted_correct_prob: float
    actual_correct: float
    recent_history_summary: str
    diagnostician_explanation: str
    critic_flagged: bool
    critic_reason: str
    hint_text: str
    session_hyperedge_id: str


@dataclass
class PilotSummary:
    n_samples: int
    n_flagged: int
    flag_rate: float
    n_session_hyperedges: int
    llm_backend: str
    fold: int
    dataset: str


def _history_summary(rows: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, row in rows.iterrows():
        outcome = "correct" if row["correct"] else "incorrect"
        parts.append(f"item {row['item_id']} kc {row['kc_id']} {outcome}")
    return "; ".join(parts) if parts else "no prior history"


if _TORCH_AVAILABLE:

    def collect_prediction_events(
        trained: TrainedFold,
        eval_df: pd.DataFrame,
        *,
        max_seq_len: int,
    ) -> list[Tier1Output]:
        """Enumerate next-step predictions with student/concept metadata."""
        sorted_df = eval_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
        kc_to_idx = trained.kc_to_idx
        item_to_idx = trained.item_to_idx
        events: list[Tier1Output] = []
        model = trained.model
        device = trained.device
        hyperedge_index = trained.clean_hyperedge_index
        concept_states = precompute_concept_states(model, hyperedge_index, device)

        for user_id, group in sorted_df.groupby("user_id", sort=False):
            if len(group) < 2:
                continue
            chunk = group.iloc[:max_seq_len]
            length = len(chunk)
            concept_ids = torch.zeros(max_seq_len, dtype=torch.long)
            exercise_ids = torch.zeros(max_seq_len, dtype=torch.long)
            correct = torch.zeros(max_seq_len, dtype=torch.float)
            mapped_kc = chunk["kc_id"].map(kc_to_idx)
            mapped_item = chunk["item_id"].map(item_to_idx)
            if mapped_kc.isna().any() or mapped_item.isna().any():
                continue
            concept_ids[:length] = torch.from_numpy(mapped_kc.to_numpy(dtype=np.int64))
            exercise_ids[:length] = torch.from_numpy(mapped_item.to_numpy(dtype=np.int64))
            correct[:length] = torch.from_numpy(chunk["correct"].to_numpy(dtype=np.float32))
            responses = shift_responses_for_next_step(correct.unsqueeze(0)).squeeze(0)
            batch = DH2KTBatch(
                concept_ids=concept_ids.unsqueeze(0).to(device),
                exercise_ids=exercise_ids.unsqueeze(0).to(device),
                responses=responses.unsqueeze(0).to(device),
                hyperedge_index={k: v.to(device) for k, v in hyperedge_index.items()},
                concept_states=concept_states,
            )
            with torch.no_grad():
                logits = model(batch)
            pred = torch.sigmoid(logits[0, :-1, 0]).cpu().numpy()
            rows = chunk.reset_index(drop=True)
            for t in range(length - 1):
                target_row = rows.iloc[t + 1]
                history = _history_summary(rows.iloc[max(0, t - 4) : t + 1])
                events.append(
                    Tier1Output(
                        student_id=str(user_id),
                        concept_id=str(int(target_row["kc_id"])),
                        predicted_correct_prob=float(pred[t]),
                        causal_hint_effect=None,
                        recent_history_summary=history,
                    )
                )
        return events

    def sample_tier1_outputs(
        trained: TrainedFold,
        eval_df: pd.DataFrame,
        *,
        sample_size: int,
        max_seq_len: int,
        seed: int = 42,
        max_users: int | None = None,
    ) -> list[Tier1Output]:
        if max_users is not None:
            users = eval_df["user_id"].unique()[:max_users]
            eval_df = eval_df[eval_df["user_id"].isin(users)]
        events = collect_prediction_events(trained, eval_df, max_seq_len=max_seq_len)
        if not events:
            raise ValueError("no prediction events collected from eval split")
        rng = np.random.default_rng(seed)
        if sample_size >= len(events):
            return events
        indices = rng.choice(len(events), size=sample_size, replace=False)
        return [events[int(i)] for i in indices]

    def run_tier2_pilot(
        samples: list[Tier1Output],
        llm,
        *,
        fold: int,
    ) -> tuple[list[PilotRecord], list[Hyperedge]]:
        diagnostician = DiagnosticianAgent(llm)
        critic = CriticAgent(llm)
        tutor = TutorHintAgent(llm)
        records: list[PilotRecord] = []
        hyperedges: list[Hyperedge] = []

        for i, tier1_output in enumerate(samples):
            explanation = diagnostician.explain(tier1_output)
            verdict = critic.review(tier1_output, explanation)
            hint = tutor.propose_hint(tier1_output)
            session_he = tutor.write_session_hyperedge(hint, tier1_output, fold=fold)
            hyperedges.append(session_he)
            records.append(
                PilotRecord(
                    student_id=tier1_output.student_id,
                    concept_id=tier1_output.concept_id,
                    predicted_correct_prob=tier1_output.predicted_correct_prob,
                    actual_correct=float("nan"),
                    recent_history_summary=tier1_output.recent_history_summary,
                    diagnostician_explanation=explanation,
                    critic_flagged=verdict.flagged,
                    critic_reason=verdict.reason,
                    hint_text=hint.text,
                    session_hyperedge_id=session_he.hyperedge_id,
                )
            )
            if (i + 1) % 50 == 0:
                logger.info("tier2 pilot progress: %d/%d", i + 1, len(samples))
        return records, hyperedges


def write_pilot_outputs(
    records: list[PilotRecord],
    summary: PilotSummary,
    *,
    records_path: Path,
    summary_path: Path,
    hyperedges_path: Path | None = None,
    hyperedges: list | None = None,
) -> None:
    records_path = Path(records_path)
    summary_path = Path(summary_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    if hyperedges_path is not None and hyperedges is not None:
        hyperedges_path.write_text(
            json.dumps([asdict(he) for he in hyperedges], indent=2),
            encoding="utf-8",
        )
