"""Tests for Tier 2 pilot (M8)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dh2a_kt.agents.critic import CriticAgent
from dh2a_kt.agents.critic_calibration import explanation_cites_probability
from dh2a_kt.agents.diagnostician import DiagnosticianAgent, Tier1Output
from dh2a_kt.agents.llm_backends import StubLLMClient
from dh2a_kt.agents.tutor_hint import TutorHintAgent

pytest.importorskip("torch")

from dh2a_kt.tier2.pilot import (  # noqa: E402
    PilotRecord,
    PilotSummary,
    run_tier2_pilot,
    sample_tier1_outputs,
    write_pilot_outputs,
)
from dh2a_kt.train.checkpoint import load_trained_fold, save_trained_fold  # noqa: E402
from dh2a_kt.train.tier1 import TrainingBudget, train_fold  # noqa: E402


def _synthetic_logs(n_users: int = 12, seq_len: int = 10) -> pd.DataFrame:
    rows = []
    for u in range(n_users):
        for t in range(seq_len):
            rows.append(
                {
                    "user_id": u,
                    "item_id": (u + t) % 5,
                    "kc_id": t % 4,
                    "timestamp": u * 1000 + t,
                    "correct": (u + t) % 2,
                }
            )
    return pd.DataFrame(rows)


def _tier1_output(prob: float = 0.42) -> Tier1Output:
    return Tier1Output(
        student_id="7",
        concept_id="3",
        predicted_correct_prob=prob,
        causal_hint_effect=None,
        recent_history_summary="item 1 kc 2 correct",
    )


def test_explanation_cites_probability_percent_form():
    assert explanation_cites_probability("Student has 69.3% chance", 0.693)
    assert explanation_cites_probability("P(correct) is about 75.6 percent", 0.756)
    assert not explanation_cites_probability("Student seems fine", 0.756)


def test_critic_calibration_overrides_false_flag():
    llm = StubLLMClient()
    tier1 = _tier1_output(0.693)
    verdict = CriticAgent(llm).review(
        tier1,
        "The model predicts 69.3% chance of correctness for this concept.",
    )
    assert verdict.flagged is False
    assert "calibration" in verdict.reason.lower() or "0.693" in verdict.reason


def test_stub_llm_agents_roundtrip():
    llm = StubLLMClient()
    tier1 = _tier1_output(0.615)
    explanation = DiagnosticianAgent(llm).explain(tier1)
    verdict = CriticAgent(llm).review(tier1, explanation)
    hint = TutorHintAgent(llm).propose_hint(tier1)
    assert "0.615" in explanation
    assert verdict.flagged is False
    assert hint.text


def test_stub_critic_flags_missing_probability():
    llm = StubLLMClient()
    tier1 = _tier1_output(0.615)
    verdict = CriticAgent(llm).review(tier1, "The student seems fine overall.")
    assert verdict.flagged is True


def test_run_tier2_pilot_toy():
    train_df = _synthetic_logs(n_users=10, seq_len=8)
    eval_df = _synthetic_logs(n_users=6, seq_len=8)
    e_pre = pd.DataFrame({"src_kc": [0, 1], "dst_kc": [1, 2], "weight": [1.0, 1.0]})
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=2,
        lr=1e-2,
        max_seq_len=8,
        matched_p0=True,
    )
    trained = train_fold(
        train_df,
        eval_df,
        e_pre,
        budget,
        device="cpu",
        hidden_dim=16,
    )
    samples = sample_tier1_outputs(trained, eval_df, sample_size=5, max_seq_len=8, seed=0)
    records, hyperedges = run_tier2_pilot(samples, StubLLMClient(), fold=0)
    assert len(records) == len(samples)
    assert len(hyperedges) == len(samples)
    assert all(not r.critic_flagged for r in records)


def test_checkpoint_roundtrip(tmp_path: Path):
    train_df = _synthetic_logs(n_users=8, seq_len=8)
    eval_df = _synthetic_logs(n_users=4, seq_len=8)
    e_pre = pd.DataFrame({"src_kc": [0], "dst_kc": [1], "weight": [1.0]})
    budget = TrainingBudget(
        reference_model="gkt",
        batch_size=4,
        epochs=1,
        lr=1e-2,
        max_seq_len=8,
        matched_p0=True,
    )
    trained = train_fold(train_df, eval_df, e_pre, budget, device="cpu", hidden_dim=16)
    ckpt = tmp_path / "fold0.pt"
    save_trained_fold(ckpt, trained)
    loaded = load_trained_fold(ckpt, device="cpu")
    assert loaded.kc_to_idx == trained.kc_to_idx
    assert len(loaded.clean_hyperedges) == len(trained.clean_hyperedges)


def test_write_pilot_outputs(tmp_path: Path):
    record = PilotRecord(
        student_id="1",
        concept_id="2",
        predicted_correct_prob=0.5,
        actual_correct=float("nan"),
        recent_history_summary="history",
        diagnostician_explanation="expl",
        critic_flagged=False,
        critic_reason="OK",
        hint_text="hint",
        session_hyperedge_id="agent_hint_1_2",
    )
    summary = PilotSummary(
        n_samples=1,
        n_flagged=0,
        flag_rate=0.0,
        n_session_hyperedges=1,
        llm_backend="stub",
        fold=0,
        dataset="toy",
    )
    records_path = tmp_path / "pilot.jsonl"
    summary_path = tmp_path / "summary.json"
    write_pilot_outputs([record], summary, records_path=records_path, summary_path=summary_path)
    assert records_path.exists()
    assert summary_path.exists()
