"""Tutor/Hint Agent (Tier 2) — selects/generates the next hint or exercise,
reading Tier 1 output. Writes a new hyperedge back into the graph (the
agentic closed-loop from Idea B/C). docs/idea-D-plan.md Pha 4."""

from __future__ import annotations

from dataclasses import dataclass

from dh2a_kt.agents import LLMClient
from dh2a_kt.agents.diagnostician import Tier1Output
from dh2a_kt.hyperedge.construction import Hyperedge

SYSTEM_PROMPT = """You are the Tutor/Hint Agent. Given a Tier-1 knowledge-state
prediction, propose ONE concrete next hint or exercise for the student.
Be specific and actionable; do not restate the probability back to the user."""


@dataclass
class HintProposal:
    text: str
    target_concept_id: str


class TutorHintAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose_hint(self, tier1_output: Tier1Output) -> HintProposal:
        user_prompt = (
            f"Student {tier1_output.student_id}, concept {tier1_output.concept_id}, "
            f"P(correct)={tier1_output.predicted_correct_prob:.3f}. "
            f"History: {tier1_output.recent_history_summary}\n"
            "Propose one next hint or exercise."
        )
        text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        return HintProposal(text=text, target_concept_id=tier1_output.concept_id)

    def write_session_hyperedge(
        self, proposal: HintProposal, tier1_output: Tier1Output, *, fold: int
    ) -> Hyperedge:
        """The agentic closed-loop write: this hint becomes a new graph
        member the next Tier-1 pass (or a later pilot round) can read."""
        return Hyperedge(
            hyperedge_id=f"agent_hint_{tier1_output.student_id}_{tier1_output.concept_id}",
            kind="session",
            members=[
                ("student", hash(tier1_output.student_id) % (2**31)),
                ("concept", hash(proposal.target_concept_id) % (2**31)),
                ("hint", hash(proposal.text) % (2**31)),
            ],
            fold=fold,
            train_only=False,  # generated at inference/pilot time, not from train logs
            provenance={"source": "tutor_hint_agent", "text": proposal.text},
        )
