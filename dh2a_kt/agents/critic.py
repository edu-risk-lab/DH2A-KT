"""Critic/Judger Agent (Tier 2) — mandatory quality gate on Diagnostician
output, per docs/idea-D-plan.md ("bat buoc, khong tuy chon"). Flags drift
or hallucination before anything is treated as ground truth in a case study.
"""

from __future__ import annotations

from dataclasses import dataclass

from dh2a_kt.agents import LLMClient
from dh2a_kt.agents.diagnostician import Tier1Output

SYSTEM_PROMPT = """You are the Critic Agent. You check a Diagnostician
explanation against the Tier-1 numeric prediction it was supposed to explain.
Flag ANY case where the explanation contradicts, ignores, or invents a
different number than the one given. Answer with exactly one word first
(OK or FLAG) then a one-sentence reason."""


@dataclass
class CriticVerdict:
    flagged: bool
    reason: str


class CriticAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, tier1_output: Tier1Output, diagnostician_explanation: str) -> CriticVerdict:
        user_prompt = (
            f"Tier-1 P(correct)={tier1_output.predicted_correct_prob:.3f}.\n"
            f"Diagnostician explanation: {diagnostician_explanation}"
        )
        response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        flagged = response.strip().upper().startswith("FLAG")
        return CriticVerdict(flagged=flagged, reason=response)
