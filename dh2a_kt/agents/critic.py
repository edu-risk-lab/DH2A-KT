"""Critic/Judger Agent (Tier 2) — mandatory quality gate on Diagnostician
output, per docs/idea-D-plan.md ("bat buoc, khong tuy chon"). Flags drift
or hallucination before anything is treated as ground truth in a case study.
"""

from __future__ import annotations

from dataclasses import dataclass

from dh2a_kt.agents import LLMClient
from dh2a_kt.agents.critic_calibration import explanation_cites_probability
from dh2a_kt.agents.diagnostician import Tier1Output

SYSTEM_PROMPT = """You are the Critic Agent. You check a Diagnostician
explanation against the Tier-1 numeric prediction it was supposed to explain.
Flag ONLY when the explanation contradicts, ignores, or invents a materially
different probability than Tier-1 P(correct).

Do NOT flag equivalent formats: 0.693, 69.3%, and "69 percent" all match
P(correct)=0.693. Rounding to one decimal place in percent form is OK.

Answer with exactly one word first (OK or FLAG) then a one-sentence reason."""


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
        if flagged and explanation_cites_probability(
            diagnostician_explanation, tier1_output.predicted_correct_prob
        ):
            return CriticVerdict(
                flagged=False,
                reason=f"OK (format calibration) overrode LLM FLAG: {response.strip()}",
            )
        return CriticVerdict(flagged=flagged, reason=response)
