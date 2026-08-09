"""Diagnostician Agent (Tier 2) — docs/idea-D-plan.md / Idea C section "Doi agent".

Reads Tier 1's frozen knowledge-state output for one student and produces a
natural-language rationale. Does NOT re-infer knowledge state itself (that
would defeat the point of being a wrapper around Tier 1, and would reproduce
the "3-5 LLM calls per event" cost problem Idea B had before Idea C fixed it).

KBS upgrade: explanations must end with a structured ``GroundedKCs: [id, ...]``
trailer so ID-anchored KC-Jaccard can be computed corpus-wide. For the
independent-generation (IG) negative control, ``omit_tier1_prob=True`` strips
frozen ``P(correct)`` from the prompt while keeping interaction history.
"""

from __future__ import annotations

from dataclasses import dataclass

from dh2a_kt.agents import LLMClient

SYSTEM_PROMPT = """You are the Diagnostician Agent in a knowledge-tracing pilot.
You are given a Tier-1 model's ALREADY-COMPUTED knowledge-state prediction for
one student on one concept. Explain it in plain language for a teacher.
Do not invent a different probability or contradict the given score — your
job is explanation, not re-diagnosis. If the score seems surprising, say so
and suggest what a teacher should check, but do not silently override it.
Always end your reply with a single trailer line of the form:
GroundedKCs: [id1, id2, ...]
listing the integer knowledge-component IDs your rationale refers to
(target concept and any history KCs you cite). Use only IDs present in the
user."""

SYSTEM_PROMPT_IG = """You are the Diagnostician Agent in a knowledge-tracing pilot.
You see only the student's recent interaction history (no Tier-1 probability).
Write a short teacher-facing rationale about the student's likely readiness
on the named target concept, based solely on that history.
Always end your reply with a single trailer line of the form:
GroundedKCs: [id1, id2, ...]
listing the integer knowledge-component IDs your rationale refers to
(target concept and any history KCs you cite). Use only IDs present in the
prompt."""


@dataclass
class Tier1Output:
    student_id: str
    concept_id: str
    predicted_correct_prob: float
    causal_hint_effect: float | None  # from models/causal_layer.py, if available
    recent_history_summary: str


class DiagnosticianAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def explain(self, tier1_output: Tier1Output, *, omit_tier1_prob: bool = False) -> str:
        if omit_tier1_prob:
            user_prompt = (
                f"Student {tier1_output.student_id}, concept {tier1_output.concept_id}.\n"
                f"Recent history: {tier1_output.recent_history_summary}\n"
                "Explain the student's likely readiness on this concept to a teacher "
                "in 2-3 sentences. End with GroundedKCs: [...]."
            )
            return self.llm.complete(SYSTEM_PROMPT_IG, user_prompt)
        user_prompt = (
            f"Student {tier1_output.student_id}, concept {tier1_output.concept_id}.\n"
            f"Tier-1 predicted P(correct)={tier1_output.predicted_correct_prob:.3f}.\n"
            f"Estimated hint causal effect: {tier1_output.causal_hint_effect}.\n"
            f"Recent history: {tier1_output.recent_history_summary}\n"
            "Explain this prediction to a teacher in 2-3 sentences. "
            "End with GroundedKCs: [...]."
        )
        return self.llm.complete(SYSTEM_PROMPT, user_prompt)
