"""Tier 2 (AgentDG-KT pilot) — agents that read Tier 1's frozen output and
produce natural-language explanations/hints. docs/idea-D-plan.md Pha 4-5.

All agents share the same LLM-client contract so the backend (local quantized
model vs. API) is swappable without touching agent logic — see
``LLMClient`` protocol below.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal contract every agent depends on. Implement this against a
    local runtime (e.g. vLLM/Ollama serving Qwen3-8B-Instruct quantized, per
    docs/idea-D-plan.md section 7) or an API client — agents don't care which.
    """

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        ...
