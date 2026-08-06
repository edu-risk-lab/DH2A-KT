"""Swappable LLM backends for Tier 2 agents (M8).

``StubLLMClient`` is deterministic — used in tests and smoke runs without GPU/LLM.
``OllamaLLMClient`` talks to a local Ollama server (stdlib only, no extra deps).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


_PROB_RE = re.compile(r"P\(correct\)=([\d.]+)")


class StubLLMClient:
    """Rule-based backend: echoes Tier-1 numbers faithfully for gate testing."""

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        prob_match = _PROB_RE.search(user_prompt)
        prob = prob_match.group(1) if prob_match else "unknown"

        if "Critic Agent" in system_prompt:
            explanation = user_prompt.split("Diagnostician explanation:", 1)[-1].strip()
            if prob in explanation:
                return f"OK The explanation cites P(correct)={prob}."
            return f"FLAG The explanation does not faithfully cite P(correct)={prob}."

        if "Tutor/Hint Agent" in system_prompt:
            concept = "this concept"
            if "concept " in user_prompt:
                concept = user_prompt.split("concept ", 1)[1].split(",")[0]
            return (
                f"Try a short practice set on concept {concept} "
                f"(current P(correct)={prob}) before the next graded item."
            )

        return (
            f"The Tier-1 model estimates P(correct)={prob} for this student on this concept. "
            "Review recent attempts in the history summary and check for slipping prerequisites."
        )


@dataclass
class OllamaLLMClient:
    """HTTP client for Ollama ``/api/chat`` (local Qwen or similar)."""

    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 120.0

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Ollama unreachable at {self.base_url} (model={self.model!r}): {exc}"
            ) from exc
        message = body.get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError(f"Ollama returned empty content: {body!r}")
        return str(content).strip()


def resolve_llm_client(backend: str, *, ollama_model: str = "qwen2.5:7b", ollama_url: str = "http://127.0.0.1:11434"):
    if backend == "stub":
        return StubLLMClient()
    if backend == "ollama":
        return OllamaLLMClient(model=ollama_model, base_url=ollama_url)
    raise ValueError(f"unknown llm backend {backend!r}; expected 'stub' or 'ollama'")
