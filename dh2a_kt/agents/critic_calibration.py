"""Deterministic calibration helpers for the Critic agent."""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"(?<![\d.])(0\.\d+|\d+\.\d+|\d+)(?![\d.])")


def explanation_cites_probability(
    explanation: str,
    prob: float,
    *,
    atol: float = 0.015,
) -> bool:
    """Return True if *explanation* cites *prob* as decimal or equivalent percent.

    Accepts e.g. 0.693, 69.3%, or 69.3 percent for Tier-1 probability 0.693.
    """
    if not explanation.strip():
        return False
    targets = {prob, round(prob, 3)}
    for match in _NUM_RE.finditer(explanation):
        raw = match.group(1)
        try:
            val = float(raw)
        except ValueError:
            continue
        if val > 1.0:
            if val <= 100.0:
                val = val / 100.0
            else:
                continue
        if any(abs(val - t) <= atol for t in targets):
            return True
    return False
