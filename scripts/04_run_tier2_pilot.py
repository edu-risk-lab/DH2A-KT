#!/usr/bin/env python3
"""Pha 4-5 driver (docs/idea-D-plan.md): run the Tier 2 agent pilot
(Diagnostician -> Critic -> Tutor/Hint) over a sampled subset of Tier 1
output. Requires an LLMClient implementation (dh2a_kt.agents.LLMClient) wired
to a local runtime (Qwen3-8B-Instruct quantized, per section 7 of the plan)
or an API client, and Tier 1 (script 03) to be trained first.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--sample-size", type=int, default=500,
                         help="pilot sample size per docs/idea-D-plan.md Pha 4 (vai tram-vai nghin interaction)")
    parser.parse_args()
    raise NotImplementedError(
        "Pha 4-5 (docs/idea-D-plan.md): requires a trained Tier 1 model "
        "(script 03) and an LLMClient backend. Agent classes are implemented "
        "in dh2a_kt/agents/; this driver's orchestration + LLM backend wiring "
        "is left as a TODO."
    )


if __name__ == "__main__":
    raise SystemExit(main())
