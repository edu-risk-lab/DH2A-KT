"""ID-anchored KC-Jaccard for Tier-2 explanation faithfulness (KBS upgrade).

Precedent: Ni et al. (2026) KC-Jaccard on named concepts. XES3G5M has no
synonym/name vocabulary in-repo, so we match integer KC IDs only:

  G = KCs the Diagnostician claims to ground on (``GroundedKCs: [...]``
      trailer, with regex fallback on ``kc <id>`` mentions)
  S = {target concept} ∪ KCs in recent history ∪ 1-hop neighbors in E_pre

  JC = |G ∩ S| / |G ∪ S|   (0 if both empty)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

GROUNDED_TRAILER_RE = re.compile(
    r"GroundedKCs\s*:\s*\[([^\]]*)\]",
    re.IGNORECASE,
)
KC_MENTION_RE = re.compile(r"\bkc\s*[#:]?\s*(\d+)\b", re.IGNORECASE)
HISTORY_KC_RE = re.compile(r"\bkc\s+(\d+)\b", re.IGNORECASE)
INT_TOKEN_RE = re.compile(r"\d+")


def parse_grounded_kcs(explanation: str) -> set[int]:
    """Extract grounded KC IDs from a Diagnostician explanation.

    Prefers the structured ``GroundedKCs: [..]`` trailer; falls back to
    free-text ``kc <id>`` mentions for older JSONL without the trailer.
    """
    text = explanation or ""
    m = GROUNDED_TRAILER_RE.search(text)
    if m:
        return {int(x) for x in INT_TOKEN_RE.findall(m.group(1))}
    return {int(x) for x in KC_MENTION_RE.findall(text)}


def parse_history_kcs(recent_history_summary: str) -> set[int]:
    return {int(x) for x in HISTORY_KC_RE.findall(recent_history_summary or "")}


def build_e_pre_adjacency(e_pre: pd.DataFrame) -> dict[int, set[int]]:
    """Undirected 1-hop adjacency from audited prerequisite edges."""
    adj: dict[int, set[int]] = defaultdict(set)
    if e_pre is None or e_pre.empty:
        return {}
    for src, dst in zip(e_pre["src_kc"].tolist(), e_pre["dst_kc"].tolist()):
        s, d = int(src), int(dst)
        adj[s].add(d)
        adj[d].add(s)
    return dict(adj)


def support_set(
    *,
    concept_id: int | str,
    recent_history_summary: str,
    adjacency: dict[int, set[int]] | None = None,
) -> set[int]:
    """S = {target} ∪ history KCs ∪ 1-hop E_pre neighbors of those."""
    target = int(concept_id)
    seeds = {target} | parse_history_kcs(recent_history_summary)
    neighbors: set[int] = set()
    if adjacency:
        for kc in seeds:
            neighbors |= adjacency.get(kc, set())
    return seeds | neighbors


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class FaithfulnessRow:
    student_id: str
    concept_id: str
    grounded_kcs: list[int]
    support_kcs: list[int]
    kc_jaccard: float
    critic_flagged: bool | None
    predicted_correct_prob: float | None


@dataclass
class FaithfulnessReport:
    n_samples: int
    mean_kc_jaccard: float
    median_kc_jaccard: float
    n_empty_grounded: int
    n_flagged: int | None
    flag_rate: float | None
    source_records: str
    rows: list[FaithfulnessRow]


def score_record(
    record: dict,
    *,
    adjacency: dict[int, set[int]] | None = None,
) -> FaithfulnessRow:
    grounded = parse_grounded_kcs(str(record.get("diagnostician_explanation", "")))
    support = support_set(
        concept_id=record.get("concept_id", -1),
        recent_history_summary=str(record.get("recent_history_summary", "")),
        adjacency=adjacency,
    )
    flagged = record.get("critic_flagged")
    prob = record.get("predicted_correct_prob")
    return FaithfulnessRow(
        student_id=str(record.get("student_id", "")),
        concept_id=str(record.get("concept_id", "")),
        grounded_kcs=sorted(grounded),
        support_kcs=sorted(support),
        kc_jaccard=jaccard(grounded, support),
        critic_flagged=bool(flagged) if flagged is not None else None,
        predicted_correct_prob=float(prob) if prob is not None else None,
    )


def evaluate_explanation_faithfulness(
    records: list[dict],
    *,
    e_pre: pd.DataFrame | None = None,
    source_records: str = "",
) -> FaithfulnessReport:
    if not records:
        raise ValueError("no records to score")
    adjacency = build_e_pre_adjacency(e_pre) if e_pre is not None else {}
    rows = [score_record(r, adjacency=adjacency) for r in records]
    scores = [r.kc_jaccard for r in rows]
    mean_jc = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    mid = len(sorted_scores) // 2
    if len(sorted_scores) % 2:
        median_jc = sorted_scores[mid]
    else:
        median_jc = 0.5 * (sorted_scores[mid - 1] + sorted_scores[mid])
    flagged_vals = [r.critic_flagged for r in rows if r.critic_flagged is not None]
    n_flagged = sum(1 for f in flagged_vals if f) if flagged_vals else None
    flag_rate = (n_flagged / len(flagged_vals)) if flagged_vals else None
    return FaithfulnessReport(
        n_samples=len(rows),
        mean_kc_jaccard=mean_jc,
        median_kc_jaccard=median_jc,
        n_empty_grounded=sum(1 for r in rows if not r.grounded_kcs),
        n_flagged=n_flagged,
        flag_rate=flag_rate,
        source_records=source_records,
        rows=rows,
    )


def read_pilot_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_faithfulness_report(report: FaithfulnessReport, output_path: Path, *, include_rows: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    if not include_rows:
        payload.pop("rows", None)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
