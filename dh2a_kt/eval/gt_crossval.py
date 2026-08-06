"""Ground-truth cross-validation for concept-prerequisite hyperedges (M7).

Compares train-only inferred prerequisite structure against Junyi's expert
DAG using P0's ``compute_overlap_metrics`` / ``precision_recall_sweep`` (P0
Section 3.8, Table 15).  Runs the comparison for both raw ``E_pre`` and
hyperedge chain projections so we can report whether grouping chains changes
alignment with expert annotations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dh2a_kt.hyperedge.construction import Hyperedge, build_concept_prerequisite_hyperedges
from dh2a_kt.p0_bridge import (
    DEFAULT_K_LIST,
    align_kc_ids,
    compute_overlap_metrics,
    diagnose_disagreement,
    load_expert_dag,
    precision_recall_sweep,
)

try:
    from src.gt_cross_validation import disagreement_dict_to_dataframe  # noqa: WPS433
except ImportError:  # pragma: no cover

    def disagreement_dict_to_dataframe(diag: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for bucket, triples in diag.items():
            for i, t in enumerate(triples):
                rows.append(
                    {
                        "category": bucket,
                        "rank": i + 1,
                        "src_name": t[0],
                        "dst_name": t[1],
                        "score": t[2],
                    }
                )
        return pd.DataFrame(rows)


logger = logging.getLogger(__name__)

JUNYI_RAW_DIRNAME = "junyi"
JUNYI_EXPERT_CANDIDATES = (
    "junyi_dag.csv",
    "relationship_annotation_training.csv",
    "relationship_annotation_testing.csv",
)


def _p0_stable_int64(value: object) -> int:
    """Same exercise-name hash as ``external/p0_leakage_audit/src/preprocess.py``."""
    import hashlib

    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def resolve_junyi_expert_dag_path(p0_root: Path, gt_cfg: dict, repo_root: Path) -> Path:
    """Locate Junyi expert prerequisite CSV (Chang et al. annotation release)."""
    configured = gt_cfg.get("expert_dag_path")
    candidates: list[Path] = []
    if configured:
        path = Path(configured)
        candidates.append(path if path.is_absolute() else (repo_root / configured).resolve())
    raw_junyi = p0_root / "data" / "raw" / JUNYI_RAW_DIRNAME
    for name in JUNYI_EXPERT_CANDIDATES:
        candidates.append((raw_junyi / name).resolve())
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Junyi expert DAG not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def build_junyi_kc_name_to_id(p0_root: Path, *, cache_path: Path | None = None) -> dict[str, int]:
    """Build exercise-name → ``kc_id`` map using P0's stable hash."""
    raw_dir = p0_root / "data" / "raw" / JUNYI_RAW_DIRNAME
    names: set[str] = set()
    exercise_table = raw_dir / "junyi_Exercise_table.csv"
    if exercise_table.exists():
        table = pd.read_csv(exercise_table, usecols=["name"])
        names.update(table["name"].astype(str).str.strip())
    for fname in JUNYI_EXPERT_CANDIDATES:
        path = raw_dir / fname
        if not path.exists():
            continue
        ann = pd.read_csv(path)
        if {"Exercise_A", "Exercise_B"}.issubset(ann.columns):
            names.update(ann["Exercise_A"].astype(str).str.strip())
            names.update(ann["Exercise_B"].astype(str).str.strip())
    names.discard("")
    if not names:
        raise FileNotFoundError(
            f"No Junyi exercise names found under {raw_dir} "
            "(need junyi_Exercise_table.csv and/or relationship_annotation_*.csv)."
        )
    mapping = {name: _p0_stable_int64(name) for name in sorted(names)}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        logger.info("Wrote Junyi KC name map (%d names) to %s", len(mapping), cache_path)
    return mapping


def load_or_build_junyi_kc_name_to_id(p0_root: Path) -> dict[str, int]:
    mapping_path = p0_root / "data" / "processed" / JUNYI_RAW_DIRNAME / "kc_name_to_id.json"
    if mapping_path.exists():
        return {str(k): int(v) for k, v in json.loads(mapping_path.read_text(encoding="utf-8")).items()}
    logger.info("kc_name_to_id.json missing — building from Junyi raw tables")
    return build_junyi_kc_name_to_id(p0_root, cache_path=mapping_path)


@dataclass
class GTCrossvalResult:
    dataset: str
    fold: int
    alignment_report: dict[str, Any]
    n_expert_edges_matched: int
    n_e_pre_edges: int
    n_hyperedge_inferred_edges: int
    e_pre_sweep: pd.DataFrame
    hyperedge_sweep: pd.DataFrame
    e_pre_summary: dict[str, Any]
    hyperedge_summary: dict[str, Any]
    disagreement_hyperedge: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=lambda: ["e_pre", "hyperedge_chain"])


def hyperedges_to_inferred_prerequisite_edges(hyperedges: list[Hyperedge]) -> pd.DataFrame:
    """Project ordered concept chains to consecutive directed prerequisite pairs.

    Each hyperedge ``[(concept, c0), (concept, c1), ...]`` yields edges
    ``(c0, c1), (c1, c2), ...``.  Duplicate pairs inherit the maximum chain
    length as ``support`` (and ``weight`` for P0 sort compatibility).
    """
    support: dict[tuple[int, int], float] = {}
    for he in hyperedges:
        concepts = [int(m[1]) for m in he.members if m[0] == "concept"]
        chain_len = float(he.provenance.get("chain_len", len(concepts)))
        for i in range(len(concepts) - 1):
            edge = (concepts[i], concepts[i + 1])
            support[edge] = max(support.get(edge, 0.0), chain_len)
    rows = [
        {"src_kc": u, "dst_kc": v, "support": sc, "weight": sc}
        for (u, v), sc in support.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["src_kc", "dst_kc", "support", "weight"])
    return pd.DataFrame(rows)


def _resolve_k_list(k_list: list[int] | None, n_expert: int) -> list[int]:
    base = list(k_list if k_list is not None else DEFAULT_K_LIST)
    extra = [k for k in (n_expert, 5000) if k not in base]
    return sorted(set(base + extra))


def _summary_block(
    inferred: pd.DataFrame,
    expert_matched: pd.DataFrame,
    n_expert: int,
) -> dict[str, Any]:
    k_expert = max(n_expert, 0)
    return {
        "K_equal_|expert|": compute_overlap_metrics(inferred, expert_matched, k_expert),
        "K_5000": compute_overlap_metrics(inferred, expert_matched, 5000),
        "n_inferred_edges_available": len(inferred),
    }


def run_gt_crossval_for_fold(
    *,
    e_pre: pd.DataFrame,
    hyperedges: list[Hyperedge],
    expert_matched: pd.DataFrame,
    alignment_report: dict[str, Any],
    dataset: str,
    fold: int,
    k_list: list[int] | None = None,
    kc_name_to_id: dict[str, int] | None = None,
    n_disagreement_examples: int = 20,
) -> GTCrossvalResult:
    """Compare ``E_pre`` and hyperedge chain projections to matched expert edges."""
    n_expert = len(expert_matched)
    sweep_k = _resolve_k_list(k_list, n_expert)

    hyperedge_inferred = hyperedges_to_inferred_prerequisite_edges(hyperedges)

    e_pre_sweep = precision_recall_sweep(e_pre, expert_matched, sweep_k)
    hyperedge_sweep = precision_recall_sweep(hyperedge_inferred, expert_matched, sweep_k)

    diag_k = max(n_expert, 1)
    disagreement = diagnose_disagreement(
        hyperedge_inferred,
        expert_matched,
        top_k=diag_k,
        n_examples=n_disagreement_examples,
        kc_name_to_id=kc_name_to_id,
    )

    return GTCrossvalResult(
        dataset=dataset,
        fold=fold,
        alignment_report=alignment_report,
        n_expert_edges_matched=n_expert,
        n_e_pre_edges=len(e_pre),
        n_hyperedge_inferred_edges=len(hyperedge_inferred),
        e_pre_sweep=e_pre_sweep,
        hyperedge_sweep=hyperedge_sweep,
        e_pre_summary=_summary_block(e_pre, expert_matched, n_expert),
        hyperedge_summary=_summary_block(hyperedge_inferred, expert_matched, n_expert),
        disagreement_hyperedge=disagreement,
    )


def load_junyi_gt_inputs(
    dh2_cfg: dict,
    p0_cfg: dict,
    *,
    fold: int,
    p0_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, Any]]:
    """Load expert DAG, matched expert edges, train KC ids, and ``kc_name_to_id``."""
    from dh2a_kt.hyperedge.p0_inputs import get_fold_splits, load_interactions_with_ids

    from dh2a_kt.hyperedge.p0_inputs import REPO_ROOT

    gt_cfg = dh2_cfg.get("ground_truth_crossval", {})

    expert_path = resolve_junyi_expert_dag_path(p0_root, gt_cfg, REPO_ROOT)
    kc_name_to_id = load_or_build_junyi_kc_name_to_id(p0_root)

    expert_loaded = load_expert_dag(
        expert_path,
        schema="junyi_name",
        kc_name_to_id=kc_name_to_id,
        prereq_threshold=float(gt_cfg.get("prereq_threshold", 0.5)),
    )
    expert_loaded = expert_loaded.drop_duplicates(subset=["src_kc", "dst_kc"], keep="first")

    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, fold)
    train_kc_ids = set(int(x) for x in splits["train"]["kc_id"].unique())

    aligned, alignment_report = align_kc_ids(expert_loaded.copy(), train_kc_ids=train_kc_ids)
    alignment_report["expert_dag_path"] = str(expert_path)
    expert_matched = aligned.loc[aligned["status"] == "matched", ["src_kc", "dst_kc", "confidence_score"]].copy()

    return expert_matched, expert_loaded, kc_name_to_id, alignment_report


def build_gt_crossval_from_p0_exports(
    dh2_cfg: dict,
    p0_cfg: dict,
    *,
    fold: int,
    p0_root: Path,
) -> GTCrossvalResult:
    """End-to-end M7: load Junyi inputs, build hyperedges, run GT CV."""
    from dh2a_kt.hyperedge.p0_inputs import load_e_pre

    he_cfg = dh2_cfg.get("hyperedge", {}).get("concept_prerequisite", {})
    gt_cfg = dh2_cfg.get("ground_truth_crossval", {})

    expert_matched, _expert_loaded, kc_name_to_id, alignment_report = load_junyi_gt_inputs(
        dh2_cfg,
        p0_cfg,
        fold=fold,
        p0_root=p0_root,
    )
    alignment_report["fold"] = fold
    alignment_report["dataset"] = dh2_cfg["dataset"]

    e_pre = load_e_pre(p0_cfg, fold)
    hyperedges = build_concept_prerequisite_hyperedges(
        e_pre,
        fold=fold,
        min_chain_len=int(he_cfg.get("min_chain_len", 3)),
        max_chain_len=int(he_cfg.get("max_chain_len", 8)),
    )

    return run_gt_crossval_for_fold(
        e_pre=e_pre,
        hyperedges=hyperedges,
        expert_matched=expert_matched,
        alignment_report=alignment_report,
        dataset=dh2_cfg["dataset"],
        fold=fold,
        k_list=list(gt_cfg.get("top_k_list", DEFAULT_K_LIST)),
        kc_name_to_id=kc_name_to_id,
    )


def write_gt_crossval_result(result: GTCrossvalResult, output_dir: Path) -> None:
    """Write Table-15-style artefacts for both inference sources."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "alignment_report.json").write_text(
        json.dumps(result.alignment_report, indent=2),
        encoding="utf-8",
    )

    result.e_pre_sweep.to_csv(output_dir / "overlap_metrics_e_pre_at_K.csv", index=False)
    result.hyperedge_sweep.to_csv(output_dir / "overlap_metrics_hyperedge_at_K.csv", index=False)

    summary = {
        "dataset": result.dataset,
        "fold": result.fold,
        "n_expert_edges_matched": result.n_expert_edges_matched,
        "n_e_pre_edges": result.n_e_pre_edges,
        "n_hyperedge_inferred_edges": result.n_hyperedge_inferred_edges,
        "e_pre": result.e_pre_summary,
        "hyperedge_chain": result.hyperedge_summary,
    }
    (output_dir / "overlap_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    disagreement_dict_to_dataframe(result.disagreement_hyperedge).to_csv(
        output_dir / "disagreement_examples_hyperedge.csv",
        index=False,
    )

    table_rows = []
    for source, sweep in (("e_pre", result.e_pre_sweep), ("hyperedge_chain", result.hyperedge_sweep)):
        for _, row in sweep.iterrows():
            table_rows.append(
                {
                    "source": source,
                    "top_k": int(row["top_k"]),
                    "edge_precision": float(row["edge_precision"]),
                    "edge_recall": float(row["edge_recall"]),
                    "edge_f1": float(row["edge_f1"]),
                    "direction_agreement": float(row["direction_agreement"]),
                    "reachability_f1": float(row["reachability_f1"]),
                }
            )
    pd.DataFrame(table_rows).to_csv(output_dir / "gt_crossval_table15_format.csv", index=False)

    logger.info("Wrote GT cross-validation artefacts under %s", output_dir)
