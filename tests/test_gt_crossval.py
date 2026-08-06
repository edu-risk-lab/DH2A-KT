"""M7: ground-truth cross-validation helpers (unit + optional Junyi integration)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dh2a_kt.eval.gt_crossval import (
    GTCrossvalResult,
    hyperedges_to_inferred_prerequisite_edges,
    run_gt_crossval_for_fold,
    write_gt_crossval_result,
)
from dh2a_kt.hyperedge.construction import Hyperedge, build_concept_prerequisite_hyperedges
from dh2a_kt.hyperedge.p0_inputs import REPO_ROOT, P0_ROOT, e_pre_export_path, load_configs

JUNYI_CONFIG = REPO_ROOT / "configs" / "junyi.yaml"
JUNYI_PARQUET = P0_ROOT / "data" / "processed" / "junyi.parquet"
JUNYI_EXPERT = P0_ROOT / "data" / "raw" / "junyi" / "relationship_annotation_training.csv"
JUNYI_EXERCISE_TABLE = P0_ROOT / "data" / "raw" / "junyi" / "junyi_Exercise_table.csv"


def test_hyperedges_to_inferred_prerequisite_edges_chain():
    he = Hyperedge(
        hyperedge_id="c1",
        kind="concept_prerequisite",
        members=[("concept", 1), ("concept", 2), ("concept", 3), ("concept", 4)],
        fold=0,
        train_only=True,
        provenance={"chain_len": 4},
    )
    inferred = hyperedges_to_inferred_prerequisite_edges([he])
    assert set(zip(inferred["src_kc"], inferred["dst_kc"])) == {(1, 2), (2, 3), (3, 4)}
    assert inferred.loc[inferred["src_kc"] == 2, "support"].iloc[0] == pytest.approx(4.0)


def test_hyperedges_to_inferred_dedup_keeps_max_chain_len():
    he_short = Hyperedge(
        hyperedge_id="s",
        kind="concept_prerequisite",
        members=[("concept", 1), ("concept", 2), ("concept", 3)],
        fold=0,
        train_only=True,
        provenance={"chain_len": 3},
    )
    he_long = Hyperedge(
        hyperedge_id="l",
        kind="concept_prerequisite",
        members=[("concept", 1), ("concept", 2), ("concept", 3), ("concept", 4), ("concept", 5)],
        fold=0,
        train_only=True,
        provenance={"chain_len": 5},
    )
    inferred = hyperedges_to_inferred_prerequisite_edges([he_short, he_long])
    row = inferred.loc[(inferred["src_kc"] == 1) & (inferred["dst_kc"] == 2)].iloc[0]
    assert row["support"] == pytest.approx(5.0)


def test_run_gt_crossval_for_fold_toy():
    e_pre = pd.DataFrame(
        {
            "src_kc": [1, 2, 3, 10],
            "dst_kc": [2, 3, 4, 11],
            "weight": [5.0, 4.0, 3.0, 1.0],
        }
    )
    hyperedges = build_concept_prerequisite_hyperedges(
        e_pre,
        fold=0,
        min_chain_len=3,
        max_chain_len=8,
    )
    expert = pd.DataFrame(
        {
            "src_kc": [1, 2],
            "dst_kc": [2, 3],
            "confidence_score": [1.0, 1.0],
        }
    )
    result = run_gt_crossval_for_fold(
        e_pre=e_pre,
        hyperedges=hyperedges,
        expert_matched=expert,
        alignment_report={"alignment_rate": 1.0, "matched": 2},
        dataset="toy",
        fold=0,
        k_list=[2, 4],
    )
    assert isinstance(result, GTCrossvalResult)
    assert result.n_expert_edges_matched == 2
    assert len(result.e_pre_sweep) >= 2
    assert len(result.hyperedge_sweep) >= 2
    assert "edge_precision" in result.e_pre_sweep.columns
    assert result.e_pre_summary["K_equal_|expert|"]["edge_recall"] >= 0.0


def test_write_gt_crossval_result(tmp_path: Path):
    sweep = pd.DataFrame(
        {
            "top_k": [2],
            "n_inferred_truncated": [2],
            "n_expert_edges": [2],
            "edge_precision": [0.5],
            "edge_recall": [0.5],
            "edge_f1": [0.5],
            "direction_agreement": [1.0],
            "reachability_precision": [0.5],
            "reachability_recall": [0.5],
            "reachability_f1": [0.5],
            "node_jaccard": [0.5],
        }
    )
    result = GTCrossvalResult(
        dataset="toy",
        fold=0,
        alignment_report={"alignment_rate": 1.0},
        n_expert_edges_matched=2,
        n_e_pre_edges=2,
        n_hyperedge_inferred_edges=1,
        e_pre_sweep=sweep,
        hyperedge_sweep=sweep,
        e_pre_summary={"K_equal_|expert|": {"edge_f1": 0.5}, "K_5000": {"edge_f1": 0.5}},
        hyperedge_summary={"K_equal_|expert|": {"edge_f1": 0.5}, "K_5000": {"edge_f1": 0.5}},
        disagreement_hyperedge={"inferred_only": [], "expert_only": [], "wrong_direction": []},
    )
    out = tmp_path / "gt"
    write_gt_crossval_result(result, out)
    assert (out / "overlap_metrics_e_pre_at_K.csv").exists()
    assert (out / "overlap_metrics_hyperedge_at_K.csv").exists()
    assert (out / "gt_crossval_table15_format.csv").exists()
    assert (out / "overlap_metrics_summary.json").exists()


def test_build_junyi_kc_name_to_id_matches_p0_hash():
    from dh2a_kt.eval.gt_crossval import _p0_stable_int64, build_junyi_kc_name_to_id

    if not JUNYI_EXERCISE_TABLE.exists():
        pytest.skip("Junyi exercise table not present")
    mapping = build_junyi_kc_name_to_id(P0_ROOT)
    assert "radius_diameter_and_circumference" in mapping
    assert mapping["radius_diameter_and_circumference"] == _p0_stable_int64(
        "radius_diameter_and_circumference"
    )


def _junyi_data_available() -> bool:
    if not JUNYI_CONFIG.exists():
        return False
    _dh2, p0_cfg, _ = load_configs(JUNYI_CONFIG)
    return all(
        p.exists()
        for p in (
            JUNYI_PARQUET,
            JUNYI_EXPERT,
            JUNYI_EXERCISE_TABLE,
            e_pre_export_path(p0_cfg, 0),
        )
    )


@pytest.mark.slow
@pytest.mark.skipif(not _junyi_data_available(), reason="P0 Junyi processed data not present")
def test_gt_crossval_junyi_fold0_integration():
    from dh2a_kt.eval.gt_crossval import build_gt_crossval_from_p0_exports

    dh2_cfg, p0_cfg, _ = load_configs(JUNYI_CONFIG)
    result = build_gt_crossval_from_p0_exports(
        dh2_cfg,
        p0_cfg,
        fold=0,
        p0_root=P0_ROOT,
    )
    assert result.n_expert_edges_matched > 0
    assert result.n_e_pre_edges > 0
    assert result.n_hyperedge_inferred_edges > 0
    assert result.alignment_report.get("alignment_rate", 0) >= 0.8
    e_row = result.e_pre_sweep.loc[result.e_pre_sweep["top_k"] == result.n_expert_edges_matched]
    assert not e_row.empty
    assert 0.0 <= float(e_row.iloc[0]["edge_f1"]) <= 1.0
