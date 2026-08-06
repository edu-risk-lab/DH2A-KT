"""M6 gate: manipulation check with real eval_auc_fn."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dh2a_kt.eval.manipulation_check import run_manipulation_check
from dh2a_kt.hyperedge.construction import Hyperedge, resolve_concept_prerequisite_hyperedges
from dh2a_kt.hyperedge.indexing import destroy_hyperedges
from dh2a_kt.hyperedge.p0_inputs import REPO_ROOT, get_fold_splits, load_configs, load_e_pre, load_interactions_with_ids
from dh2a_kt.eval.tier1_eval import (
    make_eval_auc_fn,
    run_manipulation_check_for_fold,
    write_manipulation_check_result,
)
from dh2a_kt.train.tier1 import ConceptPrerequisiteSpec, TrainingBudget, resolve_training_budget, train_fold

pytest.importorskip("torch")

XES3G5M_CONFIG = REPO_ROOT / "configs" / "xes3g5m.yaml"
P0_PARQUET = REPO_ROOT / "external" / "p0_leakage_audit" / "data" / "processed" / "xes3g5m.parquet"


def _toy_hyperedges(n: int = 20) -> list[Hyperedge]:
    return [
        Hyperedge(
            hyperedge_id=f"h{i}",
            kind="concept_prerequisite",
            fold=0,
            train_only=True,
            members=[("concept", i), ("concept", i + 1)],
        )
        for i in range(n)
    ]


def test_run_manipulation_check_records_verdict():
    seen: list[int] = []

    def eval_fn(hyperedges: list[Hyperedge]) -> float:
        seen.append(len(hyperedges))
        return 0.80 if len(hyperedges) >= 10 else 0.76

    result = run_manipulation_check(
        _toy_hyperedges(20),
        eval_auc_fn=eval_fn,
        p=0.90,
        operator="node_drop",
        seed=42,
    )
    assert len(seen) == 2
    assert isinstance(result.verdict, str)
    assert result.passes_manipulation_check == (result.auc_drop > 0.003)
    assert result.auc_clean >= result.auc_destroyed or result.auc_drop <= 0


def _synthetic_logs(n_users: int = 12, seq_len: int = 10) -> pd.DataFrame:
    rows = []
    for u in range(n_users):
        for t in range(seq_len):
            rows.append(
                {
                    "user_id": u,
                    "item_id": (u + t) % 5,
                    "kc_id": t % 4,
                    "timestamp": u * 1000 + t,
                    "correct": (u + t) % 2,
                }
            )
    return pd.DataFrame(rows)


def test_make_eval_auc_fn_with_trained_model():
    train_df = _synthetic_logs(n_users=12, seq_len=10)
    eval_df = _synthetic_logs(n_users=8, seq_len=10)
    e_pre = pd.DataFrame({"src_kc": [0, 1, 2], "dst_kc": [1, 2, 3], "weight": [1.0, 1.0, 1.0]})
    budget = TrainingBudget(
        reference_model="gkt", batch_size=4, epochs=12, lr=1e-2, max_seq_len=10, matched_p0=True
    )
    spec = ConceptPrerequisiteSpec(fold=0, source="chain", min_chain_len=3, max_chain_len=8)
    trained = train_fold(
        train_df,
        eval_df,
        e_pre,
        budget,
        device="cpu",
        hidden_dim=16,
        n_hypergraph_layers=2,
        graph_dropout=0.15,
        graph_sensitivity_weight=0.5,
        hyperedge_spec=spec,
    )
    hyperedges = resolve_concept_prerequisite_hyperedges(e_pre, fold=0, source="chain")
    eval_fn = make_eval_auc_fn(trained)
    auc_clean = eval_fn(hyperedges)
    auc_destroyed = eval_fn(destroy_hyperedges(hyperedges, p=0.9, seed=42))
    assert np.isfinite(auc_clean)
    assert np.isfinite(auc_destroyed)


def _real_data_available() -> bool:
    if not XES3G5M_CONFIG.exists() or not P0_PARQUET.exists():
        return False
    from dh2a_kt.hyperedge.p0_inputs import e_pre_export_path

    _dh2, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    return e_pre_export_path(p0_cfg, 0).exists()


@pytest.mark.slow
@pytest.mark.skipif(not _real_data_available(), reason="P0 XES3G5M processed data not present")
def test_manipulation_check_integration_xes3g5m_fold0(tmp_path: Path):
    dh2_cfg, p0_cfg, _ = load_configs(XES3G5M_CONFIG)
    interactions = load_interactions_with_ids(p0_cfg)
    splits = get_fold_splits(interactions, p0_cfg, 0)
    eval_df = pd.concat([splits["valid"], splits["test"]], ignore_index=True)
    e_pre = load_e_pre(p0_cfg, 0)
    budget = replace(resolve_training_budget(p0_cfg, reference_model="gkt", max_seq_len=50), epochs=1)

    result = run_manipulation_check_for_fold(
        splits["train"],
        eval_df,
        e_pre,
        fold=0,
        budget=budget,
        device="cpu",
        hidden_dim=32,
        n_hypergraph_layers=2,
        graph_dropout=0.15,
        max_users=32,
        p=float(dh2_cfg["manipulation_check"]["p"]),
        operator=dh2_cfg["manipulation_check"]["operator"],
        seed=int(dh2_cfg["manipulation_check"]["seed"]),
    )

    out = tmp_path / "manipulation_check.json"
    write_manipulation_check_result(result, dataset="xes3g5m", fold=0, output_path=out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert "passes_manipulation_check" in payload
    assert "verdict" in payload
    assert np.isfinite(payload["auc_clean"])
    assert np.isfinite(payload["auc_destroyed"])
    assert isinstance(payload["passes_manipulation_check"], bool)
    assert len(payload["verdict"]) > 0
