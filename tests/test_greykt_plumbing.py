"""GreyKT plumbing that must run without torch_geometric."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from dh2a_kt.config_helpers import concept_prerequisite_spec_from_config
from dh2a_kt.train.greykt import assert_black_auc_matches_native, enforce_cb_diagnostic

REPO = Path(__file__).resolve().parents[1]


def test_xes3g5m_greykt_block_and_max_hops_inherit_chain_len():
    cfg = yaml.safe_load((REPO / "configs" / "xes3g5m.yaml").read_text(encoding="utf-8"))
    g = cfg["greykt"]
    for key in ("prior_strength", "kappa_b", "hop_decay", "recency_decay", "kappa_r", "mc_samples"):
        assert key in g
    assert "max_hops" not in g  # inherit ell_max from chain hyperedges
    spec = concept_prerequisite_spec_from_config(cfg, fold=0)
    assert spec.max_chain_len == 8


def test_enforce_cb_diagnostic_gate(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        enforce_cb_diagnostic(missing)
    assert enforce_cb_diagnostic(missing, force=True) is None

    supported = tmp_path / "ok.json"
    supported.write_text(json.dumps({"verdict": "SUPPORTED"}), encoding="utf-8")
    assert enforce_cb_diagnostic(supported) == "SUPPORTED"

    blocked = tmp_path / "no.json"
    blocked.write_text(json.dumps({"verdict": "NOT_SUPPORTED"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="NOT_SUPPORTED"):
        enforce_cb_diagnostic(blocked)
    assert enforce_cb_diagnostic(blocked, force=True) == "NOT_SUPPORTED"


@pytest.mark.parametrize(
    "rel",
    [
        "scripts/21_diagnose_black_confidence.py",
        "scripts/22_run_greykt.py",
        "dh2a_kt/train/greykt.py",
        "dh2a_kt/train/greykt_inputs.py",
        "dh2a_kt/diagnostics/mc_dropout.py",
        "dh2a_kt/diagnostics/black_confidence.py",
    ],
)
def test_greykt_sources_parse(rel: str):
    src = (REPO / rel).read_text(encoding="utf-8")
    ast.parse(src)


def test_wrap_black_auc_audit_rejects_chance_level_disagreement():
    assert_black_auc_matches_native(0.750, 0.751)
    with pytest.raises(RuntimeError, match="disagrees"):
        assert_black_auc_matches_native(0.535, 0.750)
    with pytest.raises(RuntimeError, match="not finite"):
        assert_black_auc_matches_native(float("nan"), 0.750)


def test_script_22_selects_on_valid_and_audits_native_black():
    src = (REPO / "scripts" / "22_run_greykt.py").read_text(encoding="utf-8")
    assert "valid_loader=valid_cached" in src
    assert "assert_black_auc_matches_native" in src
    assert "native_dh2kt_auc" in src
    assert '"valid"' in src and '"test"' in src and '"valid+test"' in src
    train_src = (REPO / "dh2a_kt" / "train" / "greykt.py").read_text(encoding="utf-8")
    assert "valid_nll" in train_src
    assert "GreyKT(cfg, black_box=bb)" in train_src
    assert "Do not use this run for a paper table" in train_src
