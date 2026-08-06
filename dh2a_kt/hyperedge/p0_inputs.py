"""Load P0-preprocessed inputs for hyperedge construction (M1).

Paths in P0 YAML configs are relative to ``external/p0_leakage_audit/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from dh2a_kt.p0_bridge import learner_based_folds

REPO_ROOT = Path(__file__).resolve().parents[2]
P0_ROOT = REPO_ROOT / "external" / "p0_leakage_audit"


def load_configs(dh2_config_path: Path) -> tuple[dict, dict, Path]:
    """Return (dh2_cfg, p0_cfg, p0_config_path)."""
    dh2_config_path = dh2_config_path.resolve()
    dh2_cfg = yaml.safe_load(dh2_config_path.read_text(encoding="utf-8"))
    p0_config_path = REPO_ROOT / dh2_cfg["p0_config"]
    if not p0_config_path.exists():
        raise FileNotFoundError(p0_config_path)
    p0_cfg = yaml.safe_load(p0_config_path.read_text(encoding="utf-8"))
    return dh2_cfg, p0_cfg, p0_config_path


def p0_path(p0_cfg: dict, key: str) -> Path:
    """Resolve a path field from a P0 config against the P0 repo root."""
    rel = p0_cfg[key]
    return (P0_ROOT / rel).resolve()


def e_pre_export_path(p0_cfg: dict, fold: int) -> Path:
    dataset = p0_cfg["dataset"]
    return P0_ROOT / "data" / "processed" / dataset / f"fold_{fold}" / "e_pre_train_only.csv"


def load_e_pre(p0_cfg: dict, fold: int) -> pd.DataFrame:
    path = e_pre_export_path(p0_cfg, fold)
    if not path.exists():
        raise FileNotFoundError(
            f"P0 export missing: {path}\n"
            "Run graph_builder from external/p0_leakage_audit first "
            "(see README section 4)."
        )
    df = pd.read_csv(path)
    for col in ("src_kc", "dst_kc"):
        if col not in df.columns:
            raise ValueError(f"{path} missing column {col!r}")
    return df


def load_interactions_with_ids(p0_cfg: dict) -> pd.DataFrame:
    from src.io_utils import load_interactions  # noqa: WPS433 — P0 bridge path

    path = p0_path(p0_cfg, "processed_path")
    if not path.exists():
        raise FileNotFoundError(path)
    df = load_interactions(path)
    df = df.reset_index(drop=True)
    df["interaction_id"] = df.index.astype("int64")
    return df


def get_fold_splits(interactions: pd.DataFrame, p0_cfg: dict, fold: int) -> dict[str, pd.DataFrame]:
    ratios = tuple(p0_cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    split_cfg = p0_cfg.get("split", {})
    for f, _seed, splits in learner_based_folds(interactions, ratios, split_cfg):
        if f == fold:
            return splits
    raise ValueError(f"fold {fold} not found (n_folds={split_cfg.get('n_folds', 1)})")


def held_out_interaction_ids(splits: dict[str, pd.DataFrame]) -> set[int]:
    ids: set[int] = set()
    for name in ("valid", "test"):
        part = splits.get(name)
        if part is not None and not part.empty:
            ids.update(part["interaction_id"].astype(int).tolist())
    return ids


def build_concept_member_interaction_map(
    train_df: pd.DataFrame,
) -> dict[tuple[str, int], set[int]]:
    mapping: dict[tuple[str, int], set[int]] = {}
    for kc, iids in train_df.groupby("kc_id")["interaction_id"]:
        mapping[("concept", int(kc))] = set(iids.astype(int).tolist())
    return mapping
