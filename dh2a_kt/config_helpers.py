"""Parse DH2A-KT YAML config fragments shared by training/eval drivers."""

from __future__ import annotations

from dh2a_kt.train.tier1 import ConceptPrerequisiteSpec


def concept_prerequisite_spec_from_config(dh2_cfg: dict, *, fold: int) -> ConceptPrerequisiteSpec:
    he_cfg = dh2_cfg.get("hyperedge", {}).get("concept_prerequisite", {})
    source = str(he_cfg.get("source", "chain"))
    if source == "p0_e_pre":
        source = "chain"
    return ConceptPrerequisiteSpec(
        fold=fold,
        source=source,
        min_chain_len=int(he_cfg.get("min_chain_len", 3)),
        max_chain_len=int(he_cfg.get("max_chain_len", 8)),
    )
