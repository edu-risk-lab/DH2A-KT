"""Concept-prerequisite hyperedge sources: chain vs pairwise vs neighborhood.

On XES3G5M fold 0 the ``chain`` source turns E_pre's 1,163 edges into 446,305
hyperedges (76% at the length cap, 58% permutations of an existing member set,
one hub concept in 29% of them). These tests pin the sparser alternatives.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dh2a_kt.hyperedge.construction import (
    hyperedges_from_e_pre_neighborhood,
    resolve_concept_prerequisite_hyperedges,
)


def _e_pre() -> pd.DataFrame:
    # 1 -> 3, 2 -> 3, 3 -> 4, plus an isolated pair 5 -> 6.
    return pd.DataFrame(
        {
            "src_kc": [1, 2, 3, 5],
            "dst_kc": [3, 3, 4, 6],
            "weight": [0.2, 0.4, 0.5, 0.1],
        }
    )


def test_neighborhood_builds_one_hyperedge_per_target_concept():
    hyperedges = hyperedges_from_e_pre_neighborhood(_e_pre(), fold=0)
    by_target = {he.members[0][1]: he for he in hyperedges}

    assert sorted(by_target) == [3, 4, 6]
    assert {m[1] for m in by_target[3].members} == {3, 1, 2}
    assert {m[1] for m in by_target[4].members} == {4, 3}
    assert by_target[3].provenance["n_prerequisites"] == 2
    assert by_target[3].provenance["mean_weight"] == pytest.approx(0.3)
    assert all(he.kind == "concept_prerequisite" and he.train_only for he in hyperedges)


def test_neighborhood_is_far_sparser_than_chain():
    e_pre = _e_pre()
    chain = resolve_concept_prerequisite_hyperedges(
        e_pre, fold=0, source="chain", min_chain_len=2, max_chain_len=8
    )
    neighborhood = resolve_concept_prerequisite_hyperedges(
        e_pre, fold=0, source="neighborhood"
    )
    pairwise = resolve_concept_prerequisite_hyperedges(e_pre, fold=0, source="pairwise")

    assert len(pairwise) == len(e_pre)
    assert len(neighborhood) <= len(pairwise) <= len(chain)

    def incidence(hyperedges) -> int:
        return sum(len(he.members) for he in hyperedges)

    assert incidence(neighborhood) < incidence(chain)


def test_neighborhood_drops_self_loops_and_targets_without_prerequisites():
    e_pre = pd.DataFrame({"src_kc": [7, 8], "dst_kc": [7, 9], "weight": [1.0, 1.0]})
    hyperedges = hyperedges_from_e_pre_neighborhood(e_pre, fold=0)
    targets = [he.members[0][1] for he in hyperedges]
    assert targets == [9]  # concept 7 only had a self edge


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="neighborhood"):
        resolve_concept_prerequisite_hyperedges(_e_pre(), fold=0, source="bogus")


def test_question_hyperedges_carry_the_kc_set_of_each_multi_kc_question():
    from dh2a_kt.hyperedge.construction import build_question_hyperedges

    hyperedges = build_question_hyperedges({10: [1, 2], 11: [3], 12: [2, 4, 5]}, fold=0)

    # Question 11 exercises a single KC, so it yields no hyperedge.
    assert len(hyperedges) == 2
    by_item = {he.provenance["item_id"]: he for he in hyperedges}
    assert sorted(by_item) == [10, 12]
    assert {m[1] for m in by_item[10].members} == {1, 2}
    assert {m[1] for m in by_item[12].members} == {2, 4, 5}
    assert all(he.kind == "question_concepts" for he in hyperedges)
    assert all(m[0] == "concept" for he in hyperedges for m in he.members)


def test_question_hyperedges_are_not_train_only():
    from dh2a_kt.hyperedge.construction import build_question_hyperedges

    # A question's KC set is published metadata, not inferred from outcomes, so
    # it must not be gated behind the train-only selection used for sessions.
    hyperedges = build_question_hyperedges({10: [1, 2]}, fold=0)
    assert hyperedges[0].train_only is False


def test_question_hyperedges_deduplicate_and_sort_members():
    from dh2a_kt.hyperedge.construction import build_question_hyperedges

    hyperedges = build_question_hyperedges({10: [5, 1, 5, 1]}, fold=0)
    assert [m[1] for m in hyperedges[0].members] == [1, 5]


def test_question_hyperedges_are_far_sparser_than_chains():
    from dh2a_kt.hyperedge.construction import build_question_hyperedges

    # 939 of 7,121 XES3G5M questions are multi-KC, against 446,305 chains.
    sets = {item: [item % 7, (item * 3) % 7] for item in range(500)}
    hyperedges = build_question_hyperedges(sets, fold=0)
    assert len(hyperedges) < 500
    assert max(len(he.members) for he in hyperedges) <= 2
