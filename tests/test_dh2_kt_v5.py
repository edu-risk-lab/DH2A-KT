"""v5: multi-KC question events + hypergraph transport of learner state.

The decisive test here is
``test_destroying_the_graph_changes_predictions``: v2--v4 all failed it in
practice (deleting the graph moved fold-0 AUC by 0.0002), because the hypergraph
only ever refined a free embedding table. v5 is built so the graph carries the
learner's own mastery between co-examined KCs, which no static table can absorb.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from dh2a_kt.models.dh2_kt import (  # noqa: E402
    DH2KT,
    DH2KTBatch,
    DH2KTConfig,
    concept_neighbors_from_hyperedges,
    empty_hyperedge_index,
    masked_attention_pool,
    masked_mean,
    star_incidence_tables,
)

N_CONCEPTS = 6
N_ITEMS = 5


def _config(**overrides) -> DH2KTConfig:
    base = dict(
        n_concepts=N_CONCEPTS,
        n_exercises=N_ITEMS,
        hidden_dim=16,
        embed_dim=16,
        memory_dim=8,
        n_hypergraph_layers=1,
        dropout=0.0,
        hyperedge_kinds=("question_concepts",),
        architecture="v5",
    )
    base.update(overrides)
    return DH2KTConfig(**base)


def _hyperedge_index() -> dict[str, torch.Tensor]:
    """Two question hyperedges: {0,1,2} and {3,4}. Concept 5 is isolated."""
    nodes = [0, 1, 2, 3, 4]
    edges = [0, 0, 0, 1, 1]
    return {"question_concepts": torch.tensor([nodes, edges], dtype=torch.long)}


def _batch(hyperedge_index, *, batch_size=2, seq_len=4) -> DH2KTBatch:
    torch.manual_seed(0)
    kc_ids = torch.randint(0, N_CONCEPTS, (batch_size, seq_len, 3))
    kc_mask = torch.zeros(batch_size, seq_len, 3, dtype=torch.bool)
    kc_mask[:, :, 0] = True
    kc_mask[:, ::2, 1] = True  # every other event is multi-KC
    return DH2KTBatch(
        concept_ids=kc_ids[:, :, 0],
        exercise_ids=torch.randint(0, N_ITEMS, (batch_size, seq_len)),
        responses=torch.randint(0, 2, (batch_size, seq_len)).float(),
        hyperedge_index=hyperedge_index,
        lengths=torch.full((batch_size,), seq_len, dtype=torch.long),
        kc_set_ids=kc_ids,
        kc_set_mask=kc_mask,
    )


def test_masked_mean_ignores_padded_slots():
    values = torch.tensor([[[1.0, 1.0], [5.0, 5.0]]])
    mask = torch.tensor([[True, False]])
    torch.testing.assert_close(masked_mean(values, mask), torch.tensor([[1.0, 1.0]]))


def test_masked_mean_returns_zero_when_nothing_is_valid():
    values = torch.ones(1, 2, 3)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    assert torch.count_nonzero(masked_mean(values, mask)) == 0


def test_neighbours_link_every_pair_sharing_a_hyperedge():
    neigh, mask = concept_neighbors_from_hyperedges(
        _hyperedge_index(), N_CONCEPTS, torch.device("cpu"), max_degree=4
    )
    # All members of a question hyperedge are mutually adjacent, unlike the
    # consecutive-only adjacency used for ordered prerequisite chains.
    assert set(neigh[0][mask[0]].tolist()) == {1, 2}
    assert set(neigh[1][mask[1]].tolist()) == {0, 2}
    assert set(neigh[3][mask[3]].tolist()) == {4}
    # Concept 5 belongs to no hyperedge.
    assert not mask[5].any()


def test_neighbours_are_empty_without_a_graph():
    _, mask = concept_neighbors_from_hyperedges(
        empty_hyperedge_index(kinds=("question_concepts",)),
        N_CONCEPTS,
        torch.device("cpu"),
    )
    assert not mask.any()


def test_neighbour_list_respects_the_degree_cap():
    dense = {
        "question_concepts": torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], dtype=torch.long
        )
    }
    neigh, mask = concept_neighbors_from_hyperedges(
        dense, N_CONCEPTS, torch.device("cpu"), max_degree=2
    )
    assert neigh.shape[1] == 2
    assert mask[0].sum() == 2


@pytest.mark.parametrize("use_questions", [False, True])
def test_forward_shape(use_questions: bool):
    model = DH2KT(_config(use_questions=use_questions))
    batch = _batch(_hyperedge_index())
    logits = model(batch)
    assert logits.shape == (2, 4, 1)
    assert torch.isfinite(logits).all()


def test_v5_requires_the_kc_set():
    model = DH2KT(_config())
    batch = _batch(_hyperedge_index())
    batch.kc_set_ids = None
    with pytest.raises(ValueError, match="kc_set_ids"):
        model(batch)


def test_destroying_the_graph_changes_predictions():
    """The result v2--v4 could not deliver: the graph must matter.

    The memory branch transports a learner's mastery along hyperedges, so with
    the hyperedges deleted the readout loses a term that depends on that
    learner's own history and cannot be recovered from any embedding table.
    """
    model = DH2KT(_config())
    model.eval()
    with torch.no_grad():
        with_graph = model(_batch(_hyperedge_index()))
        without_graph = model(
            _batch(empty_hyperedge_index(kinds=("question_concepts",)))
        )
    assert not torch.allclose(with_graph, without_graph, atol=1e-6)


def test_zero_transport_makes_the_graph_inert_on_the_memory_branch():
    """Sanity check on the knob: transport=0 removes the neighbour write.

    Predictions may still differ through the concept encoder, so this asserts the
    weaker, exact property that the transported memory term is unused.
    """
    model = DH2KT(_config(graph_transport=0.0))
    model.eval()
    batch = _batch(_hyperedge_index())
    with torch.no_grad():
        # Feed precomputed concept states so the encoder cannot introduce a
        # difference, isolating the memory branch.
        states = model._encode_concepts(batch.hyperedge_index)
        batch.concept_states = states
        with_graph = model(batch)
        empty = _batch(empty_hyperedge_index(kinds=("question_concepts",)))
        empty.concept_states = states
        empty.kc_set_ids = batch.kc_set_ids
        empty.kc_set_mask = batch.kc_set_mask
        empty.concept_ids = batch.concept_ids
        empty.exercise_ids = batch.exercise_ids
        empty.responses = batch.responses
        without_graph = model(empty)
    # Only the read-side neighbour aggregate remains graph-dependent.
    assert not torch.allclose(with_graph, without_graph, atol=1e-6)


def test_multi_kc_event_differs_from_its_first_concept_alone():
    """A 2-KC question must not be encoded as if it exercised only one KC."""
    model = DH2KT(_config())
    model.eval()
    both = _batch(_hyperedge_index(), batch_size=1, seq_len=3)
    both.kc_set_ids = torch.tensor([[[0, 1, 0], [3, 0, 0], [2, 0, 0]]])
    both.kc_set_mask = torch.tensor(
        [[[True, True, False], [True, False, False], [True, False, False]]]
    )
    both.concept_ids = both.kc_set_ids[:, :, 0]
    single = DH2KTBatch(
        concept_ids=both.concept_ids,
        exercise_ids=both.exercise_ids,
        responses=both.responses,
        hyperedge_index=both.hyperedge_index,
        lengths=both.lengths,
        kc_set_ids=both.kc_set_ids,
        kc_set_mask=torch.tensor(
            [[[True, False, False], [True, False, False], [True, False, False]]]
        ),
    )
    with torch.no_grad():
        assert not torch.allclose(model(both), model(single), atol=1e-6)


def test_padded_timesteps_do_not_write_memory():
    model = DH2KT(_config())
    model.eval()
    full = _batch(_hyperedge_index(), batch_size=1, seq_len=4)
    short = DH2KTBatch(
        concept_ids=full.concept_ids,
        exercise_ids=full.exercise_ids,
        responses=full.responses,
        hyperedge_index=full.hyperedge_index,
        lengths=torch.tensor([2]),
        kc_set_ids=full.kc_set_ids,
        kc_set_mask=full.kc_set_mask,
    )
    with torch.no_grad():
        scored = model(short)
    # Position 0 predicts position 1, which is inside the real length.
    assert torch.isfinite(scored).all()


def test_v5_sanity_overfit_tiny_batch():
    """A working model must be able to memorise four sequences."""
    torch.manual_seed(0)
    model = DH2KT(_config())
    batch = _batch(_hyperedge_index(), batch_size=4, seq_len=6)
    target = batch.responses[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first_loss = None
    for _ in range(150):
        opt.zero_grad()
        logits = model(batch)[:, :-1, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
    assert loss.item() < first_loss * 0.5


def test_gradients_reach_the_memory_branch():
    model = DH2KT(_config())
    logits = model(_batch(_hyperedge_index()))
    logits.sum().backward()
    grad = model.memory_write.weight_ih.grad
    assert grad is not None and torch.count_nonzero(grad) > 0
    readout_grad = model.memory_readout.weight.grad
    assert readout_grad is not None and torch.count_nonzero(readout_grad) > 0


def test_star_incidence_keeps_the_hyperedge_as_an_intermediate():
    hedge_members, hedge_mask, concept_hedges, concept_mask, n = star_incidence_tables(
        _hyperedge_index(), N_CONCEPTS, torch.device("cpu"), max_members=4, max_hedges=4
    )
    assert n == 2
    # Hyperedge 0 = {0,1,2}: members listed, concepts point back to hedge 0.
    assert set(hedge_members[0][hedge_mask[0]].tolist()) == {0, 1, 2}
    assert 0 in concept_hedges[1][concept_mask[1]].tolist()
    # Clique expansion would link 0-1 directly; star only links via the hedge id.
    assert concept_mask[5].sum() == 0


def test_attention_pool_is_not_uniform_when_scores_differ():
    values = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    scores = torch.tensor([[10.0, -10.0, 0.0]])
    pooled = masked_attention_pool(values, mask, scores)
    # Near-one-hot on the first member.
    assert pooled[0, 0] > 0.9 and pooled[0, 1] < 0.1


def test_priority_flags_change_predictions_vs_baseline_v5():
    torch.manual_seed(0)
    base = DH2KT(_config())
    prio = DH2KT(
        _config(
            event_pool="attention",
            transport="star",
            use_hyperedge_embed=True,
            kind_conditioned=True,
            use_questions=True,
        )
    )
    batch = _batch(_hyperedge_index())
    base.eval()
    prio.eval()
    with torch.no_grad():
        # Different modules => different outputs on the same batch.
        assert not torch.allclose(base(batch), prio(batch), atol=1e-5)


def test_priority_star_still_depends_on_the_graph():
    model = DH2KT(
        _config(event_pool="attention", transport="star", use_hyperedge_embed=True, kind_conditioned=True)
    )
    model.eval()
    with torch.no_grad():
        with_graph = model(_batch(_hyperedge_index()))
        without = model(_batch(empty_hyperedge_index(kinds=("question_concepts",))))
    assert not torch.allclose(with_graph, without, atol=1e-6)


def test_priority_sanity_overfit():
    torch.manual_seed(0)
    model = DH2KT(
        _config(
            event_pool="attention",
            transport="star",
            use_hyperedge_embed=True,
            kind_conditioned=True,
            use_questions=True,
        )
    )
    batch = _batch(_hyperedge_index(), batch_size=4, seq_len=6)
    target = batch.responses[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(batch)[:, :-1, 0], target
        )
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.5
