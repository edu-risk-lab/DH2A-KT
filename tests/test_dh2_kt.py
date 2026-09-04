"""M3 gate tests for dh2a_kt/models/dh2_kt.py::DH2KT.forward()."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from dh2a_kt.models.dh2_kt import DH2KT, DH2KTBatch, DH2KTConfig, empty_hyperedge_index


def _toy_hyperedge_index() -> dict[str, torch.Tensor]:
    # Concepts {0,1,2} in hyperedge 0; {1,2,3} in hyperedge 1.
    return {
        "concept_prerequisite": torch.tensor(
            [[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]],
            dtype=torch.long,
        ),
    }


def _make_toy_batch(
    batch_size: int = 4,
    seq_len: int = 5,
    *,
    n_concepts: int = 8,
    n_exercises: int = 6,
) -> DH2KTBatch:
    return DH2KTBatch(
        concept_ids=torch.randint(0, n_concepts, (batch_size, seq_len)),
        exercise_ids=torch.randint(0, n_exercises, (batch_size, seq_len)),
        responses=torch.randint(0, 2, (batch_size, seq_len)).float(),
        hyperedge_index=_toy_hyperedge_index(),
    )


def test_dh2kt_forward_output_shape():
    config = DH2KTConfig(n_concepts=8, n_exercises=6, hidden_dim=32, embed_dim=32)
    model = DH2KT(config)
    batch = _make_toy_batch(batch_size=4, seq_len=5, n_concepts=8, n_exercises=6)
    logits = model(batch)
    assert logits.shape == (4, 5, 1)


def test_dh2kt_sanity_overfit_tiny_batch():
    torch.manual_seed(0)
    config = DH2KTConfig(
        n_concepts=4,
        n_exercises=4,
        hidden_dim=32,
        embed_dim=32,
        n_hypergraph_layers=1,
        dropout=0.0,
    )
    model = DH2KT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    concept_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]] * 4)
    exercise_ids = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0]] * 4)
    labels = (concept_ids % 2).float()
    responses = torch.zeros_like(labels)
    responses[:, 1:] = labels[:, :-1]

    batch = DH2KTBatch(
        concept_ids=concept_ids,
        exercise_ids=exercise_ids,
        responses=responses,
        hyperedge_index={
            "concept_prerequisite": torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=torch.long),
        },
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets = labels[:, 1:]
    for _ in range(400):
        optimizer.zero_grad()
        logits = model(batch)[:, :-1, 0]
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(model(batch)[:, :-1, 0], targets).item()
    assert final_loss < 0.05


def test_dh2kt_forward_differs_with_vs_without_graph():
    """Graph path must change predictions when hyperedges are present vs empty."""
    torch.manual_seed(1)
    config = DH2KTConfig(
        n_concepts=8,
        n_exercises=6,
        hidden_dim=32,
        embed_dim=32,
        n_hypergraph_layers=2,
        dropout=0.0,
    )
    model = DH2KT(config)
    model.eval()

    batch_with_graph = _make_toy_batch(batch_size=2, seq_len=4)
    batch_no_graph = DH2KTBatch(
        concept_ids=batch_with_graph.concept_ids,
        exercise_ids=batch_with_graph.exercise_ids,
        responses=batch_with_graph.responses,
        hyperedge_index=empty_hyperedge_index(),
    )

    with torch.no_grad():
        logits_graph = model(batch_with_graph)
        logits_empty = model(batch_no_graph)

    diff = (logits_graph - logits_empty).abs().max().item()
    assert diff > 1e-4, f"expected graph to affect logits, max diff={diff}"


def _v3_config(**kwargs) -> DH2KTConfig:
    defaults = dict(
        n_concepts=8,
        n_exercises=6,
        hidden_dim=32,
        embed_dim=32,
        n_hypergraph_layers=1,
        dropout=0.0,
        hyperedge_kinds=("concept_prerequisite",),
        architecture="v3",
        diffusion_alpha=0.5,
    )
    defaults.update(kwargs)
    return DH2KTConfig(**defaults)


def test_v3_forward_output_shape():
    model = DH2KT(_v3_config())
    batch = _make_toy_batch(batch_size=4, seq_len=5)
    logits = model(batch)
    assert logits.shape == (4, 5, 1)
    assert model.query_head is not None
    assert model.kind_mix is None


def test_v3_next_concept_changes_logit_at_previous_step():
    """Head at t must see c_{t+1}: swapping the last concept moves logit[:, T-2]."""
    torch.manual_seed(0)
    model = DH2KT(_v3_config())
    model.eval()
    batch = _make_toy_batch(batch_size=3, seq_len=6)
    alt_ids = batch.concept_ids.clone()
    alt_ids[:, -1] = (alt_ids[:, -1] + 3) % 8
    alt = DH2KTBatch(
        concept_ids=alt_ids,
        exercise_ids=batch.exercise_ids,
        responses=batch.responses,
        hyperedge_index=batch.hyperedge_index,
    )
    with torch.no_grad():
        base = model(batch)
        swapped = model(alt)
    assert (base[:, -2] - swapped[:, -2]).abs().max().item() > 1e-5
    assert (base[:, :-2] - swapped[:, :-2]).abs().max().item() < 1e-5


def test_v3_forward_differs_with_vs_without_graph():
    torch.manual_seed(2)
    model = DH2KT(_v3_config(n_hypergraph_layers=2))
    model.eval()
    batch_with_graph = _make_toy_batch(batch_size=2, seq_len=4)
    batch_no_graph = DH2KTBatch(
        concept_ids=batch_with_graph.concept_ids,
        exercise_ids=batch_with_graph.exercise_ids,
        responses=batch_with_graph.responses,
        hyperedge_index=empty_hyperedge_index(),
    )
    with torch.no_grad():
        logits_graph = model(batch_with_graph)
        logits_empty = model(batch_no_graph)
    diff = (logits_graph - logits_empty).abs().max().item()
    assert diff > 1e-4, f"expected graph to affect v3 logits, max diff={diff}"


def test_v3_sanity_overfit_tiny_batch():
    torch.manual_seed(0)
    model = DH2KT(_v3_config(n_concepts=4, n_exercises=4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    concept_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]] * 4)
    exercise_ids = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0]] * 4)
    labels = (concept_ids % 2).float()
    responses = torch.zeros_like(labels)
    responses[:, 1:] = labels[:, :-1]
    batch = DH2KTBatch(
        concept_ids=concept_ids,
        exercise_ids=exercise_ids,
        responses=responses,
        hyperedge_index={
            "concept_prerequisite": torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=torch.long),
        },
        lengths=torch.full((4,), 8, dtype=torch.long),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets = labels[:, 1:]
    for _ in range(400):
        optimizer.zero_grad()
        loss = loss_fn(model(batch)[:, :-1, 0], targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = loss_fn(model(batch)[:, :-1, 0], targets).item()
    assert final_loss < 0.05


def test_v3_heterogeneous_kind_mix_and_session_edges():
    config = _v3_config(hyperedge_kinds=("concept_prerequisite", "session"))
    model = DH2KT(config)
    assert model.kind_mix is not None
    batch = DH2KTBatch(
        concept_ids=torch.randint(0, 8, (2, 5)),
        exercise_ids=torch.randint(0, 6, (2, 5)),
        responses=torch.randint(0, 2, (2, 5)).float(),
        hyperedge_index={
            "concept_prerequisite": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            "session": torch.tensor([[2, 3, 4], [0, 0, 0]], dtype=torch.long),
        },
    )
    logits = model(batch)
    assert logits.shape == (2, 5, 1)


def test_normalized_concept_adjacency_is_row_stochastic():
    from dh2a_kt.models.dh2_kt import normalized_concept_adjacency

    index = _toy_hyperedge_index()
    adj = normalized_concept_adjacency(index, n_concepts=8, device=torch.device("cpu"))
    assert adj.shape == (8, 8)
    row_sums = adj.sum(dim=1)
    connected = row_sums > 0
    assert torch.allclose(row_sums[connected], torch.ones_like(row_sums[connected]))


def _v4_config(**kwargs) -> DH2KTConfig:
    defaults = dict(
        n_concepts=8,
        n_exercises=6,
        hidden_dim=32,
        embed_dim=32,
        n_hypergraph_layers=1,
        dropout=0.0,
        hyperedge_kinds=("concept_prerequisite",),
        architecture="v4",
    )
    defaults.update(kwargs)
    return DH2KTConfig(**defaults)


def test_v4_forward_output_shape():
    model = DH2KT(_v4_config())
    logits = model(_make_toy_batch(batch_size=4, seq_len=5))
    assert logits.shape == (4, 5, 1)


def test_v4_next_concept_changes_logit_at_previous_step():
    torch.manual_seed(0)
    model = DH2KT(_v4_config())
    model.eval()
    batch = _make_toy_batch(batch_size=3, seq_len=6)
    alt_ids = batch.concept_ids.clone()
    alt_ids[:, -1] = (alt_ids[:, -1] + 3) % 8
    alt = DH2KTBatch(
        concept_ids=alt_ids,
        exercise_ids=batch.exercise_ids,
        responses=batch.responses,
        hyperedge_index=batch.hyperedge_index,
    )
    with torch.no_grad():
        base = model(batch)
        swapped = model(alt)
    assert (base[:, -2] - swapped[:, -2]).abs().max().item() > 1e-5
    assert (base[:, :-2] - swapped[:, :-2]).abs().max().item() < 1e-5


def test_v4_uses_response_of_the_same_step():
    """Flipping r_t must move logit[:, t]; v2/v3 alignment could not see it."""
    torch.manual_seed(0)
    model = DH2KT(_v4_config())
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    flipped = batch.responses.clone()
    flipped[:, 2] = 1.0 - flipped[:, 2]
    alt = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=flipped,
        hyperedge_index=batch.hyperedge_index,
    )
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, 2] - changed[:, 2]).abs().max().item() > 1e-5
    assert (base[:, :2] - changed[:, :2]).abs().max().item() < 1e-5


def test_v4_keeps_identity_of_concepts_outside_any_hyperedge():
    """On XES3G5M 445/865 concepts have no hyperedge; they must not become zero."""
    model = DH2KT(_v4_config())
    model.eval()
    with torch.no_grad():
        states = model._encode_concepts(_toy_hyperedge_index())
    # Concepts 4..7 appear in no hyperedge of the toy index.
    absent = states[4:]
    assert absent.abs().sum(dim=-1).min().item() > 0.0
    assert torch.allclose(absent, model.concept_embed.weight[4:])


def test_v4_forward_differs_with_vs_without_graph():
    torch.manual_seed(2)
    model = DH2KT(_v4_config(n_hypergraph_layers=2))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=4)
    no_graph = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=batch.responses,
        hyperedge_index=empty_hyperedge_index(),
    )
    with torch.no_grad():
        diff = (model(batch) - model(no_graph)).abs().max().item()
    assert diff > 1e-4, f"expected graph to affect v4 logits, max diff={diff}"


@pytest.mark.parametrize("use_questions", [False, True])
def test_v4_sanity_overfit_tiny_batch(use_questions: bool):
    torch.manual_seed(0)
    model = DH2KT(_v4_config(n_concepts=4, n_exercises=4, use_questions=use_questions))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    concept_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]] * 4)
    exercise_ids = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0]] * 4)
    labels = (concept_ids % 2).float()
    batch = DH2KTBatch(
        concept_ids=concept_ids,
        exercise_ids=exercise_ids,
        responses=labels,  # v4 alignment: response of the same step
        hyperedge_index={
            "concept_prerequisite": torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=torch.long),
        },
        lengths=torch.full((4,), 8, dtype=torch.long),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets = labels[:, 1:]
    for _ in range(400):
        optimizer.zero_grad()
        loss = loss_fn(model(batch)[:, :-1, 0], targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = loss_fn(model(batch)[:, :-1, 0], targets).item()
    assert final_loss < 0.05


def test_v4_recap_attention_is_causal():
    """Future responses must not move logits at earlier positions (E2 leak check)."""
    torch.manual_seed(0)
    model = DH2KT(_v4_config(recap_attention=True, use_questions=True))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=8)
    cut = 3
    flipped = batch.responses.clone()
    flipped[:, cut + 1 :] = 1.0 - flipped[:, cut + 1 :]
    alt = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=flipped,
        hyperedge_index=batch.hyperedge_index,
        lengths=batch.lengths,
    )
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, : cut + 1] - changed[:, : cut + 1]).abs().max().item() < 1e-5
    # Future position should move (sanity: the flip is not a no-op overall).
    assert (base[:, cut + 1 :] - changed[:, cut + 1 :]).abs().max().item() > 1e-5


def test_v4_question_kc_agg_changes_logits():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(use_questions=True, question_kc_agg=True))
    # Item 0 → concepts {0,1}; item 1 → {2}
    model.set_question_kc_table(
        torch.tensor([[0, 1], [2, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.long),
        torch.tensor(
            [[True, True], [True, False], [False, False], [False, False], [False, False], [False, False]]
        ),
    )
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    plain = DH2KT(_v4_config(use_questions=True))
    plain.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("exercise_kc_")},
        strict=False,
    )
    plain.eval()
    with torch.no_grad():
        diff = (model(batch) - plain(batch)).abs().max().item()
    assert diff > 1e-4


def test_v4_question_graph_incidence_changes_logits():
    """Flipping a Q–KC membership must move logits that use that question."""
    torch.manual_seed(0)
    model = DH2KT(_v4_config(use_questions=True, question_graph=True))
    n_ex, n_c = 6, 8
    A = torch.zeros(n_ex, n_c)
    A[0, 0] = 1.0
    A[0, 1] = 1.0
    A[1, 2] = 1.0
    model.set_question_kc_incidence(A)
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    # Force item 0 into the sequence so the flipped row is scored.
    batch.exercise_ids[:, :] = 0
    with torch.no_grad():
        base = model(batch)
    A2 = A.clone()
    A2[0, 2] = 1.0  # add a third KC for item 0
    model.set_question_kc_incidence(A2)
    with torch.no_grad():
        changed = model(batch)
    assert (base - changed).abs().max().item() > 1e-4


def test_v4_question_graph_ablation_differs_from_full():
    """Zeroed incidence (empty agg) differs from a connected incidence."""
    torch.manual_seed(1)
    model = DH2KT(_v4_config(use_questions=True, question_graph=True))
    n_ex, n_c = 6, 8
    A = torch.zeros(n_ex, n_c)
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    model.set_question_kc_incidence(A)
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    with torch.no_grad():
        full = model(batch)
    model.set_question_kc_incidence(torch.zeros_like(A))
    with torch.no_grad():
        empty = model(batch)
    assert (full - empty).abs().max().item() > 1e-4
    # The legacy hard-off arm changes capacity and is not a valid attribution twin.
    off = DH2KT(_v4_config(use_questions=True, question_graph=False))
    assert not hasattr(off, "question_embed")


def test_v4_zero_incidence_twin_has_identical_trainable_capacity():
    observed = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            question_incidence_control="observed",
        )
    )
    zero = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            question_incidence_control="zero",
        )
    )
    observed_shapes = {
        name: tuple(parameter.shape) for name, parameter in observed.named_parameters()
    }
    zero_shapes = {
        name: tuple(parameter.shape) for name, parameter in zero.named_parameters()
    }
    assert observed_shapes == zero_shapes
    assert sum(p.numel() for p in observed.parameters()) == sum(
        p.numel() for p in zero.parameters()
    )
    assert hasattr(observed, "question_embed")
    assert hasattr(zero, "question_embed")


def test_v4_question_graph_no_future_leak():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(use_questions=True, question_graph=True))
    n_ex, n_c = 6, 8
    A = torch.zeros(n_ex, n_c)
    for i in range(min(n_ex, n_c)):
        A[i, i] = 1.0
    model.set_question_kc_incidence(A)
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    cut = 2
    flipped = batch.responses.clone()
    flipped[:, cut + 1 :] = 1.0 - flipped[:, cut + 1 :]
    alt = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=flipped,
        hyperedge_index=batch.hyperedge_index,
        lengths=batch.lengths,
    )
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, : cut + 1] - changed[:, : cut + 1]).abs().max().item() < 1e-5
    assert (base[:, cut + 1 :] - changed[:, cut + 1 :]).abs().max().item() > 1e-5


def test_v4_question_hypergraph_empty_incidence_differs():
    """Zeroing multi-KC question hyperedges must move logits (graph is live)."""
    torch.manual_seed(0)
    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            question_hypergraph=True,
            hyperedge_kinds=("question_concepts",),
        )
    )
    A = torch.zeros(6, 8)
    A[0, 0] = 1.0
    A[0, 1] = 1.0
    model.set_question_kc_incidence(A)
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    batch.concept_ids[:, :] = 0
    batch.exercise_ids[:, :] = 0
    batch.hyperedge_index = {
        "question_concepts": torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    }
    with torch.no_grad():
        live = model(batch)
    batch.hyperedge_index = {
        "question_concepts": torch.empty((2, 0), dtype=torch.long),
    }
    with torch.no_grad():
        empty = model(batch)
    assert (live - empty).abs().max().item() > 1e-4
    off = DH2KT(_v4_config(use_questions=True, question_graph=True))
    assert not hasattr(off, "question_hconv")


def test_v4_question_hypergraph_membership_changes_logits():
    torch.manual_seed(1)
    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            question_hypergraph=True,
            hyperedge_kinds=("question_concepts",),
        )
    )
    model.set_question_kc_incidence(torch.eye(6, 8))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    batch.concept_ids[:, :] = 0
    pair = {
        "question_concepts": torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    }
    triple = {
        "question_concepts": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
    }
    batch.hyperedge_index = pair
    with torch.no_grad():
        a = model(batch)
    batch.hyperedge_index = triple
    with torch.no_grad():
        b = model(batch)
    assert (a - b).abs().max().item() > 1e-4


def test_v4_question_hypergraph_no_future_leak():
    torch.manual_seed(0)
    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            question_hypergraph=True,
            hyperedge_kinds=("question_concepts",),
        )
    )
    model.set_question_kc_incidence(torch.eye(6, 8))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    batch.hyperedge_index = {
        "question_concepts": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
    }
    cut = 2
    flipped = batch.responses.clone()
    flipped[:, cut + 1 :] = 1.0 - flipped[:, cut + 1 :]
    alt = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=flipped,
        hyperedge_index=batch.hyperedge_index,
        lengths=batch.lengths,
    )
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, : cut + 1] - changed[:, : cut + 1]).abs().max().item() < 1e-5
    assert (base[:, cut + 1 :] - changed[:, cut + 1 :]).abs().max().item() > 1e-5


def test_v4_hint_hypergraph_empty_incidence_differs():
    torch.manual_seed(0)
    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            hint_hypergraph=True,
        )
    )
    A = torch.eye(6, 8)
    model.set_question_kc_incidence(A)
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    batch.exercise_ids[:, :] = 0
    model.set_hint_hyperedge_index(torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    with torch.no_grad():
        live = model(batch)
    model.set_hint_hyperedge_index(torch.empty((2, 0), dtype=torch.long))
    with torch.no_grad():
        empty = model(batch)
    assert (live - empty).abs().max().item() > 1e-4
    off = DH2KT(_v4_config(use_questions=True, question_graph=True))
    assert not hasattr(off, "hint_item_embed")
    assert not hasattr(off, "hint_hconv")


def test_v4_hint_hypergraph_membership_changes_logits():
    torch.manual_seed(1)
    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            hint_hypergraph=True,
        )
    )
    model.set_question_kc_incidence(torch.eye(6, 8))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    batch.exercise_ids[:, :] = 0
    model.set_hint_hyperedge_index(torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    with torch.no_grad():
        a = model(batch)
    model.set_hint_hyperedge_index(torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long))
    with torch.no_grad():
        b = model(batch)
    assert (a - b).abs().max().item() > 1e-4


def test_v4_hint_hypergraph_no_future_leak():
    torch.manual_seed(0)
    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            hint_hypergraph=True,
        )
    )
    model.set_question_kc_incidence(torch.eye(6, 8))
    model.set_hint_hyperedge_index(torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    cut = 2
    flipped = batch.responses.clone()
    flipped[:, cut + 1 :] = 1.0 - flipped[:, cut + 1 :]
    alt = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=flipped,
        hyperedge_index=batch.hyperedge_index,
        lengths=batch.lengths,
    )
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, : cut + 1] - changed[:, : cut + 1]).abs().max().item() < 1e-5
    assert (base[:, cut + 1 :] - changed[:, cut + 1 :]).abs().max().item() > 1e-5


def test_v4_hint_hypergraph_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    from dh2a_kt.train.checkpoint import load_trained_fold, save_trained_fold
    from dh2a_kt.train.tier1 import TrainedFold

    model = DH2KT(
        _v4_config(
            use_questions=True,
            question_graph=True,
            hint_hypergraph=True,
        )
    )
    model.set_question_kc_incidence(torch.eye(6, 8))
    model.set_hint_hyperedge_index(torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long))
    path = tmp_path / "hint.pt"
    save_trained_fold(
        path,
        TrainedFold(
            model=model,
            eval_loader=None,  # type: ignore[arg-type]
            kc_to_idx={i: i for i in range(8)},
            item_to_idx={i: i for i in range(6)},
            clean_hyperedge_index={},
            clean_hyperedges=[],
            device=torch.device("cpu"),
        ),
    )
    loaded = load_trained_fold(path, device="cpu")
    assert loaded.model.config.hint_hypergraph
    assert tuple(loaded.model.hint_edge_index.shape) == (2, 3)


def test_v4_time_gap_changes_logits():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(time_gap=True))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    zeros = torch.zeros(2, 5)
    batch.time_gaps = zeros
    with torch.no_grad():
        a = model(batch)
    batch.time_gaps = zeros + 3.0
    with torch.no_grad():
        b = model(batch)
    assert (a - b).abs().max().item() > 1e-4
    off = DH2KT(_v4_config())
    assert not hasattr(off, "time_gap_proj")


def _clone_batch_gaps(batch: DH2KTBatch, gaps: torch.Tensor) -> DH2KTBatch:
    return DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=batch.responses,
        hyperedge_index=batch.hyperedge_index,
        lengths=batch.lengths,
        time_gaps=gaps,
        durations=batch.durations,
        idles=batch.idles,
    )


def test_v4_time_gap_mode_lstm_prefix_stable_when_future_gaps_change():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(time_gap=True, time_gap_mode="lstm"))
    model.eval()
    cut = 2
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    zeros = torch.zeros(2, 6)
    batch.time_gaps = zeros
    alt = _clone_batch_gaps(batch, zeros.clone())
    alt.time_gaps[:, cut + 1 :] = 3.0
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, : cut + 1] - changed[:, : cut + 1]).abs().max().item() < 1e-5
    assert (base[:, cut + 1 :] - changed[:, cut + 1 :]).abs().max().item() > 1e-5


def test_v4_time_gap_mode_query_uses_next_gap_not_lstm():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(time_gap=True, time_gap_mode="query"))
    model.eval()
    cut = 2
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    zeros = torch.zeros(2, 6)
    batch.time_gaps = zeros
    alt = _clone_batch_gaps(batch, zeros.clone())
    alt.time_gaps[:, cut + 1 :] = 3.0
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    # next_gaps[cut] = gaps[cut+1], so the scored logit at `cut` moves;
    # earlier positions do not (query-only, no LSTM injection).
    assert (base[:, :cut] - changed[:, :cut]).abs().max().item() < 1e-5
    assert (base[:, cut] - changed[:, cut]).abs().max().item() > 1e-5


def test_v4_concept_forget_changes_logits():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(concept_forget=True))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    zeros = torch.zeros(2, 5)
    batch.time_gaps = zeros
    with torch.no_grad():
        a = model(batch)
    batch.time_gaps = zeros + 3.0
    with torch.no_grad():
        b = model(batch)
    assert (a - b).abs().max().item() > 1e-4
    off = DH2KT(_v4_config())
    assert not hasattr(off, "forget_log_theta")


def test_v4_time_split_query_does_not_use_next_duration():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(time_split=True, time_gap_mode="query"))
    model.eval()
    cut = 2
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    zeros = torch.zeros(2, 6)
    batch.durations = zeros
    batch.idles = zeros
    alt = _clone_batch_gaps(batch, None)
    alt.durations = zeros.clone()
    alt.idles = zeros.clone()
    alt.durations[:, cut + 1 :] = 3.0
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base - changed).abs().max().item() < 1e-5
    alt.idles[:, cut + 1 :] = 3.0
    with torch.no_grad():
        idle_changed = model(alt)
    assert (base[:, cut] - idle_changed[:, cut]).abs().max().item() > 1e-5


def test_v4_saw_input_not_on_query():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(saw_input=True))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=6)
    cut = 2
    batch.saw_flags = torch.zeros(2, 6, dtype=torch.long)
    alt = DH2KTBatch(
        concept_ids=batch.concept_ids,
        exercise_ids=batch.exercise_ids,
        responses=batch.responses,
        hyperedge_index=batch.hyperedge_index,
        lengths=batch.lengths,
        saw_flags=batch.saw_flags.clone(),
    )
    alt.saw_flags[:, cut + 1 :] = 1
    with torch.no_grad():
        base = model(batch)
        changed = model(alt)
    assert (base[:, : cut + 1] - changed[:, : cut + 1]).abs().max().item() < 1e-5
    assert (base[:, cut + 1 :] - changed[:, cut + 1 :]).abs().max().item() > 1e-5


def test_v4_saw_input_current_step_changes_logit():
    torch.manual_seed(1)
    model = DH2KT(_v4_config(saw_input=True))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    batch.saw_flags = torch.zeros(2, 5, dtype=torch.long)
    with torch.no_grad():
        a = model(batch)
    batch.saw_flags[:, 1] = 1
    with torch.no_grad():
        b = model(batch)
    assert (a - b).abs().max().item() > 1e-4


def test_v4_expert_graph_empty_vs_live():
    torch.manual_seed(0)
    model = DH2KT(_v4_config(expert_graph=True))
    model.eval()
    batch = _make_toy_batch(batch_size=2, seq_len=5)
    A = torch.zeros(8, 8)
    A[1, 0] = 1.0
    model.set_expert_adjacency(A)
    with torch.no_grad():
        live = model(batch)
    model.set_expert_adjacency(torch.zeros(8, 8))
    with torch.no_grad():
        empty = model(batch)
    assert (live - empty).abs().max().item() > 1e-4
    off = DH2KT(_v4_config())
    assert not hasattr(off, "expert_gcn_linear")


def test_v4_group_embed_absent_without_flag():
    off = DH2KT(_v4_config())
    on = DH2KT(_v4_config(group_embed=True, n_groups=4))
    assert not hasattr(off, "group_embed_table")
    assert on.group_embed_table.num_embeddings == 4


def test_dataset_time_gaps_survive_chunk_boundary():
    import numpy as np
    import pandas as pd

    from dh2a_kt.train.tier1 import UserSequenceDataset

    logs = pd.DataFrame(
        {
            "user_id": [1] * 5,
            "item_id": [0, 1, 2, 3, 4],
            "kc_id": [0, 0, 1, 1, 1],
            "timestamp": [0, 10, 20, 50, 51],
            "correct": [1, 0, 1, 0, 1],
        }
    )
    ds = UserSequenceDataset(
        logs,
        {0: 0, 1: 1},
        {i: i for i in range(5)},
        max_seq_len=3,
        window_mode="chunked",
        include_time_gaps=True,
    )
    assert len(ds) == 2
    first = ds[0]["time_gaps"][:3].tolist()
    second = ds[1]["time_gaps"][:2].tolist()
    assert first[0] == 0.0
    assert second[0] == pytest.approx(float(np.log1p(30)))


def test_v4_recap_attention_adds_projection():
    plain = DH2KT(_v4_config()).state_dict()
    recap = DH2KT(_v4_config(recap_attention=True)).state_dict()
    assert "recap_proj.weight" not in plain
    assert "recap_proj.weight" in recap



def test_v2_state_dict_has_no_v3_head():
    v2 = DH2KT(DH2KTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16,
                           hyperedge_kinds=("concept_prerequisite",), architecture="v2"))
    keys = v2.state_dict()
    assert "query_head.weight" not in keys
    v3 = DH2KT(_v3_config(hidden_dim=16, embed_dim=16))
    assert "query_head.weight" in v3.state_dict()
