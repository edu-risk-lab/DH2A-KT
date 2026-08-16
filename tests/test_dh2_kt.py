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


def test_v4_question_variant_adds_item_parameters():
    plain = DH2KT(_v4_config()).state_dict()
    questioned = DH2KT(_v4_config(use_questions=True)).state_dict()
    assert "item_scale.weight" not in plain
    assert {"item_scale.weight", "item_bias.weight", "concept_var.weight"} <= set(questioned)


def test_v2_state_dict_has_no_v3_head():
    v2 = DH2KT(DH2KTConfig(n_concepts=8, n_exercises=6, hidden_dim=16, embed_dim=16,
                           hyperedge_kinds=("concept_prerequisite",), architecture="v2"))
    keys = v2.state_dict()
    assert "query_head.weight" not in keys
    v3 = DH2KT(_v3_config(hidden_dim=16, embed_dim=16))
    assert "query_head.weight" in v3.state_dict()
