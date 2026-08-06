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
