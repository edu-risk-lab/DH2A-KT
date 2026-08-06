"""Tier 1 core model: DH2-KT (Dynamic Heterogeneous Hypergraph Knowledge
Tracing). docs/idea-D-plan.md Pha 2 ("Encoder + causal layer").

M3 implements ``DH2KT.forward()``:
  - one ``HypergraphConv`` stack per configured hyperedge kind;
  - concept states = hypergraph-refined embeddings for nodes in at least one
    hyperedge (non-participating nodes are zero — prevents graph-inert bypass);
  - dual-gated temporal update over graph concept states (+ past response only;
    no exercise embedding in the interaction path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import HypergraphConv

    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False


@dataclass
class DH2KTConfig:
    n_concepts: int
    n_exercises: int
    embed_dim: int = 128
    hidden_dim: int = 128
    n_hypergraph_layers: int = 2
    dropout: float = 0.2
    hyperedge_kinds: tuple[str, ...] = (
        "concept_prerequisite",
        "session",
        "discussion_thread",
        "teacher_intervention",
    )


@dataclass
class DH2KTBatch:
    """Minimal batch contract for ``DH2KT.forward`` (M3).

    ``hyperedge_index`` maps each active kind to a PyG incidence matrix
    ``[2, num_incidence]`` (row 0 = concept node id, row 1 = hyperedge id).
    Kinds with no edges may be omitted or supplied as empty tensors.
    """

    concept_ids: torch.Tensor
    exercise_ids: torch.Tensor
    responses: torch.Tensor
    hyperedge_index: dict[str, torch.Tensor] = field(default_factory=dict)
    concept_states: torch.Tensor | None = None


if _TORCH_AVAILABLE:

    def empty_hyperedge_index(device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        """Incidence matrix with zero hyperedges (graph-destroyed / dropout state)."""
        return {"concept_prerequisite": torch.empty((2, 0), dtype=torch.long, device=device)}

    class DualGatedUpdate(nn.Module):
        """Balance prior hidden state vs new interaction evidence (HGKT-style)."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.forget_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.input_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.candidate = nn.Linear(hidden_dim * 2, hidden_dim)

        def forward(self, prev_h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            combined = torch.cat([prev_h, x_t], dim=-1)
            g_forget = torch.sigmoid(self.forget_gate(combined))
            g_input = torch.sigmoid(self.input_gate(combined))
            c = torch.tanh(self.candidate(combined))
            return g_forget * prev_h + g_input * c

    class DH2KT(nn.Module):
        """Tier 1 DH²-KT encoder + next-step correctness head."""

        def __init__(self, config: DH2KTConfig):
            super().__init__()
            if not _PYG_AVAILABLE:
                raise ImportError(
                    "DH2KT requires torch_geometric. Install with: "
                    "pip install torch_geometric (see docs/execution-plan.md M3)"
                )
            self.config = config
            self.concept_embed = nn.Embedding(config.n_concepts, config.hidden_dim)
            self.response_proj = nn.Linear(1, config.hidden_dim)
            self.concept_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.graph_concept_gain = nn.Parameter(torch.tensor(2.0))
            self.dual_gate = DualGatedUpdate(config.hidden_dim)
            self.out_proj = nn.Linear(config.hidden_dim, 1)

            self.hypergraph_layers = nn.ModuleDict()
            for kind in config.hyperedge_kinds:
                layers: list[nn.Module] = []
                in_dim = config.hidden_dim
                for _ in range(config.n_hypergraph_layers):
                    layers.append(HypergraphConv(in_dim, config.hidden_dim))
                    in_dim = config.hidden_dim
                self.hypergraph_layers[kind] = nn.ModuleList(layers)

        def _encode_concepts(
            self,
            hyperedge_index: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            concept_x = self.concept_embed.weight
            kind_outputs: list[torch.Tensor] = []
            participated = torch.zeros(concept_x.size(0), dtype=torch.bool, device=concept_x.device)

            for kind, layers in self.hypergraph_layers.items():
                edge_index = hyperedge_index.get(kind)
                if edge_index is None or edge_index.numel() == 0:
                    continue
                participated[edge_index[0].unique()] = True
                h = concept_x
                for conv in layers:
                    h = conv(h, edge_index)
                    h = F.relu(h)
                    h = F.dropout(h, p=self.config.dropout, training=self.training)
                kind_outputs.append(h)

            if not kind_outputs:
                return torch.zeros_like(concept_x)
            graph_x = torch.stack(kind_outputs, dim=0).mean(dim=0)
            out = torch.zeros_like(concept_x)
            out[participated] = graph_x[participated]
            return out

        def encode_sequence(self, batch: DH2KTBatch) -> torch.Tensor:
            """Return hidden states after each timestep, shape ``(B, T, hidden_dim)``."""
            concept_x = (
                batch.concept_states
                if batch.concept_states is not None
                else self._encode_concepts(batch.hyperedge_index)
            )
            batch_size, seq_len = batch.concept_ids.shape
            device = batch.concept_ids.device
            h = torch.zeros(batch_size, self.config.hidden_dim, device=device)
            states: list[torch.Tensor] = []

            for t in range(seq_len):
                c_t = concept_x[batch.concept_ids[:, t]] * self.graph_concept_gain
                r_t = self.response_proj(batch.responses[:, t].unsqueeze(-1).float())
                x_t = self.concept_proj(c_t) + r_t
                h = self.dual_gate(h, x_t)
                states.append(h)

            return torch.stack(states, dim=1)

        def forward(self, batch: DH2KTBatch) -> torch.Tensor:
            """Per-timestep logits, shape ``(batch, seq_len, 1)``.

            ``logits[:, t]`` predicts correctness at step ``t + 1`` given
            interactions up to and including step ``t`` (KT next-step setup).
            """
            return self.out_proj(self.encode_sequence(batch))

else:  # pragma: no cover

    class DH2KT:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "DH2KT requires PyTorch. Install with: pip install torch (see requirements.txt)"
            )
