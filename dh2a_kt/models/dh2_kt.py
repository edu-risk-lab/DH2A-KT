"""Tier 1 core model: DH2-KT (Dynamic Heterogeneous Hypergraph Knowledge
Tracing). docs/idea-D-plan.md Pha 2 ("Encoder + causal layer").

``architecture="v2"`` (default, Table 4 / GreyKT checkpoints):
  - one ``HypergraphConv`` stack per hyperedge kind;
  - concept states = hypergraph-refined embeddings (non-participating nodes
    zero — prevents graph-inert bypass);
  - dual-gated temporal update over a single student hidden state;
  - scalar head ``out_proj(h_t)`` that does **not** see the next concept.

``architecture="v3"`` (AUC upgrade, still graph-only / concept-level):
  - same hypergraph concept encoder + participation mask;
  - per-concept knowledge memory updated by the dual gate (dynamic graph);
  - read-side neighbor aggregation on the concept adjacency induced by
    hyperedges (knowledge diffusion along prereq/session structure);
  - next-step head queries ``(memory[c_{t+1}], e_{c_{t+1}})`` — GKT-style
    readout without putting an exercise embedding in the interaction path.
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
    architecture: str = "v2"
    """``v2``: scalar history head (Table 4). ``v3``: per-concept memory +
    next-concept query. Default v2 so existing checkpoints load strictly."""
    diffusion_alpha: float = 0.5
    """v3 only: weight of neighbor knowledge when reading ``c_{t+1}``."""


@dataclass
class DH2KTBatch:
    """Minimal batch contract for ``DH2KT.forward`` (M3).

    ``hyperedge_index`` maps each active kind to a PyG incidence matrix
    ``[2, num_incidence]`` (row 0 = concept node id, row 1 = hyperedge id).
    Kinds with no edges may be omitted or supplied as empty tensors.
    ``lengths`` is optional; v3 uses it to skip padded timesteps so they
    do not write into concept-0 memory.
    """

    concept_ids: torch.Tensor
    exercise_ids: torch.Tensor
    responses: torch.Tensor
    hyperedge_index: dict[str, torch.Tensor] = field(default_factory=dict)
    concept_states: torch.Tensor | None = None
    lengths: torch.Tensor | None = None


if _TORCH_AVAILABLE:

    def empty_hyperedge_index(
        device: torch.device | str | None = None,
        kinds: tuple[str, ...] = ("concept_prerequisite",),
    ) -> dict[str, torch.Tensor]:
        """Incidence matrix with zero hyperedges (graph-destroyed / dropout)."""
        return {
            kind: torch.empty((2, 0), dtype=torch.long, device=device) for kind in kinds
        }

    def normalized_concept_adjacency(
        hyperedge_index: dict[str, torch.Tensor],
        n_concepts: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Row-normalized concept–concept adjacency from hyperedge incidence.

        Within each hyperedge, consecutive listed concept members become an
        undirected edge (chain order for prerequisite hyperedges; listing
        order for session co-practice). Shape ``(n_concepts, n_concepts)``.
        """
        A = torch.zeros(n_concepts, n_concepts, device=device)
        for edge_index in hyperedge_index.values():
            if edge_index is None or edge_index.numel() == 0:
                continue
            nodes = edge_index[0].long()
            hedges = edge_index[1].long()
            order = torch.argsort(hedges)
            nodes = nodes[order]
            hedges = hedges[order]
            if nodes.numel() < 2:
                continue
            same = hedges[1:] == hedges[:-1]
            src = nodes[:-1][same]
            dst = nodes[1:][same]
            if src.numel() == 0:
                continue
            A[src, dst] = 1.0
            A[dst, src] = 1.0
        A.fill_diagonal_(0)
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        return A / deg

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
            if config.architecture not in ("v2", "v3"):
                raise ValueError(
                    f"architecture must be 'v2' or 'v3', got {config.architecture!r}"
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

            self.query_head: nn.Linear | None = None
            self.kind_mix: nn.Parameter | None = None
            if config.architecture == "v3":
                self.query_head = nn.Linear(config.hidden_dim * 2, 1)
                if len(config.hyperedge_kinds) > 1:
                    self.kind_mix = nn.Parameter(torch.zeros(len(config.hyperedge_kinds)))

        def _encode_concepts(
            self,
            hyperedge_index: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            concept_x = self.concept_embed.weight
            kind_outputs: list[torch.Tensor] = []
            kind_indices: list[int] = []
            participated = torch.zeros(
                concept_x.size(0), dtype=torch.bool, device=concept_x.device
            )
            kind_to_i = {k: i for i, k in enumerate(self.config.hyperedge_kinds)}

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
                kind_indices.append(kind_to_i[kind])

            if not kind_outputs:
                return torch.zeros_like(concept_x)
            if self.kind_mix is not None and len(kind_outputs) > 1:
                logits = self.kind_mix[kind_indices]
                weights = torch.softmax(logits, dim=0).view(-1, 1, 1)
                graph_x = (weights * torch.stack(kind_outputs, dim=0)).sum(dim=0)
            else:
                graph_x = torch.stack(kind_outputs, dim=0).mean(dim=0)
            out = torch.zeros_like(concept_x)
            out[participated] = graph_x[participated]
            return out

        def encode_sequence(self, batch: DH2KTBatch) -> torch.Tensor:
            """Return hidden states after each timestep, shape ``(B, T, hidden_dim)``.

            v2: student dual-gate trajectory. v3: queried per-concept memory
            at ``c_{t+1}`` (the representation that feeds the correctness head).
            """
            if self.config.architecture == "v3":
                return self._encode_sequence_v3(batch)[0]
            return self._encode_sequence_v2(batch)

        def _encode_sequence_v2(self, batch: DH2KTBatch) -> torch.Tensor:
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

        def _encode_sequence_v3(
            self, batch: DH2KTBatch
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Queried memory and next-concept graph embeddings, both ``(B, T, H)``."""
            concept_x = (
                batch.concept_states
                if batch.concept_states is not None
                else self._encode_concepts(batch.hyperedge_index)
            )
            batch_size, seq_len = batch.concept_ids.shape
            device = batch.concept_ids.device
            n_concepts = self.config.n_concepts
            hidden = self.config.hidden_dim
            batch_idx = torch.arange(batch_size, device=device)
            memory = torch.zeros(batch_size, n_concepts, hidden, device=device)
            adj = normalized_concept_adjacency(
                batch.hyperedge_index, n_concepts, device
            )
            lengths = batch.lengths
            queried: list[torch.Tensor] = []
            next_embeds: list[torch.Tensor] = []

            for t in range(seq_len):
                c_t = batch.concept_ids[:, t].clamp(0, n_concepts - 1)
                valid = (
                    torch.ones(batch_size, dtype=torch.bool, device=device)
                    if lengths is None
                    else t < lengths
                )
                e_t = concept_x[c_t] * self.graph_concept_gain
                r_t = self.response_proj(batch.responses[:, t].unsqueeze(-1).float())
                x_t = self.concept_proj(e_t) + r_t
                m_old = memory[batch_idx, c_t]
                m_new = self.dual_gate(m_old, x_t)
                if valid.any():
                    memory = memory.clone()
                    upd = m_new
                    if not valid.all():
                        upd = torch.where(valid.unsqueeze(-1), m_new, m_old)
                    memory[batch_idx, c_t] = upd

                c_next = (
                    batch.concept_ids[:, t + 1].clamp(0, n_concepts - 1)
                    if t + 1 < seq_len
                    else c_t
                )
                e_next = concept_x[c_next] * self.graph_concept_gain
                m_next = memory[batch_idx, c_next]
                neighbor_w = adj[c_next]
                agg = torch.bmm(neighbor_w.unsqueeze(1), memory).squeeze(1)
                m_query = m_next + self.config.diffusion_alpha * agg
                queried.append(m_query)
                next_embeds.append(e_next)

            return torch.stack(queried, dim=1), torch.stack(next_embeds, dim=1)

        def forward(self, batch: DH2KTBatch) -> torch.Tensor:
            """Per-timestep logits, shape ``(batch, seq_len, 1)``.

            ``logits[:, t]`` predicts correctness at step ``t + 1`` given
            interactions up to and including step ``t`` (KT next-step setup).
            v3 additionally conditions on the identity of concept ``t + 1``.
            """
            if self.config.architecture == "v3":
                queried, next_embeds = self._encode_sequence_v3(batch)
                assert self.query_head is not None
                return self.query_head(torch.cat([queried, next_embeds], dim=-1))
            return self.out_proj(self._encode_sequence_v2(batch))

else:  # pragma: no cover

    class DH2KT:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "DH2KT requires PyTorch. Install with: pip install torch (see requirements.txt)"
            )
