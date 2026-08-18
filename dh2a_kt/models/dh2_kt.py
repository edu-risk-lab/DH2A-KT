"""Tier 1 core model: DH2-KT (Dynamic Heterogeneous Hypergraph Knowledge
Tracing). docs/idea-D-plan.md Pha 2 ("Encoder + causal layer").

``architecture="v2"`` (default, Table 4 / GreyKT checkpoints):
  - one ``HypergraphConv`` stack per hyperedge kind;
  - concept states = hypergraph-refined embeddings (non-participating nodes
    zero — prevents graph-inert bypass);
  - dual-gated temporal update over a single student hidden state;
  - scalar head ``out_proj(h_t)`` that does **not** see the next concept.

``architecture="v4"`` (AUC track; ``use_questions`` adds item difficulty):
  - concept states = ``concept_embed + hypergraph refinement`` (residual, so the
    ~50% of concepts absent from any hyperedge keep their identity);
  - LSTM backbone over an embedded ``(concept, response)`` interaction;
  - the response consumed at step ``t`` is the outcome **of** step ``t``, so the
    head predicts ``t+1`` from everything observed up to ``t`` (DKT alignment);
  - bilinear readout against the graph-refined vector of ``c_{t+1}``.

``architecture="v3"`` (AUC upgrade, still graph-only / concept-level):
  - same hypergraph concept encoder + participation mask;
  - per-concept knowledge memory updated by the dual gate (dynamic graph);
  - read-side neighbor aggregation on the concept adjacency induced by
    hyperedges (knowledge diffusion along prereq/session structure);
  - next-step head queries ``(memory[c_{t+1}], e_{c_{t+1}})`` — GKT-style
    readout without putting an exercise embedding in the interaction path.

``architecture="v5"`` (graph-reliance track; needs collapsed multi-KC events):
  v2--v4 all let the hypergraph refine a *static* concept embedding table. That
  table is itself a free parameter, so anything the graph produces the embedding
  can learn directly: on XES3G5M fold 0, deleting the graph moved AUC by 0.0002
  and the M6 manipulation check by 0.00003. v5 removes the redundancy by making
  the graph move quantities no static table can hold:
  - each event is one question attempt carrying a *set* of KCs (see
    ``dh2a_kt.data.events``), pooled into a single event vector, so a multi-KC
    question is modelled as the hyperedge it actually is instead of as repeated
    rows that leak their own label;
  - a per-concept mastery memory is written on every attempt and *transported to
    hyperedge neighbours*, so graph structure carries the learner's own history
    between co-examined KCs;
  - the readout adds the target KCs' mastery and their transported neighbour
    mastery to v4's bilinear term. Destroying the graph therefore destroys a
    learner-specific signal, which is the falsifiable difference from v4.
"""

from __future__ import annotations

import logging
import math
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
    next-concept query. ``v4``: LSTM backbone + bilinear next-concept readout.
    Default v2 so existing checkpoints load strictly."""
    diffusion_alpha: float = 0.5
    """v3 only: weight of neighbor knowledge when reading ``c_{t+1}``."""
    use_questions: bool = False
    """v4/v5: add Rasch-style question difficulty on top of the concept vector."""
    n_lstm_layers: int = 1
    """v4/v5: depth of the LSTM backbone."""
    memory_dim: int = 16
    """v5 only: width of the per-concept mastery memory.

    Deliberately much smaller than ``hidden_dim``: the memory is materialised as
    ``(batch, n_concepts, memory_dim)`` and rewritten at every timestep, so a
    128-wide memory over 865 concepts is what made v3 slow.
    """
    max_degree: int = 16
    """v5 only: cap on hyperedge neighbours transported per concept."""
    graph_transport: float = 0.5
    """v5 only: damping on evidence written to a neighbour rather than the
    exercised concept. Zero makes the graph inert by construction."""
    event_pool: str = "mean"
    """v5 only: ``mean`` or ``attention`` over the KC set of an event."""
    transport: str = "clique"
    """v5 only: ``clique`` expands each hyperedge to pairwise neighbours;
    ``star`` keeps the hyperedge as an intermediate node (concept→edge→concept)."""
    use_hyperedge_embed: bool = False
    """v5 only: add a learned embedding of the question-as-hyperedge to the event."""
    kind_conditioned: bool = False
    """v5 only: project each hyperedge kind's concept states with its own Linear
    before mixing, so a KC in a question hyperedge is not forced to share the
    same refined vector as in a prerequisite hyperedge."""
    recap_attention: bool = False
    """v4 only: causal attention over LSTM history before the bilinear readout."""
    question_kc_agg: bool = False
    """v4 only: add mean of observed KCs for the question to the Rasch vector.

    Uses the train-time question→KC incidence (not prerequisite hyperedges).
    Legacy E3 path; prefer ``question_graph`` for a non-absorbable Q←KC refine.
    """
    question_graph: bool = False
    """v4 only: GIKT-style question embedding refined by observed Q–KC incidence.

    ``q_gcn = tanh(W(q_emb + A_norm @ concept_emb))`` — no response transport.
    """
    question_hypergraph: bool = False
    """v4 only: HypergraphConv on observed multi-KC ``question_concepts`` edges.

    Isolated from v5 memory/transport and from E_pre chains. Refine is added
    through a dedicated Linear so the mix is not absorbable into ``concept_embed``.
    """
    max_hyperedge_members: int = 8
    """v5 star transport: pad width for members of one hyperedge."""
    max_hedges_per_concept: int = 8
    """v5 star transport: pad width for hyperedges incident on one concept."""


@dataclass
class DH2KTBatch:
    """Minimal batch contract for ``DH2KT.forward`` (M3).

    ``hyperedge_index`` maps each active kind to a PyG incidence matrix
    ``[2, num_incidence]`` (row 0 = concept node id, row 1 = hyperedge id).
    Kinds with no edges may be omitted or supplied as empty tensors.
    ``lengths`` is optional; v3 uses it to skip padded timesteps so they
    do not write into concept-0 memory.

    ``responses`` semantics depend on the architecture: v2/v3 expect the
    response shifted by one step, v4 expects the response of the same step
    (see :func:`dh2a_kt.train.tier1.responses_for_model`).
    """

    concept_ids: torch.Tensor
    exercise_ids: torch.Tensor
    responses: torch.Tensor
    hyperedge_index: dict[str, torch.Tensor] = field(default_factory=dict)
    concept_states: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    kc_set_ids: torch.Tensor | None = None
    """v5 only: ``(B, T, K)`` padded KC set exercised by each event."""
    kc_set_mask: torch.Tensor | None = None
    """v5 only: ``(B, T, K)`` marking which KC slots are real."""


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

    def concept_neighbors_from_hyperedges(
        hyperedge_index: dict[str, torch.Tensor],
        n_concepts: int,
        device: torch.device,
        *,
        max_degree: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded co-membership neighbour list, shape ``(n_concepts, max_degree)``.

        Every pair of concepts sharing a hyperedge is adjacent. That differs from
        :func:`normalized_concept_adjacency`, which only links *consecutive*
        members because prerequisite hyperedges encode a chain order; a question
        hyperedge has no order — all of its KCs are exercised together.

        Returns ``(neighbour ids, validity mask)``. Concepts with no neighbour get
        an all-false mask row, so the caller must not read their padded ids.
        """
        neigh = torch.zeros(n_concepts, max_degree, dtype=torch.long, device=device)
        mask = torch.zeros(n_concepts, max_degree, dtype=torch.bool, device=device)
        buckets: list[set[int]] = [set() for _ in range(n_concepts)]
        for edge_index in hyperedge_index.values():
            if edge_index is None or edge_index.numel() == 0:
                continue
            nodes = edge_index[0].tolist()
            hedges = edge_index[1].tolist()
            members: dict[int, list[int]] = {}
            for node, hedge in zip(nodes, hedges, strict=True):
                members.setdefault(hedge, []).append(node)
            for group in members.values():
                unique = {n for n in group if 0 <= n < n_concepts}
                if len(unique) < 2:
                    continue
                for node in unique:
                    buckets[node].update(unique - {node})
        for concept, others in enumerate(buckets):
            if not others:
                continue
            take = sorted(others)[:max_degree]
            neigh[concept, : len(take)] = torch.tensor(take, device=device)
            mask[concept, : len(take)] = True
        return neigh, mask

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over the second-to-last axis, ignoring masked slots.

        ``values`` is ``(..., K, D)`` and ``mask`` is ``(..., K)``. Rows with no
        valid slot return zeros rather than NaN.
        """
        weights = mask.to(values.dtype).unsqueeze(-1)
        total = (values * weights).sum(dim=-2)
        count = weights.sum(dim=-2).clamp(min=1.0)
        return total / count

    def masked_attention_pool(
        values: torch.Tensor,
        mask: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax-weighted pool over the second-to-last axis.

        ``scores`` has the same leading shape as ``mask``. Empty rows (no valid
        slot) return zeros, matching :func:`masked_mean`.
        """
        neg_inf = torch.finfo(scores.dtype).min
        scored = scores.masked_fill(~mask, neg_inf)
        # Rows with no valid member would be all -inf; force them to zero weight.
        empty = ~mask.any(dim=-1)
        if empty.any():
            scored = scored.clone()
            scored[empty] = 0.0
        weights = torch.softmax(scored, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (values * weights.unsqueeze(-1)).sum(dim=-2)

    def star_incidence_tables(
        hyperedge_index: dict[str, torch.Tensor],
        n_concepts: int,
        device: torch.device,
        *,
        max_members: int = 8,
        max_hedges: int = 8,
        kinds: tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pad the bipartite incidence of a hypergraph for star message passing.

        Returns
        -------
        hedge_members : (H, max_members) long
        hedge_mask : (H, max_members) bool
        concept_hedges : (n_concepts, max_hedges) long
        concept_mask : (n_concepts, max_hedges) bool
        n_hedges : int

        Unlike :func:`concept_neighbors_from_hyperedges` (clique expansion), the
        hyperedge stays an explicit intermediate: concepts talk only through it.
        """
        members_by_hedge: list[list[int]] = []
        global_hedge = 0
        kind_filter = None if kinds is None else set(kinds)
        for kind, edge_index in hyperedge_index.items():
            if kind_filter is not None and kind not in kind_filter:
                continue
            if edge_index is None or edge_index.numel() == 0:
                continue
            nodes = edge_index[0].tolist()
            hedges = edge_index[1].tolist()
            local: dict[int, list[int]] = {}
            for node, hedge in zip(nodes, hedges, strict=True):
                if 0 <= node < n_concepts:
                    local.setdefault(hedge, []).append(node)
            for group in local.values():
                unique = sorted(set(group))
                if len(unique) < 2:
                    continue
                members_by_hedge.append(unique)
                global_hedge += 1

        n_hedges = len(members_by_hedge)
        hedge_members = torch.zeros(max(n_hedges, 1), max_members, dtype=torch.long, device=device)
        hedge_mask = torch.zeros(max(n_hedges, 1), max_members, dtype=torch.bool, device=device)
        concept_hedges = torch.zeros(n_concepts, max_hedges, dtype=torch.long, device=device)
        concept_mask = torch.zeros(n_concepts, max_hedges, dtype=torch.bool, device=device)
        if n_hedges == 0:
            return hedge_members, hedge_mask, concept_hedges, concept_mask, 0

        per_concept: list[list[int]] = [[] for _ in range(n_concepts)]
        for h, group in enumerate(members_by_hedge):
            take = group[:max_members]
            hedge_members[h, : len(take)] = torch.tensor(take, device=device)
            hedge_mask[h, : len(take)] = True
            for node in take:
                if len(per_concept[node]) < max_hedges:
                    per_concept[node].append(h)
        for c, hs in enumerate(per_concept):
            if not hs:
                continue
            concept_hedges[c, : len(hs)] = torch.tensor(hs, device=device)
            concept_mask[c, : len(hs)] = True
        return hedge_members, hedge_mask, concept_hedges, concept_mask, n_hedges

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
            if config.architecture not in ("v2", "v3", "v4", "v5"):
                raise ValueError(
                    "architecture must be 'v2', 'v3', 'v4' or 'v5', "
                    f"got {config.architecture!r}"
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
                if config.question_hypergraph and kind == "question_concepts":
                    # Dedicated one-step stack below; do not also residual-add
                    # through the absorbable v2–v4 ``_encode_concepts`` path.
                    continue
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
            if config.architecture in ("v3", "v4", "v5") and len(config.hyperedge_kinds) > 1:
                self.kind_mix = nn.Parameter(torch.zeros(len(config.hyperedge_kinds)))

            if config.architecture == "v5":
                hidden = config.hidden_dim
                mem = config.memory_dim
                # Sequence branch: same engine as v4, but over collapsed events.
                self.interaction_proj = nn.Linear(hidden + 1, hidden)
                self.input_norm = nn.LayerNorm(hidden)
                self.lstm = nn.LSTM(
                    hidden,
                    hidden,
                    num_layers=config.n_lstm_layers,
                    batch_first=True,
                    dropout=config.dropout if config.n_lstm_layers > 1 else 0.0,
                )
                self.query_proj = nn.Linear(hidden, hidden)
                self.concept_bias = nn.Embedding(config.n_concepts, 1)
                nn.init.zeros_(self.concept_bias.weight)
                # Memory branch: per-concept mastery that the hypergraph moves.
                self.memory_write = nn.GRUCell(hidden + 1, mem)
                self.memory_readout = nn.Linear(2 * mem, 1)
                self.event_pool_score = nn.Linear(hidden, 1)
                if config.use_hyperedge_embed:
                    self.hyperedge_embed = nn.Embedding(config.n_exercises, hidden)
                    nn.init.zeros_(self.hyperedge_embed.weight)
                if config.kind_conditioned:
                    self.kind_refine = nn.ModuleDict(
                        {kind: nn.Linear(hidden, hidden) for kind in config.hyperedge_kinds}
                    )
                    self.event_concept_proj = nn.Linear(hidden, hidden)
                if config.use_questions:
                    self.item_scale = nn.Embedding(config.n_exercises, 1)
                    self.concept_var = nn.Embedding(config.n_concepts, hidden)
                    self.item_bias = nn.Embedding(config.n_exercises, 1)
                    nn.init.zeros_(self.item_scale.weight)
                    nn.init.zeros_(self.concept_var.weight)
                    nn.init.zeros_(self.item_bias.weight)
                    self.item_query = nn.Embedding(config.n_exercises, hidden)
                    nn.init.zeros_(self.item_query.weight)

            if config.architecture == "v4":
                hidden = config.hidden_dim
                self.interaction_embed = nn.Embedding(2 * config.n_concepts, hidden)
                self.concept_in_proj = nn.Linear(hidden, hidden)
                self.input_norm = nn.LayerNorm(hidden)
                self.lstm = nn.LSTM(
                    hidden,
                    hidden,
                    num_layers=config.n_lstm_layers,
                    batch_first=True,
                    dropout=config.dropout if config.n_lstm_layers > 1 else 0.0,
                )
                self.query_proj = nn.Linear(hidden, hidden)
                self.concept_bias = nn.Embedding(config.n_concepts, 1)
                nn.init.zeros_(self.concept_bias.weight)
                if config.recap_attention:
                    self.recap_proj = nn.Linear(hidden, hidden)
                if config.use_questions:
                    # Rasch parameterisation (AKT/simpleKT): a scalar per item
                    # scaling a per-concept variation vector, plus an item bias.
                    self.item_scale = nn.Embedding(config.n_exercises, 1)
                    self.concept_var = nn.Embedding(config.n_concepts, hidden)
                    self.item_bias = nn.Embedding(config.n_exercises, 1)
                    nn.init.zeros_(self.item_scale.weight)
                    nn.init.zeros_(self.concept_var.weight)
                    nn.init.zeros_(self.item_bias.weight)
                if config.question_graph:
                    # Own question parameters + one propagation step over the
                    # observed question–KC incidence (GIKT-style; not absorbable
                    # into concept_embed the way E3's mean-add was).
                    self.question_embed = nn.Embedding(config.n_exercises, hidden)
                    self.question_gcn_linear = nn.Linear(hidden, hidden)
                    self.question_in_proj = nn.Linear(hidden, hidden)
                    self.register_buffer(
                        "A_qs_norm",
                        torch.zeros(config.n_exercises, config.n_concepts),
                    )
                if config.question_hypergraph:
                    # One HypergraphConv hop over multi-KC question hyperedges,
                    # then a fresh Linear (non-absorbable into concept_embed).
                    self.question_hconv = nn.ModuleList(
                        [HypergraphConv(hidden, hidden)]
                    )
                    self.question_hconv_linear = nn.Linear(hidden, hidden)

        def set_question_kc_table(
            self,
            exercise_kc_ids: "torch.Tensor",
            exercise_kc_mask: "torch.Tensor",
        ) -> None:
            """Register the observed question→KC incidence used by ``question_kc_agg``."""
            self.register_buffer("exercise_kc_ids", exercise_kc_ids.long())
            self.register_buffer("exercise_kc_mask", exercise_kc_mask.bool())

        def set_question_kc_incidence(self, A_qs: "torch.Tensor") -> None:
            """Register row-normalized Q–KC incidence for ``question_graph``.

            ``A_qs`` is ``(n_exercises, n_concepts)`` with 1 where the question
            exercises that KC (train-only observed metadata).
            """
            A = A_qs.float()
            row_norm = A.sum(dim=1, keepdim=True).clamp(min=1.0)
            normalized = A / row_norm
            if hasattr(self, "A_qs_norm"):
                self.A_qs_norm.copy_(normalized.to(device=self.A_qs_norm.device))
            else:
                self.register_buffer("A_qs_norm", normalized)

        def _question_gcn_embeddings(self) -> "torch.Tensor":
            """All question vectors after optional Q←KC propagation, ``(n_ex, H)``."""
            q_emb = self.question_embed.weight
            q_prop = self.A_qs_norm @ self.concept_embed.weight
            return torch.tanh(self.question_gcn_linear(q_emb + q_prop))

        def _question_vectors(self, exercise_ids: "torch.Tensor") -> "torch.Tensor":
            """Indexed ``q_gcn`` for a batch of item ids, ``(B, T, H)``."""
            items = exercise_ids.clamp(0, self.config.n_exercises - 1)
            return self._question_gcn_embeddings()[items]

        def _question_hconv_refine(
            self,
            hyperedge_index: dict[str, torch.Tensor],
        ) -> "torch.Tensor":
            """Concept refine from multi-KC question hyperedges, ``(n_concepts, H)``.

            Empty incidence returns zeros so ``--no-graph`` is a hard ablation.
            """
            edge_index = hyperedge_index.get("question_concepts")
            base = self.concept_embed.weight
            if edge_index is None or edge_index.numel() == 0:
                return torch.zeros_like(base)
            h = base
            for conv in self.question_hconv:
                h = conv(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
            return torch.tanh(self.question_hconv_linear(h))

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
                if (
                    self.config.architecture == "v5"
                    and self.config.kind_conditioned
                    and hasattr(self, "kind_refine")
                    and kind in self.kind_refine
                ):
                    h = self.kind_refine[kind](h)
                kind_outputs.append(h)
                kind_indices.append(kind_to_i[kind])

            if not kind_outputs:
                # v3/v4 next-concept readouts need base identity when the graph is
                # empty. v2 keeps zeros so Table-4 checkpoints stay compatible.
                if self.config.architecture in ("v3", "v4", "v5"):
                    return concept_x
                return torch.zeros_like(concept_x)
            if self.kind_mix is not None and len(kind_outputs) > 1:
                logits = self.kind_mix[kind_indices]
                weights = torch.softmax(logits, dim=0).view(-1, 1, 1)
                graph_x = (weights * torch.stack(kind_outputs, dim=0)).sum(dim=0)
            else:
                graph_x = torch.stack(kind_outputs, dim=0).mean(dim=0)
            if self.config.architecture in ("v3", "v4", "v5"):
                # Residual: HypergraphConv over a shared edge can collapse members,
                # and on XES3G5M only 420/865 concepts occur in any hyperedge — the
                # rest must keep concept_embed instead of becoming zero vectors.
                out = concept_x.clone()
                out[participated] = concept_x[participated] + graph_x[participated]
                return out
            out = torch.zeros_like(concept_x)
            out[participated] = graph_x[participated]
            return out

        def encode_sequence(self, batch: DH2KTBatch) -> torch.Tensor:
            """Return hidden states after each timestep, shape ``(B, T, hidden_dim)``.

            v2: student dual-gate trajectory. v3: queried per-concept memory
            at ``c_{t+1}``. v4: LSTM hidden states.
            """
            if self.config.architecture == "v4":
                return self._encode_sequence_v4(batch)[0]
            if self.config.architecture == "v3":
                return self._encode_sequence_v3(batch)[0]
            return self._encode_sequence_v2(batch)

        def _concept_vectors_v4(
            self,
            concept_x: torch.Tensor,
            concept_ids: torch.Tensor,
            exercise_ids: torch.Tensor | None,
        ) -> torch.Tensor:
            """Graph-refined vector per timestep, optionally Rasch-adjusted."""
            vectors = concept_x[concept_ids]
            if self.config.use_questions and exercise_ids is not None:
                items = exercise_ids.clamp(0, self.config.n_exercises - 1)
                vectors = vectors + self.item_scale(items) * self.concept_var(concept_ids)
            if (
                self.config.question_kc_agg
                and exercise_ids is not None
                and hasattr(self, "exercise_kc_ids")
            ):
                # Observed Q–KC incidence (train only): mean over KCs of the item.
                items = exercise_ids.clamp(0, self.config.n_exercises - 1)
                kc_ids = self.exercise_kc_ids[items]
                kc_mask = self.exercise_kc_mask[items]
                member = self.concept_embed(kc_ids)
                vectors = vectors + masked_mean(member, kc_mask)
            return vectors

        def _encode_sequence_v4(
            self, batch: DH2KTBatch
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """LSTM hidden states and the query vectors read at ``c_{t+1}``."""
            concept_x = (
                batch.concept_states
                if batch.concept_states is not None
                else self._encode_concepts(batch.hyperedge_index)
            )
            n_concepts = self.config.n_concepts
            concepts = batch.concept_ids.clamp(0, n_concepts - 1)
            responses = batch.responses.clamp(0.0, 1.0).round().long()
            exercises = batch.exercise_ids if batch.exercise_ids is not None else None

            interaction = self.interaction_embed(concepts + n_concepts * responses)
            observed = self._concept_vectors_v4(concept_x, concepts, exercises)
            hconv_refine = None
            if self.config.question_hypergraph:
                hconv_refine = self._question_hconv_refine(batch.hyperedge_index)
                observed = observed + hconv_refine[concepts]
            x = interaction + self.concept_in_proj(observed)
            if self.config.question_graph and exercises is not None:
                x = x + self.question_in_proj(self._question_vectors(exercises))
            x = self.input_norm(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
            hidden, _ = self.lstm(x)
            hidden = F.dropout(hidden, p=self.config.dropout, training=self.training)

            # Position t is scored against the concept asked at t+1; the final
            # column is never used by the loss or the AUC mask.
            next_concepts = torch.cat([concepts[:, 1:], concepts[:, -1:]], dim=1)
            next_exercises = (
                torch.cat([exercises[:, 1:], exercises[:, -1:]], dim=1)
                if exercises is not None
                else None
            )
            queried = self._concept_vectors_v4(concept_x, next_concepts, next_exercises)
            if self.config.question_graph and next_exercises is not None:
                queried = queried + self.question_in_proj(
                    self._question_vectors(next_exercises)
                )
            if hconv_refine is not None:
                queried = queried + hconv_refine[next_concepts]
            return hidden, queried

        def _readout_v4(self, batch: DH2KTBatch) -> torch.Tensor:
            hidden, queried = self._encode_sequence_v4(batch)
            n_concepts = self.config.n_concepts
            concepts = batch.concept_ids.clamp(0, n_concepts - 1)
            next_concepts = torch.cat([concepts[:, 1:], concepts[:, -1:]], dim=1)
            scale = math.sqrt(self.config.hidden_dim)
            q = self.query_proj(queried)
            if self.config.recap_attention:
                # Causal recap over LSTM history: retrieve sparse distant evidence
                # that a single hidden state tends to forget on long logs.
                _, t_len, hidden_dim = hidden.shape
                scores = torch.einsum("bth,bsh->bts", q, hidden) / math.sqrt(hidden_dim)
                causal = torch.ones(
                    t_len, t_len, dtype=torch.bool, device=hidden.device
                ).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
                attn = scores.softmax(dim=-1)
                context = torch.einsum("bts,bsh->bth", attn, hidden)
                readout = hidden + self.recap_proj(context)
                logits = (readout * q).sum(dim=-1, keepdim=True) / scale
            else:
                logits = (hidden * q).sum(dim=-1, keepdim=True) / scale
            logits = logits + self.concept_bias(next_concepts)
            if self.config.use_questions and batch.exercise_ids is not None:
                items = batch.exercise_ids.clamp(0, self.config.n_exercises - 1)
                next_items = torch.cat([items[:, 1:], items[:, -1:]], dim=1)
                logits = logits + self.item_bias(next_items)
            return logits

        def _event_vectors_v5(
            self,
            concept_x: torch.Tensor,
            kc_ids: torch.Tensor,
            kc_mask: torch.Tensor,
            exercise_ids: torch.Tensor | None,
        ) -> torch.Tensor:
            """Pool a question's KC set into one event vector, shape ``(B, T, H)``.

            With ``event_pool='attention'`` the question (or a learned query)
            weights the KCs instead of treating them as equal; with
            ``use_hyperedge_embed`` the question itself contributes a vector, so
            the hyperedge is an object rather than only an incidence pattern.
            """
            vectors = concept_x[kc_ids]
            if self.config.kind_conditioned and hasattr(self, "event_concept_proj"):
                vectors = self.event_concept_proj(vectors)
            if self.config.use_questions and exercise_ids is not None:
                items = exercise_ids.clamp(0, self.config.n_exercises - 1)
                scale = self.item_scale(items)
                vectors = vectors + scale.unsqueeze(-2) * self.concept_var(kc_ids)
            if self.config.event_pool == "attention":
                if (
                    self.config.use_questions
                    and exercise_ids is not None
                    and hasattr(self, "item_query")
                ):
                    items = exercise_ids.clamp(0, self.config.n_exercises - 1)
                    query = self.item_query(items).unsqueeze(-2)
                    scores = (vectors * query).sum(dim=-1) / math.sqrt(vectors.size(-1))
                else:
                    scores = self.event_pool_score(vectors).squeeze(-1)
                pooled = masked_attention_pool(vectors, kc_mask, scores)
            else:
                pooled = masked_mean(vectors, kc_mask)
            if (
                self.config.use_hyperedge_embed
                and exercise_ids is not None
                and hasattr(self, "hyperedge_embed")
            ):
                items = exercise_ids.clamp(0, self.config.n_exercises - 1)
                pooled = pooled + self.hyperedge_embed(items)
            return pooled

        def _star_transport_read(
            self,
            memory: torch.Tensor,
            q_ids: torch.Tensor,
            q_mask: torch.Tensor,
            hedge_members: torch.Tensor,
            hedge_mask: torch.Tensor,
            concept_hedges: torch.Tensor,
            concept_hmask: torch.Tensor,
        ) -> torch.Tensor:
            """Read mastery that arrived through hyperedges (star), shape ``(B, mem)``."""
            batch_size = memory.size(0)
            rows = torch.arange(batch_size, device=memory.device)
            # concept -> its hyperedges
            h_ids = concept_hedges[q_ids]  # (B, K, Hmax)
            h_valid = concept_hmask[q_ids] & q_mask.unsqueeze(-1)
            # hyperedge -> mean of member memories
            members = hedge_members[h_ids]  # (B, K, Hmax, M)
            m_valid = hedge_mask[h_ids] & h_valid.unsqueeze(-1)
            flat_m = members.reshape(batch_size, -1)
            flat_v = m_valid.reshape(batch_size, -1)
            return masked_mean(memory[rows.unsqueeze(1), flat_m], flat_v)

        def _star_transport_write(
            self,
            memory: torch.Tensor,
            evidence: torch.Tensor,
            write_ids: torch.Tensor,
            write_mask: torch.Tensor,
            hedge_members: torch.Tensor,
            hedge_mask: torch.Tensor,
            concept_hedges: torch.Tensor,
            concept_hmask: torch.Tensor,
            transport: float,
        ) -> torch.Tensor:
            """Push evidence through incident hyperedges onto their other members."""
            if transport <= 0.0 or not write_mask.any():
                return memory
            batch_size = memory.size(0)
            device = memory.device
            rows = torch.arange(batch_size, device=device)
            h_ids = concept_hedges[write_ids]
            h_valid = concept_hmask[write_ids] & write_mask.unsqueeze(-1)
            if not h_valid.any():
                return memory
            members = hedge_members[h_ids]
            m_valid = hedge_mask[h_ids] & h_valid.unsqueeze(-1)
            # Drop the writer slots themselves is unnecessary: blending with own
            # state is a no-op under small transport; keeping the gather simple.
            flat_ids = members.reshape(-1)
            flat_valid = m_valid.reshape(-1)
            if not flat_valid.any():
                return memory
            k_slots, h_max, m_max = write_ids.size(1), h_ids.size(-1), members.size(-1)
            flat_rows = (
                rows.view(batch_size, 1, 1, 1)
                .expand(-1, k_slots, h_max, m_max)
                .reshape(-1)
            )
            t_prev = memory[flat_rows[flat_valid], flat_ids[flat_valid]]
            t_inp = (
                evidence.view(batch_size, 1, 1, 1, -1)
                .expand(-1, k_slots, h_max, m_max, -1)
                .reshape(-1, evidence.size(-1))[flat_valid]
            )
            t_new = self.memory_write(t_inp, t_prev)
            blended = t_prev + transport * (t_new - t_prev)
            return memory.index_put((flat_rows[flat_valid], flat_ids[flat_valid]), blended)

        def _readout_v5(self, batch: DH2KTBatch) -> torch.Tensor:
            if batch.kc_set_ids is None or batch.kc_set_mask is None:
                raise ValueError("v5 requires kc_set_ids and kc_set_mask on the batch")

            config = self.config
            n_concepts = config.n_concepts
            device = batch.concept_ids.device
            concept_x = (
                batch.concept_states
                if batch.concept_states is not None
                else self._encode_concepts(batch.hyperedge_index)
            )
            kc_ids = batch.kc_set_ids.clamp(0, n_concepts - 1)
            kc_mask = batch.kc_set_mask.bool()
            responses = batch.responses.clamp(0.0, 1.0).unsqueeze(-1).float()
            exercises = batch.exercise_ids

            events = self._event_vectors_v5(concept_x, kc_ids, kc_mask, exercises)
            batch_size, seq_len, hidden = events.shape

            # --- sequence branch (v4's engine, now over collapsed events) ---
            x = self.input_norm(self.interaction_proj(torch.cat([events, responses], dim=-1)))
            x = F.dropout(x, p=config.dropout, training=self.training)
            lstm_out, _ = self.lstm(x)
            lstm_out = F.dropout(lstm_out, p=config.dropout, training=self.training)

            # --- memory branch: mastery that the hypergraph moves ---
            use_star = config.transport == "star"
            if use_star:
                # Prefer question hyperedges for star transport: that is the
                # genuine multi-way relation. Fall back to all kinds if none.
                hedge_members, hedge_mask, concept_hedges, concept_hmask, n_hedges = (
                    star_incidence_tables(
                        batch.hyperedge_index,
                        n_concepts,
                        device,
                        max_members=config.max_hyperedge_members,
                        max_hedges=config.max_hedges_per_concept,
                        kinds=("question_concepts",),
                    )
                )
                if n_hedges == 0:
                    hedge_members, hedge_mask, concept_hedges, concept_hmask, n_hedges = (
                        star_incidence_tables(
                            batch.hyperedge_index,
                            n_concepts,
                            device,
                            max_members=config.max_hyperedge_members,
                            max_hedges=config.max_hedges_per_concept,
                        )
                    )
                neigh = neigh_mask = None
            else:
                neigh, neigh_mask = concept_neighbors_from_hyperedges(
                    batch.hyperedge_index, n_concepts, device, max_degree=config.max_degree
                )
                hedge_members = hedge_mask = concept_hedges = concept_hmask = None

            memory = torch.zeros(batch_size, n_concepts, config.memory_dim, device=device)
            evidence = torch.cat([events, responses], dim=-1)
            lengths = batch.lengths
            rows = torch.arange(batch_size, device=device)

            next_kc_ids = torch.cat([kc_ids[:, 1:], kc_ids[:, -1:]], dim=1)
            next_kc_mask = torch.cat([kc_mask[:, 1:], kc_mask[:, -1:]], dim=1)
            mem_features: list[torch.Tensor] = []

            for t in range(seq_len):
                q_ids = next_kc_ids[:, t]
                q_mask = next_kc_mask[:, t]
                direct = masked_mean(memory[rows.unsqueeze(1), q_ids], q_mask)
                if use_star:
                    transported = self._star_transport_read(
                        memory,
                        q_ids,
                        q_mask,
                        hedge_members,
                        hedge_mask,
                        concept_hedges,
                        concept_hmask,
                    )
                else:
                    n_ids = neigh[q_ids]
                    n_valid = neigh_mask[q_ids] & q_mask.unsqueeze(-1)
                    flat = n_ids.reshape(batch_size, -1)
                    flat_valid = n_valid.reshape(batch_size, -1)
                    transported = masked_mean(memory[rows.unsqueeze(1), flat], flat_valid)
                mem_features.append(torch.cat([direct, transported], dim=-1))

                valid = (
                    torch.ones(batch_size, dtype=torch.bool, device=device)
                    if lengths is None
                    else t < lengths
                )
                if not valid.any():
                    continue
                write_ids = kc_ids[:, t]
                write_mask = kc_mask[:, t] & valid.unsqueeze(-1)
                if not write_mask.any():
                    continue
                ev = evidence[:, t]
                k_slots = write_ids.size(1)
                flat_rows = rows.unsqueeze(1).expand(-1, k_slots).reshape(-1)
                flat_ids = write_ids.reshape(-1)
                flat_sel = write_mask.reshape(-1)
                prev = memory[flat_rows[flat_sel], flat_ids[flat_sel]]
                inp = ev.unsqueeze(1).expand(-1, k_slots, -1).reshape(-1, ev.size(-1))[flat_sel]
                updated = self.memory_write(inp, prev)
                memory = memory.index_put(
                    (flat_rows[flat_sel], flat_ids[flat_sel]), updated
                )
                if config.graph_transport > 0.0:
                    if use_star:
                        memory = self._star_transport_write(
                            memory,
                            ev,
                            write_ids,
                            write_mask,
                            hedge_members,
                            hedge_mask,
                            concept_hedges,
                            concept_hmask,
                            config.graph_transport,
                        )
                    else:
                        t_ids = neigh[write_ids]
                        t_valid = neigh_mask[write_ids] & write_mask.unsqueeze(-1)
                        tf_ids = t_ids.reshape(-1)
                        tf_valid = t_valid.reshape(-1)
                        if tf_valid.any():
                            deg = t_ids.size(-1)
                            tf_rows = (
                                rows.unsqueeze(1)
                                .unsqueeze(2)
                                .expand(-1, k_slots, deg)
                                .reshape(-1)
                            )
                            t_prev = memory[tf_rows[tf_valid], tf_ids[tf_valid]]
                            t_inp = (
                                ev.unsqueeze(1)
                                .unsqueeze(2)
                                .expand(-1, k_slots, deg, -1)
                                .reshape(-1, ev.size(-1))[tf_valid]
                            )
                            t_new = self.memory_write(t_inp, t_prev)
                            blended = t_prev + config.graph_transport * (t_new - t_prev)
                            memory = memory.index_put(
                                (tf_rows[tf_valid], tf_ids[tf_valid]), blended
                            )

            queried = self._event_vectors_v5(
                concept_x,
                next_kc_ids,
                next_kc_mask,
                torch.cat([exercises[:, 1:], exercises[:, -1:]], dim=1)
                if exercises is not None
                else None,
            )
            scale = math.sqrt(hidden)
            logits = (lstm_out * self.query_proj(queried)).sum(dim=-1, keepdim=True) / scale
            logits = logits + self.memory_readout(torch.stack(mem_features, dim=1))
            logits = logits + masked_mean(
                self.concept_bias(next_kc_ids), next_kc_mask
            )
            if config.use_questions and exercises is not None:
                items = exercises.clamp(0, config.n_exercises - 1)
                next_items = torch.cat([items[:, 1:], items[:, -1:]], dim=1)
                logits = logits + self.item_bias(next_items)
            return logits

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
            v3/v4 additionally condition on the identity of concept ``t + 1``.
            """
            if self.config.architecture == "v5":
                return self._readout_v5(batch)
            if self.config.architecture == "v4":
                return self._readout_v4(batch)
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
