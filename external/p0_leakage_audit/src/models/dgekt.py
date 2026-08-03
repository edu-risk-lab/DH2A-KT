"""Native PyTorch implementation of DGEKT (Dual Graph Ensemble Knowledge Tracing) for the P0 evaluation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGEKTPyTorch(nn.Module):
    def __init__(
        self,
        num_c: int,
        emb_size: int = 64,
        hidden_dim: int = 128,
        adj_matrix: torch.Tensor | None = None,
        beta: float = 0.1,
    ):
        super().__init__()
        self.model_name = "dgekt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Embeddings
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)

        # ------------------ View 1: Sequential LSTM (Student Interaction View) ------------------
        self.lstm = nn.LSTM(
            input_size=emb_size * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_seq = nn.Linear(hidden_dim, 1)

        # ------------------ View 2: Concept Graph GNN (Concept Relationship View) ------------------
        self.gkt_gru = nn.GRUCell(emb_size * 2, hidden_dim)
        if adj_matrix is not None:
            self.register_buffer("A", adj_matrix.clone().detach().float())
        else:
            self.register_buffer("A", torch.eye(num_c).float())

        # Row normalization for GNN view
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_norm", self.A / row_norm)
        self.fc_graph = nn.Linear(hidden_dim, 1)

        # ------------------ Ensemble Gate ------------------
        # Dynamic gating mechanism to fuse View 1 and View 2
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cseqs: torch.Tensor,
        rseqs: torch.Tensor,
        cshft: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DGEKT.

        Parameters
        ----------
        cseqs : torch.Tensor of shape [batch_size, seq_len]
            Concept/skill sequence.
        rseqs : torch.Tensor of shape [batch_size, seq_len]
            Response sequence (0 or 1).
        cshft : torch.Tensor of shape [batch_size, seq_len]
            Shifted concept sequence (target concepts to predict).

        Returns
        -------
        probs : torch.Tensor of shape [batch_size, seq_len]
            Correctness prediction probabilities.
        """
        batch_size, seq_len = cseqs.shape
        device = cseqs.device

        # Embeddings for the sequence
        cseqs_clamped = torch.clamp(cseqs, min=0, max=self.num_c - 1)
        rseqs_clamped = torch.clamp(rseqs, min=0, max=1)

        c_emb_seq = self.c_embed(cseqs_clamped.long())  # [batch_size, seq_len, emb_size]
        r_emb_seq = self.r_embed(rseqs_clamped.long())  # [batch_size, seq_len, emb_size]

        # Concat inputs
        interaction_seq = torch.cat([c_emb_seq, r_emb_seq], dim=-1)  # [batch_size, seq_len, emb_size * 2]

        # ------------------ View 1: LSTM Forward ------------------
        H_seq, _ = self.lstm(interaction_seq)  # [batch_size, seq_len, hidden_dim]

        # ------------------ View 2: Concept Graph GNN Forward ------------------
        # Concept mastery matrix (updated sequentially and propagated via graph)
        M_graph = torch.zeros(batch_size, self.num_c, self.hidden_dim, device=device)

        probs = torch.zeros(batch_size, seq_len, device=device)

        for t in range(seq_len):
            # Target concept to predict
            target_c = cshft[:, t].long()
            target_c_clamped = torch.clamp(target_c, min=0, max=self.num_c - 1)

            # 1. Prediction from Sequential View
            h_seq_t = H_seq[:, t, :]  # [batch_size, hidden_dim]
            prob_seq = torch.sigmoid(self.fc_seq(h_seq_t)).squeeze(-1)  # [batch_size]

            # 2. Prediction from Concept Graph GNN View
            state_graph_target = M_graph[torch.arange(batch_size, device=device), target_c_clamped]  # [batch_size, hidden_dim]
            prob_graph = torch.sigmoid(self.fc_graph(state_graph_target)).squeeze(-1)  # [batch_size]

            # 3. Dynamic Ensemble Fusion (Gating)
            gate_input = torch.cat([h_seq_t, state_graph_target], dim=-1)  # [batch_size, hidden_dim * 2]
            gate_val = self.gate_layer(gate_input).squeeze(-1)  # [batch_size]

            # Weighted sum of predictions
            probs[:, t] = gate_val * prob_graph + (1.0 - gate_val) * prob_seq

            # 4. GNN state update and propagation
            curr_c = cseqs[:, t].long()
            curr_c_clamped = torch.clamp(curr_c, min=0, max=self.num_c - 1)
            interaction_t = interaction_seq[:, t, :]  # [batch_size, emb_size * 2]

            state_graph_curr = M_graph[torch.arange(batch_size, device=device), curr_c_clamped]
            updated_state = self.gkt_gru(interaction_t, state_graph_curr)

            delta = updated_state - state_graph_curr

            # Propagate delta to neighboring concepts using the normalized adjacency matrix column slicing
            A_col = self.A_norm[:, curr_c_clamped].t()  # [batch_size, num_c]
            propagation = A_col.unsqueeze(-1) * delta.unsqueeze(1)  # [batch_size, num_c, hidden_dim]

            # Update the mastery states
            M_graph = M_graph + self.beta * propagation
            M_graph = M_graph.clone()
            M_graph[torch.arange(batch_size, device=device), curr_c_clamped] = M_graph[torch.arange(batch_size, device=device), curr_c_clamped] + delta

        return probs
