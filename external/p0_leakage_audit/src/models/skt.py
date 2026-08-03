"""Native PyTorch implementation of SKT (Structure-based Knowledge Tracing) for the P0 evaluation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SKTPyTorch(nn.Module):
    def __init__(
        self,
        num_c: int,
        emb_size: int = 64,
        hidden_dim: int = 128,
        adj_matrix: torch.Tensor | None = None,
        beta: float = 0.1,
    ):
        super().__init__()
        self.model_name = "skt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Embeddings
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)

        # Recurrent cell for updating the interacted concept state
        self.gru_cell = nn.GRUCell(emb_size * 2, hidden_dim)

        # Adjacency matrix for structure-based propagation
        if adj_matrix is not None:
            self.register_buffer("A", adj_matrix.clone().detach().float())
        else:
            self.register_buffer("A", torch.eye(num_c).float())

        # Row normalization for the adjacency matrix to ensure stable propagation
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_norm", self.A / row_norm)

        # Predictor mapping concept state to correctness probability
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cseqs: torch.Tensor,
        rseqs: torch.Tensor,
        cshft: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for SKT.

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

        # Initialize concept mastery matrix for each learner in the batch
        # shape: [batch_size, num_c, hidden_dim]
        M = torch.zeros(batch_size, self.num_c, self.hidden_dim, device=device)

        probs = torch.zeros(batch_size, seq_len, device=device)

        # Step-by-step sequential update and prediction
        for t in range(seq_len):
            # Target concept to predict at time t
            target_c = cshft[:, t].long()  # [batch_size]

            # Predict performance on target concept using current state M_t
            # Gather state for target_c from M: [batch_size, hidden_dim]
            target_c_clamped = torch.clamp(target_c, min=0, max=self.num_c - 1)
            state_target = M[torch.arange(batch_size, device=device), target_c_clamped]
            logits = self.fc_out(state_target).squeeze(-1)  # [batch_size]
            probs[:, t] = torch.sigmoid(logits)

            # Now update the state with the interaction at step t
            curr_c = cseqs[:, t].long()  # [batch_size]
            curr_r = rseqs[:, t].long()  # [batch_size]

            # Embeddings
            curr_c_clamped = torch.clamp(curr_c, min=0, max=self.num_c - 1)
            c_emb = self.c_embed(curr_c_clamped)  # [batch_size, emb_size]
            curr_r_clamped = torch.clamp(curr_r, min=0, max=1)
            r_emb = self.r_embed(curr_r_clamped)  # [batch_size, emb_size]

            interaction = torch.cat([c_emb, r_emb], dim=-1)  # [batch_size, emb_size * 2]

            # Gather current state of the interacted concept
            state_curr = M[torch.arange(batch_size, device=device), curr_c_clamped]  # [batch_size, hidden_dim]

            # Update state of the interacted concept via GRU
            updated_state = self.gru_cell(interaction, state_curr)  # [batch_size, hidden_dim]

            # Calculate difference (delta) for propagation
            delta = updated_state - state_curr  # [batch_size, hidden_dim]

            # Propagate delta to neighboring concepts using the normalized adjacency matrix column slicing
            A_col = self.A_norm[:, curr_c_clamped].t()  # [batch_size, num_c]
            propagation = A_col.unsqueeze(-1) * delta.unsqueeze(1)  # [batch_size, num_c, hidden_dim]

            # Update the mastery states
            M = M + self.beta * propagation
            M = M.clone()
            M[torch.arange(batch_size, device=device), curr_c_clamped] = M[torch.arange(batch_size, device=device), curr_c_clamped] + delta

        return probs
