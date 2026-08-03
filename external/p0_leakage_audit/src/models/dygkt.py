"""Native PyTorch implementation of DyGKT (Dynamic Graph Knowledge Tracing) for the P0 evaluation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DyGKTPyTorch(nn.Module):
    def __init__(
        self,
        num_c: int,
        emb_size: int = 64,
        hidden_dim: int = 128,
        adj_matrix: torch.Tensor | None = None,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.model_name = "dygkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # Embeddings
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)

        # Dynamic update cells
        self.student_update = nn.GRUCell(emb_size * 2, hidden_dim)
        self.concept_update = nn.GRUCell(hidden_dim + emb_size, hidden_dim)

        # Adjacency matrix for structural propagation
        if adj_matrix is not None:
            self.register_buffer("A", adj_matrix.clone().detach().float())
        else:
            self.register_buffer("A", torch.eye(num_c).float())

        # Row normalization for the adjacency matrix
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_norm", self.A / row_norm)

        # Precompute the propagation matrix W = I + gamma * A_norm (same device as A)
        W_matrix = torch.eye(num_c, device=self.A.device, dtype=self.A.dtype) + gamma * self.A_norm
        self.register_buffer("W", W_matrix)

        # Predictor mapping joint student-concept state to correctness probability
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cseqs: torch.Tensor,
        rseqs: torch.Tensor,
        cshft: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DyGKT.

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

        # Initialize student dynamic state vector
        # shape: [batch_size, hidden_dim]
        u = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Initialize concept mastery matrix for each learner in the batch
        # shape: [batch_size, num_c, hidden_dim]
        C = torch.zeros(batch_size, self.num_c, self.hidden_dim, device=device)

        probs = torch.zeros(batch_size, seq_len, device=device)

        # Sequential step-by-step update and prediction
        for t in range(seq_len):
            # Target concept to predict at time t
            target_c = cshft[:, t].long()  # [batch_size]
            target_c_clamped = torch.clamp(target_c, min=0, max=self.num_c - 1)

            # Get target concept states: [batch_size, hidden_dim]
            c_state = C[torch.arange(batch_size, device=device), target_c_clamped]

            # Predict performance using combination of student state u and concept state c_state
            joint_input = torch.cat([u, c_state], dim=-1)  # [batch_size, hidden_dim * 2]
            logits = self.fc_out(joint_input).squeeze(-1)  # [batch_size]
            probs[:, t] = torch.sigmoid(logits)

            # Update the student and concept states with current interaction
            curr_c = cseqs[:, t].long()  # [batch_size]
            curr_r = rseqs[:, t].long()  # [batch_size]
            curr_c_clamped = torch.clamp(curr_c, min=0, max=self.num_c - 1)
            curr_r_clamped = torch.clamp(curr_r, min=0, max=1)

            c_emb = self.c_embed(curr_c_clamped)  # [batch_size, emb_size]
            r_emb = self.r_embed(curr_r_clamped)  # [batch_size, emb_size]

            # 1. Update student state u_t
            student_input = torch.cat([c_emb, r_emb], dim=-1)  # [batch_size, emb_size * 2]
            u = self.student_update(student_input, u)  # [batch_size, hidden_dim]

            # 2. Update concept state C_t[curr_c]
            curr_c_state = C[torch.arange(batch_size, device=device), curr_c_clamped]  # [batch_size, hidden_dim]
            concept_input = torch.cat([u, r_emb], dim=-1)  # [batch_size, hidden_dim + emb_size]
            updated_c_state = self.concept_update(concept_input, curr_c_state)  # [batch_size, hidden_dim]

            # Write updated concept state back into C
            C = C.clone()
            C[torch.arange(batch_size, device=device), curr_c_clamped] = updated_c_state

            # 3. Dynamic Graph Convolution Propagation
            # W shape: [num_c, num_c], C shape: [batch_size, num_c, hidden_dim]
            C = torch.matmul(self.W.unsqueeze(0), C)

        return probs
