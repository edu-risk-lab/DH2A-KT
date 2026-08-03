"""Native PyTorch implementation of GIKT (Graph-Interactive Knowledge Tracing) for the P0 evaluation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GIKTPyTorch(nn.Module):
    def __init__(
        self,
        num_q: int,
        num_c: int,
        emb_size: int = 64,
        hidden_dim: int = 128,
        bipartite_edges: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.model_name = "gikt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim

        # Embeddings
        self.q_embed = nn.Embedding(num_q, emb_size)
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)

        # Bipartite question-concept adjacency matrix
        # A_qs: [num_q, num_c], where A_qs[q, c] = 1 if question q has skill c.
        self.register_buffer("A_qs", torch.zeros(num_q, num_c))
        if bipartite_edges is not None:
            for q, c in bipartite_edges:
                if 0 <= q < num_q and 0 <= c < num_c:
                    self.A_qs[q, c] = 1.0

        # Row normalization for question embeddings (D_q^-1 * A_qs)
        row_sums = self.A_qs.sum(dim=1, keepdim=True)
        # Normalization factor, clamp at 1.0 to avoid division by zero
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_qs_norm", self.A_qs / row_norm)

        # GCN Linear mapping
        self.gcn_linear = nn.Linear(emb_size, emb_size)

        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=emb_size * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Attention layer for recap module
        self.attn_q = nn.Linear(emb_size, hidden_dim)
        self.attn_h = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

        # Output predictor
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + emb_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def get_gcn_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Perform embedding propagation over question-skill graph
        # Q_gcn = Q_init + GCN(C_init)
        c_emb = self.c_embed.weight  # [num_c, emb_size]
        q_emb = self.q_embed.weight  # [num_q, emb_size]

        # Aggregate skill embeddings to update question embeddings
        q_propagated = torch.matmul(self.A_qs_norm, c_emb)  # [num_q, emb_size]
        q_gcn = F.tanh(self.gcn_linear(q_emb + q_propagated))  # [num_q, emb_size]

        return q_gcn, c_emb

    def forward(
        self,
        qseqs: torch.Tensor,
        cseqs: torch.Tensor,
        rseqs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for batch.

        Parameters
        ----------
        qseqs : torch.Tensor of shape [batch_size, seq_len]
            Question sequence.
        cseqs : torch.Tensor of shape [batch_size, seq_len]
            Concept/skill sequence.
        rseqs : torch.Tensor of shape [batch_size, seq_len]
            Response sequence (0 or 1).

        Returns
        -------
        probs : torch.Tensor of shape [batch_size, seq_len]
            Correctness prediction probabilities.
        """
        batch_size, seq_len = qseqs.shape
        q_gcn, _ = self.get_gcn_embeddings()  # [num_q, emb_size]

        # Gather GCN embeddings for the sequence
        # We index using qseqs. clamp -1 to 0 for padding index, we will mask it out later
        qseqs_clamped = torch.clamp(qseqs, min=0)
        q_emb_seq = q_gcn[qseqs_clamped.long()]  # [batch_size, seq_len, emb_size]

        # Gather response embeddings
        rseqs_clamped = torch.clamp(rseqs, min=0)
        r_emb_seq = self.r_embed(rseqs_clamped.long())  # [batch_size, seq_len, emb_size]

        # Input to LSTM is concatenated question GCN embedding and response embedding
        lstm_input = torch.cat([q_emb_seq, r_emb_seq], dim=-1)  # [batch_size, seq_len, emb_size * 2]

        # Run LSTM to get student hidden states H
        H, _ = self.lstm(lstm_input)  # H: [batch_size, seq_len, hidden_dim]

        # For predicting correctness at step t+1, we use:
        # - hidden state H[:, t]
        # - attention over historical hidden states H[:, :t] using the GCN embedding of next question q[:, t+1]
        probs = torch.zeros(batch_size, seq_len, device=qseqs.device)

        # Loop through each step (seq_len - 1 predictions).
        # Store the prediction at t+1 so callers can compare probs[:, 1:]
        # with shifted responses. Writing to probs[:, t] would leak the
        # response at t+1 through H[:, t+1] when the caller slices y[:, 1:].
        for t in range(seq_len - 1):
            h_t = H[:, t, :]  # [batch_size, hidden_dim]
            next_q_emb = q_gcn[torch.clamp(qseqs[:, t + 1], min=0).long()]  # [batch_size, emb_size]

            # Recap module: Attention over history H[:, :t+1]
            H_hist = H[:, : t + 1, :]  # [batch_size, t+1, hidden_dim]

            # Project Query (next question) and Keys (history hidden states)
            q_proj = self.attn_q(next_q_emb).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            h_proj = self.attn_h(H_hist)  # [batch_size, t+1, hidden_dim]

            # Score and Softmax
            scores = self.attn_v(F.tanh(q_proj + h_proj)).squeeze(-1)  # [batch_size, t+1]
            attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch_size, t+1, 1]

            recap_context = torch.sum(attn_weights * H_hist, dim=1)  # [batch_size, hidden_dim]

            # Concat hidden state, recap context, and next question GCN embedding
            out_input = torch.cat([h_t, recap_context, next_q_emb], dim=-1)  # [batch_size, hidden_dim * 2 + emb_size]
            logits = self.fc_out(out_input).squeeze(-1)  # [batch_size]

            probs[:, t + 1] = torch.sigmoid(logits)

        # The prediction for the first element is standard prior correctness probability of the first question
        # We can set it to a learnable prior or simple prediction from the GCN embedding alone
        first_q_emb = q_gcn[torch.clamp(qseqs[:, 0], min=0).long()]  # [batch_size, emb_size]
        zero_h = torch.zeros(batch_size, self.hidden_dim, device=qseqs.device)
        first_input = torch.cat([zero_h, zero_h, first_q_emb], dim=-1)
        probs[:, 0] = torch.sigmoid(self.fc_out(first_input).squeeze(-1))

        return probs
