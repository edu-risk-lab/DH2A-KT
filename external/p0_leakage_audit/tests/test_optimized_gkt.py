import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from src.models.skt import SKTPyTorch
from src.models.dygkt import DyGKTPyTorch
from src.models.dgekt import DGEKTPyTorch
from pykt.models.gkt import GKT as GKTOptimized

# Original implementations for verification and side-by-side comparison
class SKTOriginal(nn.Module):
    def __init__(self, num_c, emb_size=64, hidden_dim=128, adj_matrix=None, beta=0.1):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)
        self.gru_cell = nn.GRUCell(emb_size * 2, hidden_dim)
        if adj_matrix is not None:
            self.register_buffer("A", adj_matrix.clone().detach().float())
        else:
            self.register_buffer("A", torch.eye(num_c).float())
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_norm", self.A / row_norm)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cseqs, rseqs, cshft):
        batch_size, seq_len = cseqs.shape
        device = cseqs.device
        M = torch.zeros(batch_size, self.num_c, self.hidden_dim, device=device)
        probs = torch.zeros(batch_size, seq_len, device=device)
        for t in range(seq_len):
            target_c = cshft[:, t].long()
            target_c_clamped = torch.clamp(target_c, min=0, max=self.num_c - 1)
            state_target = M[torch.arange(batch_size, device=device), target_c_clamped]
            logits = self.fc_out(state_target).squeeze(-1)
            probs[:, t] = torch.sigmoid(logits)

            curr_c = cseqs[:, t].long()
            curr_r = rseqs[:, t].long()
            curr_c_clamped = torch.clamp(curr_c, min=0, max=self.num_c - 1)
            c_emb = self.c_embed(curr_c_clamped)
            curr_r_clamped = torch.clamp(curr_r, min=0, max=1)
            r_emb = self.r_embed(curr_r_clamped)
            interaction = torch.cat([c_emb, r_emb], dim=-1)

            state_curr = M[torch.arange(batch_size, device=device), curr_c_clamped]
            updated_state = self.gru_cell(interaction, state_curr)
            delta = updated_state - state_curr

            delta_M = torch.zeros_like(M)
            delta_M[torch.arange(batch_size, device=device), curr_c_clamped] = delta
            propagation = torch.matmul(self.A_norm.unsqueeze(0), delta_M)
            M = M + delta_M + self.beta * propagation
        return probs

class DyGKTOriginal(nn.Module):
    def __init__(self, num_c, emb_size=64, hidden_dim=128, adj_matrix=None, gamma=0.1):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)
        self.student_update = nn.GRUCell(emb_size * 2, hidden_dim)
        self.concept_update = nn.GRUCell(hidden_dim + emb_size, hidden_dim)
        if adj_matrix is not None:
            self.register_buffer("A", adj_matrix.clone().detach().float())
        else:
            self.register_buffer("A", torch.eye(num_c).float())
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_norm", self.A / row_norm)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cseqs, rseqs, cshft):
        batch_size, seq_len = cseqs.shape
        device = cseqs.device
        u = torch.zeros(batch_size, self.hidden_dim, device=device)
        C = torch.zeros(batch_size, self.num_c, self.hidden_dim, device=device)
        probs = torch.zeros(batch_size, seq_len, device=device)
        for t in range(seq_len):
            target_c = cshft[:, t].long()
            target_c_clamped = torch.clamp(target_c, min=0, max=self.num_c - 1)
            c_state = C[torch.arange(batch_size, device=device), target_c_clamped]
            joint_input = torch.cat([u, c_state], dim=-1)
            logits = self.fc_out(joint_input).squeeze(-1)
            probs[:, t] = torch.sigmoid(logits)

            curr_c = cseqs[:, t].long()
            curr_r = rseqs[:, t].long()
            curr_c_clamped = torch.clamp(curr_c, min=0, max=self.num_c - 1)
            curr_r_clamped = torch.clamp(curr_r, min=0, max=1)
            c_emb = self.c_embed(curr_c_clamped)
            r_emb = self.r_embed(curr_r_clamped)

            student_input = torch.cat([c_emb, r_emb], dim=-1)
            u = self.student_update(student_input, u)

            curr_c_state = C[torch.arange(batch_size, device=device), curr_c_clamped]
            concept_input = torch.cat([u, r_emb], dim=-1)
            updated_c_state = self.concept_update(concept_input, curr_c_state)

            C = C.clone()
            C[torch.arange(batch_size, device=device), curr_c_clamped] = updated_c_state

            neighbor_contrib = torch.matmul(self.A_norm.unsqueeze(0), C)
            C = C + self.gamma * neighbor_contrib
        return probs

class DGEKTOriginal(nn.Module):
    def __init__(self, num_c, emb_size=64, hidden_dim=128, adj_matrix=None, beta=0.1):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.c_embed = nn.Embedding(num_c, emb_size)
        self.r_embed = nn.Embedding(2, emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_seq = nn.Linear(hidden_dim, 1)
        self.gkt_gru = nn.GRUCell(emb_size * 2, hidden_dim)
        if adj_matrix is not None:
            self.register_buffer("A", adj_matrix.clone().detach().float())
        else:
            self.register_buffer("A", torch.eye(num_c).float())
        row_sums = self.A.sum(dim=1, keepdim=True)
        row_norm = torch.clamp(row_sums, min=1.0)
        self.register_buffer("A_norm", self.A / row_norm)
        self.fc_graph = nn.Linear(hidden_dim, 1)
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, cseqs, rseqs, cshft):
        batch_size, seq_len = cseqs.shape
        device = cseqs.device
        cseqs_clamped = torch.clamp(cseqs, min=0, max=self.num_c - 1)
        rseqs_clamped = torch.clamp(rseqs, min=0, max=1)
        c_emb_seq = self.c_embed(cseqs_clamped.long())
        r_emb_seq = self.r_embed(rseqs_clamped.long())
        interaction_seq = torch.cat([c_emb_seq, r_emb_seq], dim=-1)

        H_seq, _ = self.lstm(interaction_seq)
        M_graph = torch.zeros(batch_size, self.num_c, self.hidden_dim, device=device)
        probs = torch.zeros(batch_size, seq_len, device=device)
        for t in range(seq_len):
            target_c = cshft[:, t].long()
            target_c_clamped = torch.clamp(target_c, min=0, max=self.num_c - 1)
            h_seq_t = H_seq[:, t, :]
            prob_seq = torch.sigmoid(self.fc_seq(h_seq_t)).squeeze(-1)

            state_graph_target = M_graph[torch.arange(batch_size, device=device), target_c_clamped]
            prob_graph = torch.sigmoid(self.fc_graph(state_graph_target)).squeeze(-1)

            gate_input = torch.cat([h_seq_t, state_graph_target], dim=-1)
            gate_val = self.gate_layer(gate_input).squeeze(-1)
            probs[:, t] = gate_val * prob_graph + (1.0 - gate_val) * prob_seq

            curr_c = cseqs[:, t].long()
            curr_c_clamped = torch.clamp(curr_c, min=0, max=self.num_c - 1)
            interaction_t = interaction_seq[:, t, :]
            state_graph_curr = M_graph[torch.arange(batch_size, device=device), curr_c_clamped]
            updated_state = self.gkt_gru(interaction_t, state_graph_curr)
            delta = updated_state - state_graph_curr

            delta_M = torch.zeros_like(M_graph)
            delta_M[torch.arange(batch_size, device=device), curr_c_clamped] = delta
            propagation = torch.matmul(self.A_norm.unsqueeze(0), delta_M)
            M_graph = M_graph + delta_M + self.beta * propagation
        return probs

class GKTOriginal(nn.Module):
    def __init__(self, num_c, hidden_dim, emb_size, graph_type="dense", graph=None, dropout=0.5, emb_type="qid", emb_path="", bias=True):
        super(GKTOriginal, self).__init__()
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2
        self.graph = nn.Parameter(graph)
        self.graph.requires_grad = False
        self.emb_type = emb_type
        self.emb_path = emb_path

        self.register_buffer("one_hot_feat", torch.eye(self.res_len * self.num_c))
        self.register_buffer("one_hot_q", torch.cat((torch.eye(self.num_c), torch.zeros(1, self.num_c)), dim=0))

        self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
        self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        from pykt.models.gkt import MLP, EraseAddGate
        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList([
            MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias),
            MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        ])
        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    def _aggregate(self, xt, qt, ht, batch_size):
        device = xt.device
        qt_mask = torch.ne(qt, -1)
        x_idx_mat = torch.arange(self.res_len * self.num_c, device=device)
        x_embedding = self.interaction_emb(x_idx_mat)
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)
        res_embedding = masked_feat.mm(x_embedding)
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.num_c * torch.ones((batch_size, self.num_c), device=device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.num_c, device=device)
        concept_embedding = self.emb_c(concept_idx_mat)

        index_tuple = (torch.arange(mask_num, device=device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)
        return tmp_ht

    def _agg_neighbors(self, tmp_ht, qt):
        device = qt.device
        qt_mask = torch.ne(qt, -1)
        masked_qt = qt[qt_mask]
        masked_tmp_ht = tmp_ht[qt_mask]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]
        self_features = self.f_self(self_ht)
        
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.num_c, 1)
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)
        
        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)
        reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)
        neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)

        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, None, None, None

    def _update(self, tmp_ht, ht, qt):
        qt_mask = torch.ne(qt, -1)
        mask_num = qt_mask.nonzero().shape[0]
        m_next, _, _, _ = self._agg_neighbors(tmp_ht, qt)
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))
        index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.num_c, self.hidden_dim))
        return h_next, None, None, None

    def _predict(self, h_next, qt):
        qt_mask = torch.ne(qt, -1)
        y = self.predict(h_next).squeeze(dim=-1)
        y[qt_mask] = torch.sigmoid(y[qt_mask])
        return y

    def _get_next_pred(self, yt, q_next):
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.num_c * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)
        pred = (yt * one_hot_qt).sum(dim=1)
        return pred

    def forward(self, q, r):
        device = q.device
        features = q * 2 + r
        questions = q
        batch_size, seq_len = features.shape
        ht = torch.zeros((batch_size, self.num_c, self.hidden_dim), device=device)
        pred_list = []
        for i in range(seq_len):
            xt = features[:, i]
            qt = questions[:, i]
            qt_mask = torch.ne(qt, -1)
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)
            h_next, _, _, _ = self._update(tmp_ht, ht, qt)
            ht[qt_mask] = h_next[qt_mask]
            yt = self._predict(h_next, qt)
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)
        return pred_res

# Test Correctness: Optimized outputs must be mathematically identical to original outputs
def test_correctness():
    num_c = 100
    batch_size = 8
    seq_len = 20
    emb_size = 16
    hidden_dim = 32
    
    # Random sparse adjacency matrix
    adj = (torch.rand(num_c, num_c) < 0.05).float()
    
    cseqs = torch.randint(0, num_c, (batch_size, seq_len))
    rseqs = torch.randint(0, 2, (batch_size, seq_len))
    cshft = torch.randint(0, num_c, (batch_size, seq_len))
    
    # 1. SKT Correctness
    torch.manual_seed(42)
    skt_orig = SKTOriginal(num_c, emb_size, hidden_dim, adj, 0.1).eval()
    skt_opt = SKTPyTorch(num_c, emb_size, hidden_dim, adj, 0.1).eval()
    skt_opt.load_state_dict(skt_orig.state_dict(), strict=False)
    
    probs_orig = skt_orig(cseqs, rseqs, cshft)
    probs_opt = skt_opt(cseqs, rseqs, cshft)
    assert torch.allclose(probs_orig, probs_opt, atol=1e-6)

    # 2. DyGKT Correctness
    torch.manual_seed(42)
    dygkt_orig = DyGKTOriginal(num_c, emb_size, hidden_dim, adj, 0.1).eval()
    dygkt_opt = DyGKTPyTorch(num_c, emb_size, hidden_dim, adj, 0.1).eval()
    dygkt_opt.load_state_dict(dygkt_orig.state_dict(), strict=False)
    
    probs_orig = dygkt_orig(cseqs, rseqs, cshft)
    probs_opt = dygkt_opt(cseqs, rseqs, cshft)
    assert torch.allclose(probs_orig, probs_opt, atol=1e-6)

    # 3. DGEKT Correctness
    torch.manual_seed(42)
    dgekt_orig = DGEKTOriginal(num_c, emb_size, hidden_dim, adj, 0.1).eval()
    dgekt_opt = DGEKTPyTorch(num_c, emb_size, hidden_dim, adj, 0.1).eval()
    dgekt_opt.load_state_dict(dgekt_orig.state_dict(), strict=False)
    
    probs_orig = dgekt_orig(cseqs, rseqs, cshft)
    probs_opt = dgekt_opt(cseqs, rseqs, cshft)
    assert torch.allclose(probs_orig, probs_opt, atol=1e-6)

    # 4. GKT (official submodule) Correctness
    torch.manual_seed(42)
    gkt_orig = GKTOriginal(num_c, hidden_dim, emb_size, graph=adj).eval()
    gkt_opt = GKTOptimized(num_c, hidden_dim, emb_size, graph=adj).eval()
    gkt_opt.load_state_dict(gkt_orig.state_dict(), strict=False)

    probs_orig = gkt_orig(cseqs, rseqs)
    probs_opt = gkt_opt(cseqs, rseqs)
    assert torch.allclose(probs_orig, probs_opt, atol=1e-6)


# Test Benchmark: Realistic sparse performance comparison
def test_benchmark_sparse():
    num_c = 500
    batch_size = 64
    seq_len = 100
    iterations = 5

    cseqs = torch.randint(0, num_c, (batch_size, seq_len))
    rseqs = torch.randint(0, 2, (batch_size, seq_len))
    cshft = torch.randint(0, num_c, (batch_size, seq_len))
    
    # 1% density sparse adjacency matrix (realistic)
    adj = (torch.rand(num_c, num_c) < 0.01).float()

    print("\n--- Correctness & Benchmark results on Realistic Sparse Graph (density 1%) ---")

    # 1. SKT
    skt_orig = SKTOriginal(num_c=num_c, adj_matrix=adj)
    skt_opt = SKTPyTorch(num_c=num_c, adj_matrix=adj)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = skt_orig(cseqs, rseqs, cshft)
    t_orig = (time.time() - t0) / iterations

    t0 = time.time()
    for _ in range(iterations):
        _ = skt_opt(cseqs, rseqs, cshft)
    t_opt = (time.time() - t0) / iterations
    
    print(f"SKT  -> Original: {t_orig:.4f}s | Optimized: {t_opt:.4f}s | Speedup: {t_orig/t_opt:.1f}x")

    # 2. DyGKT
    dygkt_orig = DyGKTOriginal(num_c=num_c, adj_matrix=adj)
    dygkt_opt = DyGKTPyTorch(num_c=num_c, adj_matrix=adj)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = dygkt_orig(cseqs, rseqs, cshft)
    t_orig = (time.time() - t0) / iterations

    t0 = time.time()
    for _ in range(iterations):
        _ = dygkt_opt(cseqs, rseqs, cshft)
    t_opt = (time.time() - t0) / iterations
    
    print(f"DyGKT-> Original: {t_orig:.4f}s | Optimized: {t_opt:.4f}s | Speedup: {t_orig/t_opt:.1f}x")

    # 3. DGEKT
    dgekt_orig = DGEKTOriginal(num_c=num_c, adj_matrix=adj)
    dgekt_opt = DGEKTPyTorch(num_c=num_c, adj_matrix=adj)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = dgekt_orig(cseqs, rseqs, cshft)
    t_orig = (time.time() - t0) / iterations

    t0 = time.time()
    for _ in range(iterations):
        _ = dgekt_opt(cseqs, rseqs, cshft)
    t_opt = (time.time() - t0) / iterations
    
    print(f"DGEKT-> Original: {t_orig:.4f}s | Optimized: {t_opt:.4f}s | Speedup: {t_orig/t_opt:.1f}x")

    # 4. GKT (official submodule)
    gkt_orig = GKTOriginal(num_c=num_c, hidden_dim=128, emb_size=64, graph=adj)
    gkt_opt = GKTOptimized(num_c=num_c, hidden_dim=128, emb_size=64, graph=adj)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = gkt_orig(cseqs, rseqs)
    t_orig = (time.time() - t0) / iterations

    t0 = time.time()
    for _ in range(iterations):
        _ = gkt_opt(cseqs, rseqs)
    t_opt = (time.time() - t0) / iterations
    
    print(f"GKT  -> Original: {t_orig:.4f}s | Optimized: {t_opt:.4f}s | Speedup: {t_orig/t_opt:.1f}x")
    print("--------------------------------------------------------------------------------\n")
