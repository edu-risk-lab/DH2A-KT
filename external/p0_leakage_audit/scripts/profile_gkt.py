# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cpu"

graph_npz_path = "results/pykt_work/xes3g5m/fold_0_seed_42/train_only/gkt_graph_p0_protocol.npz"
graph_data = np.load(graph_npz_path, allow_pickle=True)
graph_matrix = torch.tensor(graph_data['matrix']).float().to(device)
num_c = graph_matrix.shape[0]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)

class EraseAddGate(nn.Module):
    def __init__(self, feature_dim, num_c, bias=True):
        super(EraseAddGate, self).__init__()
        self.weight = nn.Parameter(torch.rand(num_c))
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        erase_gate = torch.sigmoid(self.erase(x))
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res

class CurrentGKT(nn.Module):
    def __init__(self, num_c, hidden_dim, emb_size, graph, dropout=0.5, emb_type="qid", emb_path="", bias=True):
        super(CurrentGKT, self).__init__()
        self.model_name = "gkt"
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2
        
        self.graph = nn.Parameter(graph)
        self.graph.requires_grad = False
        
        self.emb_type = emb_type
        self.emb_path = emb_path
        
        if emb_type.startswith("qid"):
            self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
            self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)

        self.f_neighbor_list = nn.ModuleList()
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

        # Precompute forward neighbors
        out_degrees = (graph > 0).sum(dim=1)
        K_out = int(out_degrees.max().item())
        K_out = max(1, K_out)
        
        forward_indices = torch.full((num_c + 1, K_out), num_c, dtype=torch.long, device=device)
        forward_weights = torch.zeros((num_c + 1, K_out), dtype=torch.float, device=device)
        
        for i in range(num_c):
            row = graph[i]
            nbr_idx = torch.nonzero(row > 0).squeeze(dim=-1)
            n_nbr = nbr_idx.numel()
            if n_nbr > 0:
                forward_indices[i, :n_nbr] = nbr_idx
                forward_weights[i, :n_nbr] = row[nbr_idx]
                
        self.register_buffer("forward_indices", forward_indices)
        self.register_buffer("forward_weights", forward_weights)

        # Precompute reverse neighbors
        in_degrees = (graph > 0).sum(dim=0)
        K_in = int(in_degrees.max().item())
        K_in = max(1, K_in)
        
        reverse_indices = torch.full((num_c + 1, K_in), num_c, dtype=torch.long, device=device)
        reverse_weights = torch.zeros((num_c + 1, K_in), dtype=torch.float, device=device)
        
        for i in range(num_c):
            col = graph[:, i]
            nbr_idx = torch.nonzero(col > 0).squeeze(dim=-1)
            n_nbr = nbr_idx.numel()
            if n_nbr > 0:
                reverse_indices[i, :n_nbr] = nbr_idx
                reverse_weights[i, :n_nbr] = col[nbr_idx]
                
        self.register_buffer("reverse_indices", reverse_indices)
        self.register_buffer("reverse_weights", reverse_weights)

    def _aggregate(self, xt, qt, ht, batch_size):
        qt_mask = torch.ne(qt, -1)
        res_embedding = self.interaction_emb(xt[qt_mask])
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.num_c * torch.ones((batch_size, self.num_c)).to(device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.num_c).to(device)
        concept_embedding = self.emb_c(concept_idx_mat)

        index_tuple = (torch.arange(mask_num).to(device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)
        return tmp_ht

    def _agg_neighbors(self, tmp_ht, qt):
        qt_mask = torch.ne(qt, -1)
        masked_qt = qt[qt_mask]
        masked_tmp_ht = tmp_ht[qt_mask]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num).to(device), masked_qt.long())
        
        self_ht = masked_tmp_ht[self_index_tuple]
        self_features = self.f_self(self_ht)
        concept_embedding, rec_embedding, z_prob = None, None, None

        D = tmp_ht.size(-1)
        padding_row = tmp_ht.new_zeros((mask_num, 1, D))
        masked_tmp_ht_padded = torch.cat((masked_tmp_ht, padding_row), dim=1)

        # Forward neighbors
        batch_fwd_indices = self.forward_indices[masked_qt.long()]
        batch_fwd_weights = self.forward_weights[masked_qt.long()]
        
        gather_index = batch_fwd_indices.unsqueeze(-1).expand(-1, -1, D)
        neighbor_ht = torch.gather(masked_tmp_ht_padded, 1, gather_index)
        
        K_out = batch_fwd_indices.size(1)
        self_ht_expanded = self_ht.unsqueeze(1).expand(-1, K_out, -1)
        neigh_ht_active = torch.cat((self_ht_expanded, neighbor_ht), dim=-1)
        
        features_active = self.f_neighbor_list[0](neigh_ht_active.view(-1, 2 * D))
        features_active = features_active.view(mask_num, K_out, self.hidden_dim)
        neigh_features_weighted = features_active * batch_fwd_weights.unsqueeze(-1)
        
        neigh_features_forward = torch.zeros((mask_num, self.num_c + 1, self.hidden_dim)).to(device).type(tmp_ht.dtype)
        scatter_index = batch_fwd_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        neigh_features_forward.scatter_add_(dim=1, index=scatter_index, src=neigh_features_weighted)
        neigh_features_forward = neigh_features_forward[:, :self.num_c, :]

        # Reverse neighbors
        batch_rev_indices = self.reverse_indices[masked_qt.long()]
        batch_rev_weights = self.reverse_weights[masked_qt.long()]
        
        gather_index_rev = batch_rev_indices.unsqueeze(-1).expand(-1, -1, D)
        neighbor_ht_rev = torch.gather(masked_tmp_ht_padded, 1, gather_index_rev)
        
        K_in = batch_rev_indices.size(1)
        self_ht_expanded_rev = self_ht.unsqueeze(1).expand(-1, K_in, -1)
        neigh_ht_active_rev = torch.cat((self_ht_expanded_rev, neighbor_ht_rev), dim=-1)
        
        features_active_rev = self.f_neighbor_list[1](neigh_ht_active_rev.view(-1, 2 * D))
        features_active_rev = features_active_rev.view(mask_num, K_in, self.hidden_dim)
        neigh_features_weighted_rev = features_active_rev * batch_rev_weights.unsqueeze(-1)
        
        neigh_features_reverse = torch.zeros((mask_num, self.num_c + 1, self.hidden_dim)).to(device).type(tmp_ht.dtype)
        scatter_index_rev = batch_rev_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        neigh_features_reverse.scatter_add_(dim=1, index=scatter_index_rev, src=neigh_features_weighted_rev)
        neigh_features_reverse = neigh_features_reverse[:, :self.num_c, :]

        neigh_features = neigh_features_forward + neigh_features_reverse

        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        
        return m_next, concept_embedding, rec_embedding, z_prob

    def _update(self, tmp_ht, ht, qt):
        qt_mask = torch.ne(qt, -1)
        mask_num = qt_mask.nonzero().shape[0]
        
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)
        
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])
        
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))
        index_tuple = (torch.arange(mask_num).to(device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.num_c, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    def _predict(self, h_next, qt):
        qt_mask = torch.ne(qt, -1)
        y = self.predict(h_next).squeeze(dim=-1)
        y[qt_mask] = torch.sigmoid(y[qt_mask])
        return y

    def _get_next_pred(self, yt, q_next):
        next_qt = q_next
        valid_mask = (next_qt != -1)
        safe_qt = torch.where(valid_mask, next_qt, torch.zeros_like(next_qt))
        pred = yt.gather(1, safe_qt.long().unsqueeze(1)).squeeze(1)
        pred = torch.where(valid_mask, pred, torch.zeros_like(pred))
        return pred

    def forward(self, q, r):
        features = q * 2 + r
        questions = q
        
        batch_size, seq_len = features.shape
        ht = torch.zeros((batch_size, self.num_c, self.hidden_dim)).to(device)
        
        pred_list = []
        for i in range(seq_len):
            xt = features[:, i]
            qt = questions[:, i]
            qt_mask = torch.ne(qt, -1)
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)
            ht[qt_mask] = h_next[qt_mask]
            yt = self._predict(h_next, qt)
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)
        return pred_res

class ProposedGKT(nn.Module):
    def __init__(self, num_c, hidden_dim, emb_size, graph, dropout=0.5, emb_type="qid", emb_path="", bias=True):
        super(ProposedGKT, self).__init__()
        self.model_name = "gkt"
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2
        self.graph = nn.Parameter(graph)
        self.graph.requires_grad = False
        self.emb_type = emb_type
        self.emb_path = emb_path
        
        if emb_type.startswith("qid"):
            self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
            self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)

        self.f_neighbor_list = nn.ModuleList()
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

        # Precompute forward neighbors
        out_degrees = (graph > 0).sum(dim=1)
        K_out = int(out_degrees.max().item())
        K_out = max(1, K_out)
        
        forward_indices = torch.full((num_c + 1, K_out), num_c, dtype=torch.long, device=device)
        forward_weights = torch.zeros((num_c + 1, K_out), dtype=torch.float, device=device)
        
        for i in range(num_c):
            row = graph[i]
            nbr_idx = torch.nonzero(row > 0).squeeze(dim=-1)
            n_nbr = nbr_idx.numel()
            if n_nbr > 0:
                forward_indices[i, :n_nbr] = nbr_idx
                forward_weights[i, :n_nbr] = row[nbr_idx]
                
        self.register_buffer("forward_indices", forward_indices)
        self.register_buffer("forward_weights", forward_weights)

        # Precompute reverse neighbors
        in_degrees = (graph > 0).sum(dim=0)
        K_in = int(in_degrees.max().item())
        K_in = max(1, K_in)
        
        reverse_indices = torch.full((num_c + 1, K_in), num_c, dtype=torch.long, device=device)
        reverse_weights = torch.zeros((num_c + 1, K_in), dtype=torch.float, device=device)
        
        for i in range(num_c):
            col = graph[:, i]
            nbr_idx = torch.nonzero(col > 0).squeeze(dim=-1)
            n_nbr = nbr_idx.numel()
            if n_nbr > 0:
                reverse_indices[i, :n_nbr] = nbr_idx
                reverse_weights[i, :n_nbr] = col[nbr_idx]
                
        self.register_buffer("reverse_indices", reverse_indices)
        self.register_buffer("reverse_weights", reverse_weights)

    def _aggregate(self, xt, qt, ht, batch_size):
        qt_mask = torch.ne(qt, -1)
        active_student_indices = torch.where(qt_mask)[0]
        mask_num = active_student_indices.shape[0]

        # Initialize concept_embedding for the entire batch with default padding embedding at index num_c
        pad_emb = self.emb_c(torch.tensor([self.num_c], device=device))  # [1, emb_size]
        concept_embedding = pad_emb.unsqueeze(0).expand(batch_size, self.num_c, -1).clone()

        if mask_num > 0:
            concept_emb_active = self.emb_c(torch.arange(self.num_c, device=device))
            concept_embedding[active_student_indices] = concept_emb_active.unsqueeze(0).expand(mask_num, -1, -1)
            
            res_embedding = self.interaction_emb(xt[qt_mask])
            concept_embedding[active_student_indices, qt[qt_mask].long()] = res_embedding

        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)
        return tmp_ht

    def _agg_neighbors(self, tmp_ht, qt):
        qt_mask = torch.ne(qt, -1)
        active_student_indices = torch.where(qt_mask)[0]
        masked_qt = qt[qt_mask]
        masked_tmp_ht = tmp_ht[qt_mask]
        mask_num = masked_tmp_ht.shape[0]
        
        m_next = tmp_ht[:, :, :self.hidden_dim].clone()
        concept_embedding, rec_embedding, z_prob = None, None, None

        if mask_num > 0:
            self_index_tuple = (torch.arange(mask_num).to(device), masked_qt.long())
            self_ht = masked_tmp_ht[self_index_tuple]
            self_features = self.f_self(self_ht)

            D = tmp_ht.size(-1)
            padding_row = tmp_ht.new_zeros((mask_num, 1, D))
            masked_tmp_ht_padded = torch.cat((masked_tmp_ht, padding_row), dim=1)

            # Forward neighbors
            batch_fwd_indices = self.forward_indices[masked_qt.long()]
            batch_fwd_weights = self.forward_weights[masked_qt.long()]
            
            gather_index = batch_fwd_indices.unsqueeze(-1).expand(-1, -1, D)
            neighbor_ht = torch.gather(masked_tmp_ht_padded, 1, gather_index)
            
            K_out = batch_fwd_indices.size(1)
            self_ht_expanded = self_ht.unsqueeze(1).expand(-1, K_out, -1)
            neigh_ht_active = torch.cat((self_ht_expanded, neighbor_ht), dim=-1)
            
            features_active = self.f_neighbor_list[0](neigh_ht_active.view(-1, 2 * D))
            features_active = features_active.view(mask_num, K_out, self.hidden_dim)
            neigh_features_weighted = features_active * batch_fwd_weights.unsqueeze(-1)
            
            neigh_features_forward = torch.zeros((mask_num, self.num_c + 1, self.hidden_dim), device=device, dtype=tmp_ht.dtype)
            scatter_index = batch_fwd_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            neigh_features_forward.scatter_add_(dim=1, index=scatter_index, src=neigh_features_weighted)
            neigh_features_forward = neigh_features_forward[:, :self.num_c, :]

            # Reverse neighbors
            batch_rev_indices = self.reverse_indices[masked_qt.long()]
            batch_rev_weights = self.reverse_weights[masked_qt.long()]
            
            gather_index_rev = batch_rev_indices.unsqueeze(-1).expand(-1, -1, D)
            neighbor_ht_rev = torch.gather(masked_tmp_ht_padded, 1, gather_index_rev)
            
            K_in = batch_rev_indices.size(1)
            self_ht_expanded_rev = self_ht.unsqueeze(1).expand(-1, K_in, -1)
            neigh_ht_active_rev = torch.cat((self_ht_expanded_rev, neighbor_ht_rev), dim=-1)
            
            features_active_rev = self.f_neighbor_list[1](neigh_ht_active_rev.view(-1, 2 * D))
            features_active_rev = features_active_rev.view(mask_num, K_in, self.hidden_dim)
            neigh_features_weighted_rev = features_active_rev * batch_rev_weights.unsqueeze(-1)
            
            neigh_features_reverse = torch.zeros((mask_num, self.num_c + 1, self.hidden_dim), device=device, dtype=tmp_ht.dtype)
            scatter_index_rev = batch_rev_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            neigh_features_reverse.scatter_add_(dim=1, index=scatter_index_rev, src=neigh_features_weighted_rev)
            neigh_features_reverse = neigh_features_reverse[:, :self.num_c, :]

            neigh_features = neigh_features_forward + neigh_features_reverse

            m_next[active_student_indices] = neigh_features
            m_next[active_student_indices, masked_qt.long()] = self_features
            
        return m_next, concept_embedding, rec_embedding, z_prob

    def _update(self, tmp_ht, ht, qt):
        qt_mask = torch.ne(qt, -1)
        active_student_indices = torch.where(qt_mask)[0]
        mask_num = active_student_indices.shape[0]
        
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)
        h_next = m_next
        
        if mask_num > 0:
            m_next_active = self.erase_add_gate(m_next[active_student_indices])
            res = self.gru(
                m_next_active.reshape(-1, self.hidden_dim), 
                ht[active_student_indices].reshape(-1, self.hidden_dim)
            )
            h_next[active_student_indices] = res.reshape(-1, self.num_c, self.hidden_dim)
            
        return h_next, concept_embedding, rec_embedding, z_prob

    def _predict(self, h_next, qt):
        y = self.predict(h_next).squeeze(dim=-1)
        return torch.sigmoid(y)

    def _get_next_pred(self, yt, q_next):
        next_qt = q_next
        valid_mask = (next_qt != -1)
        safe_qt = torch.where(valid_mask, next_qt, torch.zeros_like(next_qt))
        pred = yt.gather(1, safe_qt.long().unsqueeze(1)).squeeze(1)
        pred = torch.where(valid_mask, pred, torch.zeros_like(pred))
        return pred

    def forward(self, q, r):
        features = q * 2 + r
        questions = q
        
        batch_size, seq_len = features.shape
        ht = torch.zeros((batch_size, self.num_c, self.hidden_dim), device=device)
        
        pred_list = []
        for i in range(seq_len):
            xt = features[:, i]
            qt = questions[:, i]
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)
            ht = h_next
            yt = self._predict(h_next, qt)
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)
        return pred_res

# Tracing code
torch.manual_seed(42)
batch_size = 16
seq_len = 5

q = torch.randint(-1, num_c, (batch_size, seq_len), device=device)
q[:, 0] = torch.randint(0, num_c, (batch_size,), device=device)
for b in range(batch_size):
    stop_idx = torch.randint(2, seq_len + 1, (1,)).item()
    if stop_idx < seq_len:
        q[b, stop_idx:] = -1

r = torch.randint(0, 2, (batch_size, seq_len), device=device)
r = torch.where(q == -1, -1, r)

curr_model = CurrentGKT(num_c, hidden_dim=100, emb_size=100, graph=graph_matrix).to(device)
prop_model = ProposedGKT(num_c, hidden_dim=100, emb_size=100, graph=graph_matrix).to(device)
prop_model.load_state_dict(curr_model.state_dict())

curr_model.eval()
prop_model.eval()

with torch.no_grad():
    y_curr = curr_model(q, r)
    y_prop = prop_model(q, r)

# Verify identity
diff = torch.abs(y_curr - y_prop).max().item()
is_identical = torch.allclose(y_curr, y_prop, atol=1e-5, rtol=1e-5)
print(f"Loop check: {'PASSED' if is_identical else 'FAILED'} (max absolute difference: {diff:.6e})")

import time
# Profile speed with seq_len = 200 to match real training sequences
seq_len = 200
q_prof = torch.randint(-1, num_c, (batch_size, seq_len), device=device)
q_prof[:, 0] = torch.randint(0, num_c, (batch_size,), device=device)
for b in range(batch_size):
    stop_idx = torch.randint(seq_len // 2, seq_len, (1,)).item()
    q_prof[b, stop_idx:] = -1

r_prof = torch.randint(0, 2, (batch_size, seq_len), device=device)
r_prof = torch.where(q_prof == -1, -1, r_prof)

n_runs = 10
# Current version profiling
start = time.time()
for _ in range(n_runs):
    y = curr_model(q_prof, r_prof)
curr_time = (time.time() - start) / n_runs
print(f"Current version time per batch forward: {curr_time * 1000:.2f} ms")

# Proposed version profiling
start = time.time()
for _ in range(n_runs):
    y = prop_model(q_prof, r_prof)
prop_time = (time.time() - start) / n_runs
print(f"Proposed version time per batch forward: {prop_time * 1000:.2f} ms")

speedup = (curr_time - prop_time) / curr_time * 100
print(f"Speedup: {speedup:.2f}% (Proposed is {curr_time / prop_time:.2f}x faster)")
