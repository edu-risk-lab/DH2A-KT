import pandas as pd
import torch
from src.pykt_export import build_dense_maps, dataframe_to_pykt_csvs
from src.models.gikt import GIKTPyTorch


def test_gikt_bipartite_export_and_model(tmp_path) -> None:
    # 1. Test GIKT bipartite CSV export
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 11, 10, 12],
            "kc_id": [5, 6, 5, 7],
            "timestamp": [1, 2, 1, 2],
            "correct": [1, 0, 1, 1],
        }
    )
    valid = train.iloc[:2].copy()
    test = train.iloc[2:].copy()
    qm, cm = build_dense_maps(train)
    
    nq, nc = dataframe_to_pykt_csvs(
        train_df=train,
        valid_df=valid,
        test_df=test,
        q_map=qm,
        c_map=cm,
        out_dir=tmp_path,
        max_seq_len=10,
    )
    
    bip_path = tmp_path / "gikt_bipartite.csv"
    assert bip_path.exists()
    
    # Read bipartite edges
    bipartite_edges = []
    import csv
    with open(bip_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            bipartite_edges.append((int(row[0]), int(row[1])))
            
    assert len(bipartite_edges) > 0

    # 2. Test GIKT PyTorch model initialization
    model = GIKTPyTorch(
        num_q=nq,
        num_c=nc,
        emb_size=16,
        hidden_dim=32,
        bipartite_edges=bipartite_edges,
    )
    
    # 3. Test forward pass
    # batch_size=2, seq_len=5
    qseqs = torch.randint(0, nq, (2, 5))
    cseqs = torch.randint(0, nc, (2, 5))
    rseqs = torch.randint(0, 2, (2, 5))
    
    probs = model(qseqs, cseqs, rseqs)
    assert probs.shape == (2, 5)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    
    # 4. Test backpropagation / gradient calculation
    loss = probs.sum()
    loss.backward()
    
    # Check if gradients are propagated to embeddings
    assert model.q_embed.weight.grad is not None
    assert model.c_embed.weight.grad is not None
    assert model.gcn_linear.weight.grad is not None
