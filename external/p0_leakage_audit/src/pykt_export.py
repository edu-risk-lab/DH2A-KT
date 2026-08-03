"""Export P0 parquet splits to pyKT sequence CSV + dense ID maps (train-only vocab)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_dense_maps(train_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Question (item) and KC ids remapped to contiguous indices from **train only**."""
    kcs = sorted(int(x) for x in train_df["kc_id"].unique())
    items = sorted(int(x) for x in train_df["item_id"].unique())
    return ({k: i for i, k in enumerate(items)}, {k: i for i, k in enumerate(kcs)})


def _build_rows(df: pd.DataFrame, fold_val: int, q_map: dict[int, int], c_map: dict[int, int], max_seq_len: int) -> pd.DataFrame:
    df_mapped = df.copy()
    df_mapped["qi"] = df_mapped["item_id"].map(q_map)
    df_mapped["ci"] = df_mapped["kc_id"].map(c_map)
    df_mapped = df_mapped.dropna(subset=["qi", "ci"])
    df_mapped["qi"] = df_mapped["qi"].astype(int)
    df_mapped["ci"] = df_mapped["ci"].astype(int)
    df_mapped = df_mapped[(df_mapped["qi"] >= 0) & (df_mapped["ci"] >= 0)]
    
    if df_mapped.empty:
        return pd.DataFrame(columns=["fold", "uid", "questions", "concepts", "responses", "selectmasks", "timestamps"])
        
    df_mapped = df_mapped.sort_values(["user_id", "timestamp"])
    
    grp = df_mapped.groupby("user_id", sort=False).agg({
        "qi": list,
        "ci": list,
        "correct": list
    })
    
    rows = []
    for uid, r in grp.iterrows():
        q_list = r["qi"]
        c_list = r["ci"]
        correct_list = r["correct"]
        L = len(q_list)
        if L < 2:
            continue
        
        if L > max_seq_len:
            q_list = q_list[-max_seq_len:]
            c_list = c_list[-max_seq_len:]
            correct_list = correct_list[-max_seq_len:]
            L = max_seq_len
            
        pad_n = max_seq_len - L
        
        questions = [str(x) for x in q_list] + ["-1"] * pad_n
        concepts = [str(x) for x in c_list] + ["-1"] * pad_n
        responses = [str(int(x)) for x in correct_list] + ["0"] * pad_n
        smasks = ["1"] * L + ["0"] * pad_n
        timestamps = [str(t) for t in range(L)] + ["-1"] * pad_n
        
        rows.append(
            {
                "fold": fold_val,
                "uid": int(uid),
                "questions": ",".join(questions),
                "concepts": ",".join(concepts),
                "responses": ",".join(responses),
                "selectmasks": ",".join(smasks),
                "timestamps": ",".join(timestamps),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_pykt_csvs(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    q_map: dict[int, int],
    c_map: dict[int, int],
    out_dir: Path,
    max_seq_len: int,
) -> tuple[int, int]:
    """Write ``train_valid_sequences.csv`` (fold 0=train, 1=valid) and ``test_sequences.csv`` (fold -1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tv = pd.concat(
        [_build_rows(train_df, 0, q_map, c_map, max_seq_len), _build_rows(valid_df, 1, q_map, c_map, max_seq_len)],
        ignore_index=True,
    )
    te = _build_rows(test_df, -1, q_map, c_map, max_seq_len)
    tv.to_csv(out_dir / "train_valid_sequences.csv", index=False)
    te.to_csv(out_dir / "test_sequences.csv", index=False)
    
    # Export question-skill bipartite graph for GIKT PyTorch GCN
    train_mapped = train_df.copy()
    train_mapped["qi"] = train_mapped["item_id"].map(q_map)
    train_mapped["ci"] = train_mapped["kc_id"].map(c_map)
    train_mapped = train_mapped.dropna(subset=["qi", "ci"])
    bipartite_df = train_mapped[["qi", "ci"]].astype(int).drop_duplicates()
    bipartite_df.columns = ["question", "concept"]
    bipartite_df.to_csv(out_dir / "gikt_bipartite.csv", index=False)

    num_q = max(q_map.values(), default=-1) + 1
    num_c = max(c_map.values(), default=-1) + 1
    return int(num_q), int(num_c)
