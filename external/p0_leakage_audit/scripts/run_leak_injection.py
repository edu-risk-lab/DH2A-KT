import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.io_utils import load_yaml, load_interactions
from src.split_checker import learner_based_folds
from src.graph_builder import build_q_matrix_from_train, infer_prerequisites_from_train
from src.leakage_metrics import compute_leakage_row

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    cfg = load_yaml(Path("configs/xes3g5m.yaml"))
    dataset = cfg["dataset"]
    df = load_interactions(Path(f"data/processed/{dataset}.parquet"))
    
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.7, 0.1, 0.2]))
    graph_cfg = cfg.get("graph", {})
    
    # Run only for fold 0
    fold, split_seed, splits = next(learner_based_folds(df, ratios, cfg.get("split", {}), default_seed=42))
    
    train_base = splits["train"].copy()
    train_base["split"] = "train"
    train_base["fold"] = fold
    
    test_base = splits["test"].copy()
    test_base["split"] = "test"
    
    q_train = build_q_matrix_from_train(train_base)
    
    injection_rates = [0.0, 0.05, 0.20]
    results = []
    
    for p in injection_rates:
        logging.info(f"Running injection with p={p}")
        
        if p > 0:
            n_inject = int(len(test_base) * p)
            injected_test = test_base.sample(n=n_inject, random_state=42 + int(p*100))
            train_injected = pd.concat([train_base, injected_test], ignore_index=True)
            # Must set split to train so assertion passes
            train_injected["split"] = "train"
        else:
            train_injected = train_base.copy()
            
        pre = infer_prerequisites_from_train(
            train_injected,
            q_train,
            max_edges=int(graph_cfg.get("e_pre_max_edges", 5000)),
            top_k_per_node=int(graph_cfg.get("e_pre_top_k_per_node", 10)),
            support_quantile=float(graph_cfg.get("e_pre_support_quantile", 0.90)),
        )
        
        sim = pd.DataFrame(columns=["src_kc", "dst_kc", "weight", "source"]) # Empty sim for injection test
        
        # Compute leakage metrics
        leakage = compute_leakage_row(
            dataset=dataset,
            fold=fold,
            splits=splits, # We pass original splits so the evaluation knows what the true test set is!
            pre_df=pre,
            sim_df=sim,
            q_train=q_train,
            train_ratio=ratios[0]
        )
        
        results.append({
            "injection_rate": f"{int(p*100)}%",
            "E_pre_edges": len(pre),
            "ECR_overlap": leakage["ecr_overlap"],
            "ECR_flag": leakage["ecr_flag"],
            "TBMR": leakage["tbvr"]
        })
        
    df_res = pd.DataFrame(results)
    
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df_res.to_csv("results/tables/leak_injection.csv", index=False)
    
    tex_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        ("\\textbf{Injection rate} & $|\\Epre|$ & $\\mathrm{ECR}^{\\mathrm{overlap}}$"
         " & $\\mathrm{ECR}^{\\mathrm{flag}}$ & $\\mathrm{TBMR}$ \\\\"),
        "\\midrule",
    ]
    for r in results:
        rate = r["injection_rate"].replace("%", "\\%")
        tex_lines.append(
            f"{rate} & {r['E_pre_edges']} & {r['ECR_overlap']:.3f}"
            f" & {r['ECR_flag']:.3f} & {r['TBMR']:.3f} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open("results/tables/leak_injection.tex", "w") as f:
        f.write("\n".join(tex_lines))

if __name__ == "__main__":
    main()
