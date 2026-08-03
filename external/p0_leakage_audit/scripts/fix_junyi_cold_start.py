import pandas as pd
from pathlib import Path
from src.io_utils import load_yaml, load_interactions, dump_csv
from src.baseline_runner import _run_named_model, _cold_start_rows, MODEL_WEIGHTS
from src.split_checker import learner_based_folds
from src.cold_start_report import bin_kcs_by_frequency

def main():
    dataset = "junyi"
    df = load_interactions(Path(f"data/processed/{dataset}.parquet"))
    split_cfg = {"type": "learner_temporal", "ratios": [0.7, 0.1, 0.2], "seed": 42, "n_folds": 3}
    
    models = ["skt", "dygkt", "dgekt"]
    cold_frames = []
    
    for fold, split_seed, splits in learner_based_folds(df, [0.7, 0.1, 0.2], split_cfg, default_seed=42):
        strata_once = bin_kcs_by_frequency(splits["train"])
        for model in models:
            print(f"Running {model} for fold {fold}...")
            result, predictions = _run_named_model(
                model,
                splits,
                dataset=dataset,
                fold=fold,
                split_seed=split_seed,
                graph_construction="train_only",
                prediction_cap=None,
                full_interactions=df,
                experiment_seed=42
            )
            cold = _cold_start_rows(dataset, fold, split_seed, splits["train"], predictions, strata=strata_once)
            if not cold.empty:
                cold_frames.append(cold)
                
    if cold_frames:
        new_cold = pd.concat(cold_frames, ignore_index=True)
        csv_path = Path("results/tables/cold_start_metrics.csv")
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            # Remove any existing rows for these models on junyi
            existing = existing[~((existing["dataset"] == dataset) & (existing["model"].isin(models)))]
            final_df = pd.concat([existing, new_cold], ignore_index=True)
        else:
            final_df = new_cold
        dump_csv(final_df, csv_path)
        print(f"Appended {len(new_cold)} rows to {csv_path}")

if __name__ == "__main__":
    main()
