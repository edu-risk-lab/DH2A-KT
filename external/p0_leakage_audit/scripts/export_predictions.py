import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def export_all():
    models = ["simplekt", "gkt", "gikt"]
    folds = [0, 1, 2]
    dataset = "xes3g5m"
    
    for fold in folds:
        for model in models:
            parquet_path = Path("results/predictions") / dataset / f"fold_{fold}" / f"{model}.parquet"
            if parquet_path.exists():
                logging.info(f"Skipping {model} fold {fold}, {parquet_path} already exists.")
                continue
            
            logging.info(f"==== Exporting full predictions for {model} fold {fold} ====")
            cmd = [
                "python", "-m", "src.baseline_runner",
                "--config", f"configs/{dataset}.yaml",
                "--baseline-backend", "pykt",
                "--export-full-predictions", model,
                "--fold-idx", str(fold)
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to export {model} fold {fold}. Exit code: {e.returncode}")
                # We do not exit immediately to allow other models to try
                
if __name__ == "__main__":
    export_all()
