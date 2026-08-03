import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_script(script_path: str):
    logging.info(f"==== Starting {script_path} ====")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        logging.info(f"==== Finished {script_path} ====\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"==== FAILED {script_path} with exit code {e.returncode} ====")
        sys.exit(e.returncode)

if __name__ == "__main__":
    scripts = [
        "scripts/export_predictions.py",
        "scripts/bootstrap_auc_ci.py",
        "scripts/run_injection_auc.py"
    ]
    for s in scripts:
        run_script(s)
    
    logging.info("==== ALL TASKS COMPLETED SUCCESSFULLY ====")
