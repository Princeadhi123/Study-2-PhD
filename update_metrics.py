import os
import subprocess
import sys
from pathlib import Path

TEMPLATES = ["A", "B", "C"]
MODELS = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
BASE_DIR = Path("narrative_embedding_clustering")
SCRIPT = "04_compare_with_gmm_bic.py"

def main():
    for template in TEMPLATES:
        for model in MODELS:
            print(f"Updating metrics for Template {template}, Model {model}...")
            env = os.environ.copy()
            env["NARRATIVE_TEMPLATE_VERSION"] = template
            env["EMBEDDING_MODEL_NAME"] = model
            
            subprocess.run([sys.executable, str(BASE_DIR / SCRIPT)], check=True, env=env)

if __name__ == "__main__":
    main()
