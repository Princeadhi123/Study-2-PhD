import os
import sys
from pathlib import Path
import subprocess


TEMPLATES = ("A", "B")
MODELS = ("all-mpnet-base-v2", "all-MiniLM-L6-v2")
SCRIPTS = (
    "01_build_narratives.py",
    "02_compute_embeddings.py",
    "03_cluster_embeddings.py",
    "04_compare_with_gmm_bic.py",
    "05_visualize_embeddings.py",
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    for template in TEMPLATES:
        for model in MODELS:
            print("==> Running pipeline for template", template, "and model", model)

            env = os.environ.copy()
            env["NARRATIVE_TEMPLATE_VERSION"] = template
            env["EMBEDDING_MODEL_NAME"] = model

            for script in SCRIPTS:
                script_path = base_dir / script
                print(f"[template {template} | model {model}] Running {script}")
                subprocess.run([sys.executable, str(script_path)], check=True, env=env)


if __name__ == "__main__":
    main()
