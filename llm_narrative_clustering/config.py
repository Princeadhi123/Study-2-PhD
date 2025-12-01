from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

DERIVED_FEATURES_PATH = BASE_DIR / "diagnostics" / "cluster input features" / "derived_features.csv"
STUDENT_CLUSTERS_PATH = BASE_DIR / "diagnostics" / "student cluster labels" / "student_clusters.csv"
MARKS_WITH_CLUSTERS_PATH = BASE_DIR / "diagnostics" / "cluster input features" / "marks_with_clusters.csv"

EMBEDDING_BACKEND = "sentence-transformers"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

K_RANGE = range(2, 11)

NARRATIVE_TEMPLATE_VERSION = os.getenv("NARRATIVE_TEMPLATE_VERSION", "A")

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / f"template_{NARRATIVE_TEMPLATE_VERSION.upper()}"

def make_versioned_filename(base: str) -> str:
    version = NARRATIVE_TEMPLATE_VERSION.upper()
    if version == "A":
        return base
    p = Path(base)
    stem = p.stem
    suffix = p.suffix
    return f"{stem}_{version}{suffix}"
