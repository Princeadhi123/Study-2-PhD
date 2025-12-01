from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DERIVED_FEATURES_PATH = BASE_DIR / "diagnostics" / "cluster input features" / "derived_features.csv"
STUDENT_CLUSTERS_PATH = BASE_DIR / "diagnostics" / "student cluster labels" / "student_clusters.csv"
MARKS_WITH_CLUSTERS_PATH = BASE_DIR / "diagnostics" / "cluster input features" / "marks_with_clusters.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

EMBEDDING_BACKEND = "sentence-transformers"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

K_RANGE = range(2, 11)
