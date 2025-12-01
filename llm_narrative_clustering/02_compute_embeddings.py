from pathlib import Path

import numpy as np
import pandas as pd

from config import EMBEDDING_BACKEND, EMBEDDING_MODEL_NAME, OUTPUT_DIR


def load_model():
    if EMBEDDING_BACKEND == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required for this script. Install it with:\n"
                "pip install sentence-transformers"
            ) from exc
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    raise SystemExit(f"Unsupported embedding backend: {EMBEDDING_BACKEND}")


def main() -> None:
    narratives_path = OUTPUT_DIR / "narratives.csv"
    if not narratives_path.exists():
        raise SystemExit(f"Missing narratives file: {narratives_path}")
    df = pd.read_csv(narratives_path)
    texts = df["narrative_text"].astype(str).tolist()

    model = load_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = OUTPUT_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)

    index_path = OUTPUT_DIR / "embeddings_index.csv"
    index_df = df[["IDCode"]].copy()
    index_df["index"] = np.arange(len(index_df))
    index_df.to_csv(index_path, index=False)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved embedding index to {index_path}")


if __name__ == "__main__":
    main()
