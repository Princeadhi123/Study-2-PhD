import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL_NAME,
    NARRATIVE_TEMPLATE_VERSION,
    OUTPUT_DIR,
    make_versioned_filename,
)


def load_model():
    if EMBEDDING_BACKEND == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required for this script. Install it with:\n"
                "pip install sentence-transformers"
            ) from exc

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        print(f"Using device: {device}")
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    raise SystemExit(f"Unsupported embedding backend: {EMBEDDING_BACKEND}")


def main() -> None:
    narratives_filename = make_versioned_filename("narratives.csv")
    narratives_path = OUTPUT_DIR / narratives_filename
    if not narratives_path.exists():
        raise SystemExit(f"Missing narratives file: {narratives_path}")
    df = pd.read_csv(narratives_path)
    texts = df["narrative_text"].astype(str).tolist()

    model = load_model()
    
    # Use batch size from env or default to 32
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    print(f"Computing embeddings with batch_size={batch_size}...")
    
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_filename = make_versioned_filename("embeddings.npy")
    emb_path = OUTPUT_DIR / emb_filename
    np.save(emb_path, embeddings)

    index_filename = make_versioned_filename("embeddings_index.csv")
    index_path = OUTPUT_DIR / index_filename
    index_df = df[["IDCode"]].copy()
    index_df["index"] = np.arange(len(index_df))
    index_df.to_csv(index_path, index=False)

    print(f"Saved embeddings (template {NARRATIVE_TEMPLATE_VERSION.upper()}) to {emb_path}")
    print(f"Saved embedding index (template {NARRATIVE_TEMPLATE_VERSION.upper()}) to {index_path}")


if __name__ == "__main__":
    main()
