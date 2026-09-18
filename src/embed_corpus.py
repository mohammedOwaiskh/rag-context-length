import yaml
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import load_config, get_project_root


def embed_passages(df: pd.DataFrame, model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(df)} passages with {model_name} (batch_size={batch_size})...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize -> inner product == cosine
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS IndexFlatIP built: {index.ntotal} vectors, dim={dim}")
    return index


def main():
    cfg = load_config()
    corpus_path = get_project_root() / cfg["corpus"]["paths"]["passages"]
    index_path = get_project_root() / cfg["corpus"]["paths"]["faiss_index"]

    df = pd.read_parquet(corpus_path)
    embeddings = embed_passages(
        df,
        model_name=cfg["retrieval"]["embedding_model"],
        batch_size=cfg["retrieval"]["embedding_batch_size"],
    )

    index = build_faiss_index(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    # Save the row order the index was built with, so retrieve.py can map
    # FAISS integer positions back to passage_id unambiguously.
    order_path = Path(index_path).with_suffix(".order.parquet")
    df[["passage_id"]].reset_index(drop=True).to_parquet(order_path, index=False)
    print(f"Saved index row order to {order_path}")


if __name__ == "__main__":
    main()