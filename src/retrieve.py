import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import load_config, get_project_root


def load_index_and_order(cfg: dict):
    index_path = get_project_root() / cfg["corpus"]["paths"]["faiss_index"]
    order_path = Path(index_path).with_suffix(".order.parquet")

    index = faiss.read_index(str(index_path))
    order_df = pd.read_parquet(order_path)
    # position i in the FAISS index corresponds to order_df.passage_id[i]
    position_to_passage_id = order_df["passage_id"].tolist()
    return index, position_to_passage_id


def embed_questions(questions_df: pd.DataFrame, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(questions_df)} questions with {model_name}...")
    embeddings = model.encode(
        questions_df["question"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def retrieve_top20(index, query_embeddings: np.ndarray, top_k: int = 20):
    scores, positions = index.search(query_embeddings, top_k)
    return scores, positions


def build_retrieval_records(questions_df, scores, positions, position_to_passage_id):
    records = []
    for i, row in questions_df.reset_index(drop=True).iterrows():
        retrieved_ids = [position_to_passage_id[p] for p in positions[i]]
        retrieved_scores = scores[i].tolist()

        gold_id = row["gold_passage_id"]
        gold_rank = retrieved_ids.index(gold_id) if gold_id in retrieved_ids else -1

        records.append({
            "question_id": row["question_id"],
            "gold_passage_id": gold_id,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "gold_rank": gold_rank,  # 0-indexed position in top-20, or -1 if absent
        })
    return records


def compute_recall_at_k(records, k_values):
    n = len(records)
    recall = {}
    for k in k_values:
        hits = sum(1 for r in records if 0 <= r["gold_rank"] < k)
        recall[k] = hits / n
    return recall


def main():
    cfg = load_config()

    questions_df = pd.read_parquet(get_project_root() / cfg["corpus"]["questions_path"])
    index, position_to_passage_id = load_index_and_order(cfg)

    query_embeddings = embed_questions(questions_df, cfg["retrieval"]["embedding_model"])
    scores, positions = retrieve_top20(
        index, query_embeddings, top_k=cfg["retrieval"]["top_k_retrieved"]
    )

    records = build_retrieval_records(questions_df, scores, positions, position_to_passage_id)

    out_dir = get_project_root() / cfg["paths"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_top20.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved per-question retrieval records to {out_path}")

    recall = compute_recall_at_k(records, cfg["retrieval"]["k_values"])
    print("\nRecall@k")
    print("--------")
    for k, v in recall.items():
        print(f"Recall@{k:>2}: {v:.4f}")

    n_never_retrieved = sum(1 for r in records if r["gold_rank"] == -1)
    print(f"\nGold passage never in top-20 for {n_never_retrieved}/{len(records)} questions "
          f"({n_never_retrieved / len(records):.2%})")

if __name__ == "__main__":
    main()