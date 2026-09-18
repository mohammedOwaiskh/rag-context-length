from pathlib import Path

import pandas as pd
from datasets import load_dataset

from utils import load_config, passage_id, get_project_root


def sample_questions(cfg: dict) -> pd.DataFrame:
    ds = load_dataset("rajpurkar/squad", split=cfg["dataset"]["hf_split"])
    ds = ds.shuffle(seed=cfg["sampling"]["seed"])
    n = cfg["sampling"]["n_questions"]
    subset = ds.select(range(n))

    rows = []
    for row in subset:
        gold_answers = row["answers"]["text"]
        rows.append({
            "question_id": row["id"],
            "question": row["question"],
            "gold_answers": gold_answers,
            "gold_passage_id": passage_id(row["context"]),
        })

    df = pd.DataFrame(rows)
    print(f"Sampled {len(df)} questions (seed={cfg['sampling']['seed']})")
    return df


def verify_gold_linkage(questions_df: pd.DataFrame, corpus_path: str):
    """Every gold_passage_id must exist in the corpus, or Recall@k is meaningless."""
    corpus_df = pd.read_parquet(corpus_path)
    corpus_ids = set(corpus_df["passage_id"])
    missing = ~questions_df["gold_passage_id"].isin(corpus_ids)
    n_missing = missing.sum()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} questions have a gold_passage_id not present in the "
            f"corpus. Run build_corpus.py from the same dataset/split first."
        )
    print("Gold passage linkage verified: all questions map to a corpus passage.")


def main():
    cfg = load_config()

    out_path = get_project_root() / cfg["corpus"]["questions_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = sample_questions(cfg)
    verify_gold_linkage(df, get_project_root() / cfg["corpus"]["paths"]["passages"])

    df.to_parquet(out_path, index=False)
    print(f"Saved sampled questions to {out_path}")


if __name__ == "__main__":
    main()