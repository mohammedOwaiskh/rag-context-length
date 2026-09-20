import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import compute_em_f1, postprocess_generation


def load_retrieval_records(path: str | Path) -> dict:
    """question_id -> retrieval record dict."""
    records = {}
    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            records[r["question_id"]] = r
    return records


def run_generation(
    cfg: dict,
    backend,
    k: int,
    question_ids: list[str],
    out_path: str,
):
    questions_df = pd.read_parquet(cfg["corpus"]["questions_path"]).set_index(
        "question_id", drop=False
    )
    corpus_df = pd.read_parquet(cfg["corpus"]["paths"]["passages"]).set_index(
        "passage_id", drop=False
    )
    retrieval_path = Path(cfg["paths"]["results_dir"]) / "retrieval_top20.jsonl"
    retrieval_records = load_retrieval_records(retrieval_path)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    from generation.prompt_template import build_prompt  # local import avoids circularity

    with open(out_path, "w") as out_f:
        for qid in question_ids:
            q_row = questions_df.loc[qid]
            retrieval = retrieval_records[qid]

            top_k_ids = retrieval["retrieved_ids"][:k]
            passage_texts = [corpus_df.loc[pid, "text"] for pid in top_k_ids]

            prompt = build_prompt(q_row["question"], passage_texts)
            context_token_count = backend.count_tokens("\n\n".join(passage_texts))

            raw_answer, gen_time = backend.generate(prompt)
            generated_answer = postprocess_generation(raw_answer)

            gold_answers = list(q_row["gold_answers"])
            em, f1 = compute_em_f1(generated_answer, gold_answers)

            gold_rank = retrieval["gold_rank"]
            gold_retrieved = 0 <= gold_rank < k

            record = {
                "question_id": qid,
                "question": q_row["question"],
                "gold_answer": gold_answers,
                "gold_passage_id": retrieval["gold_passage_id"],
                "k": k,
                "retrieved_ids": top_k_ids,
                "retrieved_scores": retrieval["retrieved_scores"][:k],
                "gold_rank": gold_rank,
                "gold_retrieved": gold_retrieved,
                "context": "\n\n".join(passage_texts),
                "context_token_count": context_token_count,
                "generated_answer_raw": raw_answer,
                "generated_answer": generated_answer,
                "exact_match": em,
                "f1": f1,
                "generation_time": gen_time,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()  # ensure resume-safety even on hard crash

    print(f"Wrote {len(question_ids)} records to {out_path}")