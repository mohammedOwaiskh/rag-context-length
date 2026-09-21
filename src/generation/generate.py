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


def load_already_done(out_path: str | Path) -> set:
    """question_ids already written to out_path, so a resumed run skips them."""
    done = set()
    if Path(out_path).exists():
        with open(out_path, "r") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["question_id"])
    return done


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


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

    already_done = load_already_done(out_path)
    remaining_ids = [qid for qid in question_ids if qid not in already_done]
    if already_done:
        print(f"Resuming: {len(already_done)} already done, "
              f"{len(remaining_ids)} remaining for k={k}")

    from generation.prompt_template import build_prompt  # local import avoids circularity

    batch_size = cfg["generation"].get("batch_size", 8)

    with open(out_path, "a") as out_f:  # append: never clobber prior progress
        for batch_ids in chunked(remaining_ids, batch_size):
            batch_rows = [questions_df.loc[qid] for qid in batch_ids]
            batch_retrievals = [retrieval_records[qid] for qid in batch_ids]
            batch_top_k_ids = [r["retrieved_ids"][:k] for r in batch_retrievals]
            batch_passage_texts = [
                [corpus_df.loc[pid, "text"] for pid in ids] for ids in batch_top_k_ids
            ]
            batch_prompts = [
                build_prompt(row["question"], texts)
                for row, texts in zip(batch_rows, batch_passage_texts)
            ]

            batch_results = backend.generate_batch(batch_prompts)

            for qid, row, retrieval, top_k_ids, passage_texts, (raw_answer, gen_time) in zip(
                batch_ids, batch_rows, batch_retrievals, batch_top_k_ids,
                batch_passage_texts, batch_results,
            ):
                context_token_count = backend.count_tokens("\n\n".join(passage_texts))
                generated_answer = postprocess_generation(raw_answer)

                gold_answers = list(row["gold_answers"])
                em, f1 = compute_em_f1(generated_answer, gold_answers)

                gold_rank = retrieval["gold_rank"]
                gold_retrieved = 0 <= gold_rank < k

                record = {
                    "question_id": qid,
                    "question": row["question"],
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
            out_f.flush()

    print(f"Wrote {len(remaining_ids)} new records to {out_path} "
          f"({len(already_done) + len(remaining_ids)} total)")