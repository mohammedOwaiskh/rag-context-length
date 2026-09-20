import json
from pathlib import Path

import pandas as pd

from utils import load_config


def load_results_for_k(results_dir: str, k: int) -> pd.DataFrame:
    path = Path(results_dir) / f"results_k{k}.jsonl"
    rows = [json.loads(line) for line in open(path, "r")]
    return pd.DataFrame(rows)


def summarize_results(cfg: dict, results_dir: str) -> pd.DataFrame:
    rows = []
    for k in cfg["retrieval"]["k_values"]:
        df = load_results_for_k(results_dir, k)
        n = len(df)

        recall_at_k = df["gold_retrieved"].mean()
        em = df["exact_match"].mean()
        f1 = df["f1"].mean()
        avg_tokens = df["context_token_count"].mean()

        conditional = df[df["gold_retrieved"]]
        conditional_f1 = conditional["f1"].mean() if len(conditional) > 0 else float("nan")
        conditional_em = conditional["exact_match"].mean() if len(conditional) > 0 else float("nan")

        rows.append({
            "k": k,
            "n_questions": n,
            "recall_at_k": round(recall_at_k, 4),
            "em": round(em, 4),
            "f1": round(f1, 4),
            "avg_context_tokens": round(avg_tokens, 1),
            "conditional_f1_given_retrieved": round(conditional_f1, 4),
            "conditional_em_given_retrieved": round(conditional_em, 4),
        })

    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    summary = summarize_results(cfg, results_dir=cfg["paths"]["main_results_dir"])
    print(summary.to_string(index=False))

    out_path = Path(cfg["paths"]["main_results_dir"]) / "summary_table.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary table to {out_path}")


if __name__ == "__main__":
    main()