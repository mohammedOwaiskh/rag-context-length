import time
from pathlib import Path

import pandas as pd
import yaml

from generation.model_backend import load_backend
from generation.generate import run_generation
from evaluation.aggregate import summarize_results
from utils import load_config
from utils.logger import setup_logger

log = setup_logger("run_full_experiment")

def main():
    cfg = load_config()

    questions_df = pd.read_parquet(cfg["corpus"]["questions_path"])
    question_ids = questions_df["question_id"].tolist()
    n = len(question_ids)
    log.info(f"Main experiment: {n} questions x {cfg['retrieval']['k_values']} k-values")

    backend = load_backend(cfg)

    out_dir = Path(cfg["paths"]["main_results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in cfg["retrieval"]["k_values"]:
        out_path = out_dir / f"results_k{k}.jsonl"
        log.info(f"\n--- k={k} ---")
        start = time.time()
        run_generation(cfg, backend, k, question_ids, str(out_path))
        elapsed = time.time() - start
        log.info(f"k={k} took {elapsed / 60:.1f} min this run "
              f"({elapsed / n:.2f}s/question avg, this run only)")

    # --- Aggregate once all k's are complete ---
    all_done = all(
        len(open(out_dir / f"results_k{k}.jsonl").readlines()) >= n
        for k in cfg["retrieval"]["k_values"]
        if (out_dir / f"results_k{k}.jsonl").exists()
    )
    if all_done:
        log.info("\nAll k-conditions complete. Summary:")
        summary = summarize_results(cfg, results_dir=str(out_dir))
        log.info(summary.to_string(index=False))
        summary.to_csv(out_dir / "summary_table.csv", index=False)
        log.info(f"\nSaved to {out_dir / 'summary_table.csv'}")
    else:
        log.info("\nNot all k-conditions have all questions yet — re-run this "
              "script to continue from where it left off.")


if __name__ == "__main__":
    main()