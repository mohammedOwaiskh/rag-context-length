import time
from pathlib import Path

import pandas as pd
import yaml

from generation.model_backend import load_backend
from generation.generate import run_generation
from evaluation.aggregate import summarize_results
from utils import load_config
from utils.logger import setup_logger

log = setup_logger("run_pilot")

def main():
    cfg = load_config()

    questions_df = pd.read_parquet(cfg["corpus"]["questions_path"])
    n_pilot = cfg["pilot"]["n_questions"]
    pilot_question_ids = questions_df["question_id"].tolist()[:n_pilot]
    log.info(f"Pilot: {n_pilot} questions (first {n_pilot} of the fixed 1,500 sample)")

    backend = load_backend(cfg)

    out_dir = Path(cfg["paths"]["pilot_results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gen_times = []
    for k in cfg["retrieval"]["k_values"]:
        out_path = out_dir / f"results_k{k}.jsonl"
        log.info(f"\n--- k={k} ---")
        start = time.time()
        run_generation(cfg, backend, k, pilot_question_ids, str(out_path))
        elapsed = time.time() - start
        log.info(f"k={k} pilot batch took {elapsed:.1f}s "
              f"({elapsed / n_pilot:.2f}s/question)")
        all_gen_times.append((k, elapsed / n_pilot))

    # --- Aggregate + log.info summary table ---
    summary = summarize_results(cfg, results_dir=str(out_dir))
    log.info("\nPilot summary (Recall@k / EM / F1 / avg context tokens):")
    log.info(summary.to_string(index=False))

    # --- Extrapolate full-run time (manual gate check #2: speed) ---
    log.info("\nFull-run time extrapolation (1,500 questions x 5 k-values):")
    n_full = cfg["sampling"]["n_questions"]
    total_seconds = sum(t * n_full for _, t in all_gen_times)
    log.info(f"  Estimated total generation time: {total_seconds / 3600:.2f} hours")


if __name__ == "__main__":
    main()