from pathlib import Path

import pandas as pd
from datasets import load_dataset

from utils import load_config, passage_id, get_project_root
from utils.logger import setup_logger

log = setup_logger(__name__)


def build_corpus(cfg: dict):
    log.info("Building corpus")
    log.info("Loading dataset")

    ds = load_dataset("rajpurkar/squad", split=cfg["dataset"]["hf_split"])

    log.info(f"Loaded {len(ds)} rows. Deduplicating contexts...")

    seen = dict()

    for row in ds:
        text_context = row["context"]
        pid = passage_id(text_context)
        if pid not in seen:
            seen[pid] = {
                "passage_id": pid,
                "text": text_context,
                "title": row.get("title", None)
            }

    df = pd.DataFrame(seen.values())
    log.info(f"Corpus built. Unique passages after dedup: {len(df)}")
    return df


def main():
    cfg = load_config()

    out_path = get_project_root() / cfg["corpus"]["paths"]["passages"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_corpus(cfg)

    df["token_count"] = None

    df.to_parquet(out_path, index=False)
    log.info(f"Saved corpus to {out_path}")


if __name__ == "__main__":
    main()