# rag-context-length

**How Much Context Is Enough? Investigating the Impact of Retrieval Context Length on RAG-Based Question Answering**

A controlled, lightweight RAG experiment for a Master's-level NLP course project. We measure how the number of retrieved passages (`k`) fed to a generator affects downstream QA performance, and separate *retrieval success* from *generation/context-utilization success*.

## Research question

> Does increasing the number of retrieved passages consistently improve answer quality, or is there an optimal retrieval context size beyond which additional context becomes ineffective or harmful?

We test this without assuming the answer in advance. Four hypotheses are on the table — context saturation, context noise, retrieval/generation divergence, and (optionally) question-complexity sensitivity — and the poster's conclusion is written only after the data is in.

## Pipeline

```
Question → MiniLM embedding → FAISS top-20 retrieval → top-k context slice → Qwen2.5-3B-Instruct → generated answer → EM / F1
```

## Method summary

| Component | Choice |
|---|---|
| Dataset | SQuAD v1.1 (`train` split), 1,500 questions, seed=42 |
| Corpus | Full deduplicated set of SQuAD contexts (~18–20K passages), not just gold passages |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Index | FAISS `IndexFlatIP` (cosine via normalized inner product) |
| Generator | `Qwen/Qwen2.5-3B-Instruct` (4-bit GPU, or Q4_K_M GGUF on CPU-only) |
| k values | 1, 3, 5, 10, 20 (retrieved once as top-20, sliced as nested prefixes) |
| Generation | Deterministic: `do_sample=False`, `temperature=0` |
| Metrics | Token-level F1 (primary), Exact Match (secondary), Recall@k |

Full configuration is frozen in [`config.yaml`](./config.yaml) — every script reads from it rather than hardcoding constants.

## Repository structure

```
data/            corpus, FAISS index, sampled question set
retrieval/       corpus construction, embedding, indexing, Recall@k
generation/      prompt template, model backend, generation loop
evaluation/      normalization, EM/F1, per-k and conditional-F1 aggregation
experiments/     pilot, main k-sweep, distractor experiment, optional position experiment
results/         per-question JSONL outputs, per-k experiment tables, plots
error_analysis/  manual inspection of 100–150 failure/transition examples
notebooks/       Colab-ready wrapper around the scripts
poster/          final figures and references
```

## Setup

```bash
git clone https://github.com/<your-username>/rag-context-length.git
cd rag-context-length
pip install -r requirements.txt
```

For CPU-only local runs, additionally install the GGUF backend (kept out of the default requirements since it needs a compiled build):

```bash
pip install llama-cpp-python==0.2.90
```

## Running the pipeline

Run phases in order — each gates the next.

```bash
# Phase 1 — retrieval baseline
python retrieval/build_corpus.py
python retrieval/sample_questions.py
python retrieval/embed_corpus.py
python retrieval/retrieve.py          # prints Recall@k, sanity-check before continuing

# Phase 2 — pilot (75 questions, all k, before committing to full inference)
python experiments/run_pilot.py

# Phase 3 — main experiment (1,500 questions × 5 k-values)
python experiments/run_main_experiment.py

# Phase 4 — aggregate results
python evaluation/aggregate.py

# Phase 5 — distractor experiment
python experiments/run_distractor_experiment.py

# Phase 6 — (optional) evidence-position experiment, only after core experiments
python experiments/run_position_experiment.py
```

On Colab, use `notebooks/colab_runner.ipynb`, which calls the same scripts — no logic is duplicated between notebook and CLI paths.

Do not skip the pilot. It checks memory, generation speed, context-window headroom, prompt behavior, metric correctness, and JSONL resume logic before the expensive full run starts.

## What's fixed vs. what varies

Everything is held constant across k-conditions except the number of retrieved passages: retriever, embedding model, generator, prompt template, generation settings, and question set. This isolates retrieval depth as the sole independent variable in the main experiment. The distractor experiment additionally isolates *context noise* from *retrieval depth* by holding one relevant passage fixed and varying only the number of irrelevant passages added alongside it.

## Key analyses

- **Main table:** Recall@k, EM, F1, and average context token count at each k.
- **Conditional F1:** answer F1 restricted to questions where the gold passage *was* retrieved — isolates generation/context-utilization failures from retrieval failures.
- **Distractor curve:** answer quality as irrelevant context is added while relevant evidence stays present.
- **Error analysis:** 100–150 manually reviewed examples, prioritizing questions that flip from correct to incorrect as `k` increases.

## Reproducibility

- Fixed seed (`42`) for question sampling and distractor selection.
- Deterministic generation (`do_sample=False`, `temperature=0`).
- Retrieval computed once at top-20 and sliced, not re-queried per k.
- Per-question raw outputs (not just aggregates) saved as JSONL after every generation call, so no expensive inference is ever repeated.
- Full config, library versions, and hardware recorded per run in `config.yaml`.

## Non-goals

This project is scoped to retrieval context length only. It does **not** cover chunking strategy, embedding-model comparison, vector-DB comparison, fine-tuning, multiple generator models, agentic or Self-RAG, or multimodal/multilingual RAG.

## Limitations

Single QA dataset (SQuAD v1.1, primarily extractive), single embedding model, single retriever, single generator. Passage count is only an approximation of true context length (token counts are also recorded and reported separately). The generator may have seen SQuAD during pretraining, so results should be read as relative comparisons across context conditions rather than claims about unseen-world QA accuracy. Findings are specific to this experimental setup and are not claimed to generalize to all RAG systems.

## License

Course project — add a license if/when you intend to share beyond the course.
