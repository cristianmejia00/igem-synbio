# 02 — Topic Model

Turns the raw text corpora into numerical embeddings, clusters them with
BERTopic, and (optionally) performs a grid-search evaluation to choose the
best hyperparameters.

## Setup

A CUDA-capable GPU is **not** required but will speed up the embedding step
significantly. On CPU the embedding computation may take a few minutes per
corpus.

## Layout

The pipeline is split **per corpus** so the teams and papers analyses can be
re-run independently — updating one no longer recomputes (and, given the
stochastic nature of topic models, perturbs) the other.

```text
02-topic_model/
├── aux/                 shared helper functions imported by every notebook
├── 01-teams/            iGEM teams pipeline
├── 02-papers/           SynBio OpenAlex papers pipeline
└── 03-combined/         analysis that spans both corpora
```

Each of `01-teams/` and `02-papers/` contains the same three notebooks,
differing only in a small CONFIG block at the top (source file, column names,
hyperparameter grid):

| Notebook | Purpose |
|---|---|
| `get_embeddings.ipynb` | Concatenate title + abstract, clean, and encode with `all-MiniLM-L6-v2` → `assets/embeddings/` |
| `get_topics.ipynb` | Fit a single BERTopic model (UMAP → HDBSCAN) with manual hyperparameters → `assets/topic_models/` |
| `get_topics_with_evaluation.ipynb` | **Recommended.** Grid-search UMAP/HDBSCAN parameters, score each by C_v coherence / diversity / DBCV, pick the best → `assets/topic_models/` |

`03-combined/` holds the cross-corpus analysis:

| Notebook | Purpose |
|---|---|
| `orphans.ipynb` | Profile the outlier (topic = −1) documents of **both** corpora to understand the noise rate and which orphans could be reassigned |

### Execution order

1. `01-teams/get_embeddings.ipynb` and `02-papers/get_embeddings.ipynb`
2. `…/get_topics_with_evaluation.ipynb` (recommended) **or** `…/get_topics.ipynb`
3. `03-combined/orphans.ipynb` (needs the `*_doc_topics.txt` files from step 2)

## Shared module — `aux/`

Reusable mechanics are factored out of the notebooks so each notebook stays a
thin, dataset-specific orchestration layer. Imported via
`sys.path.insert(0, str(Path.cwd().parent))` then `from aux.<module> import …`.

| Module | Key functions |
|---|---|
| `paths.py` | `PROJECT_ROOT`, `ASSETS_DIR`, `EMBEDDINGS_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `SEED`, `EMBEDDING_MODEL`, `set_seed()` |
| `embeddings.py` | `prepare_text()`, `encode_texts()`, `save_embeddings()` |
| `topic_modeling.py` | `load_corpus()`, `fit_topic_model()`, `save_topic_outputs()` |
| `evaluation.py` | `coherence_cv()`, `topic_diversity()`, `dbcv_score()`, `grid_search()` |
| `orphans.py` | `load_outlier_inputs()`, `outlier_summary()`, `plot_outlier_rate_by_year()`, `plot_text_length()`, `language_outlier_table()`, `nearest_centroid_analysis()`, `citation_profile()`, `concept_overlap()`, `sample_outlier_docs()`, `sample_outlier_table()`, `save_orphans()`, `build_summary()` |

Paths in `paths.py` are resolved from the module's own location, so the
notebooks work regardless of the kernel's working directory.

## Notes on the pipeline

### Embeddings (`get_embeddings.ipynb`)

For each record the title and abstract are concatenated into a single string,
non-alphabetic noise is removed, and records left without usable text are
dropped. The cleaned text is encoded with the `all-MiniLM-L6-v2`
sentence-transformer (384-dim). Outputs: `<corpus>_embeddings.npy` and the
aligned `<corpus>_corpus.txt` (id + text) under `assets/embeddings/`.

### Topics (`get_topics.ipynb`)

Builds a UMAP (dimensionality reduction) → HDBSCAN (density clustering) →
BERTopic (n-gram topic extraction) pipeline. Minimum cluster size is set
manually per corpus. Saves the fitted model, a per-topic summary table, and
document-level topic assignments to `assets/topic_models/`.

### Topics with evaluation (`get_topics_with_evaluation.ipynb`) — recommended

Adds a systematic hyperparameter search over minimum cluster size and the
number of UMAP neighbours/components. Each configuration is scored with:

- **C_v coherence** — semantic consistency of the discovered topics (higher is better).
- **Topic diversity** — fraction of unique words across all topic word lists.
- **DBCV** — HDBSCAN's density-based cluster-validity score.

The best configuration is selected by C_v coherence (ties broken by diversity).

**Outlier reduction (`REDUCE_OUTLIERS`).** HDBSCAN labels documents outside any
dense cluster as topic −1 (noise). This is acceptable for the SynBio
literature (some papers are genuinely off-topic), so it is **disabled** for
papers. Every iGEM team project is by definition synthetic biology — its text
may simply be too short or idiosyncratic to land in a cluster — so for teams it
is **enabled**: `reduce_outliers` (strategy `"embeddings"`, threshold `0`)
reassigns *all* noise documents to their nearest topic without retraining.

### Orphan analysis (`03-combined/orphans.ipynb`)

Profiles the topic −1 documents of both corpora: outlier rate by year, text
length, language (papers), distance to the nearest cluster centroid, citation
profile (papers), concept overlap (papers), and random samples. Writes the
orphan documents to `assets/reports/orphans_papers.tsv` and
`assets/reports/orphans_teams.tsv`.
