# 02 — Topic Model

## Setup

A CUDA-capable GPU is **not** required but will speed up the embedding
step significantly.  On CPU the embedding computation may take a few
minutes per corpus.

## Description

This folder turns the raw text corpora into numerical embeddings,
clusters them with BERTopic, and (optionally) performs a grid-search
evaluation to choose the best hyperparameters.

### `get_embeddings.ipynb`

1. **Load corpora** — reads the iGEM teams dataset and the OpenAlex
   papers dataset produced in the previous steps.
2. **Prepare text** — for each record, concatenates the title and
   abstract into a single string and removes non-alphabetic noise.
   Records without any usable text are dropped.
3. **Encode** — feeds both corpora through the `all-MiniLM-L6-v2`
   sentence-transformer model, producing one 384-dimensional embedding
   vector per document.
4. **Save** — stores the embedding matrices as NumPy files
   (`teams_embeddings.npy`, `papers_embeddings.npy`) and the
   corresponding cleaned texts as TSV files (`teams_corpus.txt`,
   `papers_corpus.txt`) inside `assets/embeddings/`.

### `get_topics.ipynb`

1. **Load embeddings and texts** — reads the files produced by the
   previous notebook.
2. **Fit BERTopic** — for each corpus, builds a pipeline of UMAP
   (dimensionality reduction) → HDBSCAN (density-based clustering) →
   BERTopic (topic extraction with n-gram tokenisation).
   Hyperparameters such as minimum cluster size are set manually.
3. **Inspect** — prints the number of discovered topics and the count of
   outlier documents (those not assigned to any cluster).
4. **Save** — writes the fitted BERTopic models, per-topic summary
   tables, and document-level topic assignments to
   `assets/topic_models/`.

### `get_topics_with_evaluation.ipynb` **recommended**

An extended version of the notebook above that adds a systematic
hyperparameter search. This is the notebook used in the associated publication.

1. **Define parameter grids** — sets candidate values for minimum
   cluster size, number of UMAP neighbours, and number of UMAP
   components, separately for each corpus.
2. **Grid search** — fits a BERTopic model for every combination of
   parameters and evaluates each one with three metrics:
   - **C_v coherence** — measures how semantically consistent the
     discovered topics are (higher is better).
   - **Topic diversity** — fraction of unique words across all topic
     word lists (higher means less redundancy).
   - **DBCV** — a density-based cluster-validity score from HDBSCAN
     (higher indicates better-separated clusters).
3. **Select best** — picks the configuration with the highest C_v
   coherence (ties broken by diversity).
4. **Reduce iGEM outliers** — HDBSCAN labels documents that fall outside
   any dense cluster as topic −1 (noise).  While this is acceptable for
   the SynBio literature (some papers may be genuinely off-topic), every
   iGEM team project is by definition related to synthetic biology — its
   text may simply be too short or idiosyncratic to land in a cluster.
   BERTopic's `reduce_outliers` method (strategy `"embeddings"`,
   threshold `0`) reassigns **all** noise documents to their nearest
   topic based on cosine similarity, without retraining the model.
5. **Save** — stores the best models, topic tables, document assignments,
   and the full grid-search results to `assets/topic_models/`.
