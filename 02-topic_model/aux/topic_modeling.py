"""Build, fit, and persist BERTopic models (UMAP -> HDBSCAN backbone)."""
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from .paths import EMBEDDINGS_DIR, MODELS_DIR, SEED


def load_corpus(embeddings_file: str, corpus_file: str):
    """Load a saved embedding matrix and its aligned corpus table.

    Returns ``(embeddings, corpus_df)``; the row counts are asserted to match.
    """
    embeddings = np.load(EMBEDDINGS_DIR / embeddings_file)
    corpus = pd.read_csv(EMBEDDINGS_DIR / corpus_file, sep="\t")
    assert len(corpus) == embeddings.shape[0], "corpus / embeddings length mismatch"
    return embeddings, corpus


def fit_topic_model(
    docs: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    seed: int = SEED,
    verbose: bool = False,
) -> tuple[BERTopic, list[int], np.ndarray]:
    """Fit a BERTopic model using UMAP (reduction) + HDBSCAN (clustering).

    ``gen_min_span_tree=True`` is always set so ``hdbscan_model.relative_validity_``
    (DBCV) is available to the evaluation helpers; it is harmless otherwise.
    """
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        gen_min_span_tree=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_cluster_size,
        n_gram_range=(1, 3),
        language="english",
        calculate_probabilities=True,
        verbose=verbose,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs


def save_topic_outputs(
    topic_model: BERTopic,
    corpus: pd.DataFrame,
    topics: list[int],
    id_col: str,
    prefix: str,
    models_dir=MODELS_DIR,
) -> pd.DataFrame:
    """Persist the model, its topic-info table, and document-level assignments.

    Writes ``<prefix>_topic_model``, ``<prefix>_topic_info.txt`` and
    ``<prefix>_doc_topics.txt`` into ``models_dir``.  Returns the topic-info table.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(models_dir / f"{prefix}_topic_model"))

    info = topic_model.get_topic_info()
    info.to_csv(models_dir / f"{prefix}_topic_info.txt", sep="\t", index=False)

    out = corpus.copy()
    out["topic"] = topics
    out[[id_col, "topic"]].to_csv(
        models_dir / f"{prefix}_doc_topics.txt", sep="\t", index=False
    )
    return info
