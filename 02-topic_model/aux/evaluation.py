"""Topic-model quality metrics and a UMAP/HDBSCAN parameter grid search."""
from itertools import product

import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from .topic_modeling import fit_topic_model


def coherence_cv(topic_model: BERTopic, docs: list[str], top_n: int = 10) -> float:
    """Compute C_v coherence over the top-N words of each non-outlier topic.

    BERTopic may produce multi-word n-grams (e.g. 'gene expression').  We split
    those into individual tokens so every term is present in the unigram
    dictionary that Gensim builds from the corpus.
    """
    topics = topic_model.get_topics()
    topic_words = []
    for t in sorted(topics):
        if t == -1:
            continue
        words = []
        for word, _ in topics[t][:top_n]:
            words.extend(word.lower().split())  # split n-grams into unigrams
        topic_words.append(words)
    if not topic_words:
        return float("nan")
    tokenized = [doc.lower().split() for doc in docs]
    dictionary = Dictionary(tokenized)
    # Keep only tokens that exist in the dictionary
    topic_words = [
        [w for w in tw if w in dictionary.token2id]
        for tw in topic_words
    ]
    topic_words = [tw for tw in topic_words if len(tw) >= 2]
    if not topic_words:
        return float("nan")
    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    return cm.get_coherence()


def topic_diversity(topic_model: BERTopic, top_n: int = 10) -> float:
    """Fraction of unique words across all topics' top-N word lists."""
    topics = topic_model.get_topics()
    all_words = [
        word
        for t in sorted(topics) if t != -1
        for word, _ in topics[t][:top_n]
    ]
    return len(set(all_words)) / len(all_words) if all_words else float("nan")


def dbcv_score(topic_model: BERTopic) -> float:
    """HDBSCAN relative validity (DBCV). Higher is better, range [-1, 1]."""
    return topic_model.hdbscan_model.relative_validity_


def grid_search(
    docs: list[str],
    embeddings: np.ndarray,
    param_grid: dict,
    label: str = "",
):
    """Run every parameter combination, evaluate each, return results + best.

    Returns ``(results_df, best_model_data)`` where ``results_df`` is sorted by
    descending C_v coherence and ``best_model_data`` holds the fitted model,
    topics, probabilities, and metrics for the best configuration (highest C_v,
    ties broken by diversity).
    """
    combos = list(product(
        param_grid["min_cluster_size"],
        param_grid["umap_n_neighbors"],
        param_grid["umap_n_components"],
    ))
    n_total = len(combos)
    rows = []
    best_score = -np.inf
    best_model_data = None

    for i, (mcs, nn, nc) in enumerate(combos, 1):
        print(f"[{label}] {i}/{n_total}  mcs={mcs}, nn={nn}, nc={nc} ... ", end="", flush=True)
        model, topics, probs = fit_topic_model(
            docs, embeddings,
            min_cluster_size=mcs, umap_n_neighbors=nn, umap_n_components=nc,
        )
        n_topics = model.get_topic_info().Topic.max() + 1
        outlier_frac = (np.array(topics) == -1).mean()
        cv = coherence_cv(model, docs)
        div = topic_diversity(model)
        dbcv = dbcv_score(model)

        row = {
            "min_cluster_size": mcs,
            "n_neighbors": nn,
            "n_components": nc,
            "n_topics": n_topics,
            "outlier_frac": round(outlier_frac, 4),
            "coherence_cv": round(cv, 4),
            "diversity": round(div, 4),
            "dbcv": round(dbcv, 4),
        }
        rows.append(row)
        print(f"topics={n_topics}, C_v={cv:.4f}, div={div:.4f}, DBCV={dbcv:.4f}")

        # Track best model by C_v (primary), break ties by diversity
        if cv > best_score or (cv == best_score and div > (best_model_data or {}).get("diversity", -1)):
            best_score = cv
            best_model_data = {
                "model": model, "topics": topics, "probs": probs,
                "diversity": div, **row,
            }

    df = pd.DataFrame(rows).sort_values("coherence_cv", ascending=False).reset_index(drop=True)
    return df, best_model_data
