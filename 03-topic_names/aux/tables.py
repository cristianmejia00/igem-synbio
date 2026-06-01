"""Load and save the topic-info, corpus, and topic-name tables.

The ``<prefix>_topic_names.txt`` file is written by part 1 (names + descriptions)
and read back / overwritten by part 2 (adds ``global_name``), so its IO lives
here where both stages can share it.
"""
import pandas as pd

from .paths import EMBEDDINGS_DIR, MODELS_DIR


def load_topic_corpus(prefix: str, id_col: str):
    """Load topic info + document-level texts for one corpus.

    Returns ``(topic_info, df)`` where ``df`` is the document-to-topic
    assignment table merged with the corpus ``text`` column.
    """
    topic_info = pd.read_csv(MODELS_DIR / f"{prefix}_topic_info.txt", sep="\t")
    doc_topics = pd.read_csv(MODELS_DIR / f"{prefix}_doc_topics.txt", sep="\t")
    corpus = pd.read_csv(EMBEDDINGS_DIR / f"{prefix}_corpus.txt", sep="\t")
    df = doc_topics.merge(corpus, on=id_col, how="left")
    return topic_info, df


def load_topic_names(prefix: str) -> pd.DataFrame:
    """Load the part-1 topic-name table for one corpus."""
    return pd.read_csv(MODELS_DIR / f"{prefix}_topic_names.txt", sep="\t")


def save_topic_names(df: pd.DataFrame, prefix: str, models_dir=MODELS_DIR) -> None:
    """Write the topic-name table for one corpus to ``models_dir``."""
    models_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(models_dir / f"{prefix}_topic_names.txt", sep="\t", index=False)
