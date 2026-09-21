"""Load and save the topic-info, corpus, and topic-name tables.

The ``<prefix>_topic_names.txt`` file is written by part 1 (names + descriptions)
and read back / overwritten by part 2 (adds ``global_name``), so its IO lives
here where both stages can share it. All paths come from the run folder
(``run`` = the handle returned by ``setup_run.setup()``).
"""
import pandas as pd


def load_topic_corpus(run, prefix: str, id_col: str):
    """Load topic info + document-level texts for one corpus (stage 02 files).

    Returns ``(topic_info, df)`` where ``df`` is the document-to-topic
    assignment table merged with the corpus ``text`` column.
    """
    topic_info = pd.read_csv(run.get("02", f"{prefix}_topic_info.txt"), sep="\t")
    doc_topics = pd.read_csv(run.get("02", f"{prefix}_doc_topics.txt"), sep="\t")
    corpus = pd.read_csv(run.get("02", f"{prefix}_corpus.txt"), sep="\t")
    df = doc_topics.merge(corpus, on=id_col, how="left")
    return topic_info, df


def load_topic_names(run, prefix: str) -> pd.DataFrame:
    """Load the part-1 topic-name table for one corpus."""
    return pd.read_csv(run.get("03", f"{prefix}_topic_names.txt"), sep="\t")


def save_topic_names(run, df: pd.DataFrame, prefix: str) -> None:
    """Write the topic-name table for one corpus to the run folder."""
    df.to_csv(run.out(f"{prefix}_topic_names.txt"), sep="\t", index=False)
