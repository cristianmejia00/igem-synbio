"""Corpus text preparation and sentence-transformer encoding."""
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .paths import EMBEDDING_MODEL, EMBEDDINGS_DIR


def prepare_text(df: pd.DataFrame, title_col: str, abstract_col: str) -> pd.DataFrame:
    """Concatenate title + abstract into a ``text`` column, clean, drop empties.

    Light cleaning only: keep alphabetic characters and whitespace, collapse the
    rest.  Rows left without any usable text are removed and the index reset.
    """
    df = df.copy()
    df[title_col] = df[title_col].fillna("").astype(str)
    df[abstract_col] = df[abstract_col].fillna("").astype(str)
    df["text"] = (df[title_col] + " " + df[abstract_col]).str.strip()
    df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
    df["text"] = df["text"].str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def encode_texts(
    texts,
    model_name: str = EMBEDDING_MODEL,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Embed an iterable of strings with a sentence-transformer model."""
    model = SentenceTransformer(model_name)
    print(f"Loaded model: {model_name}")
    return model.encode(list(texts), show_progress_bar=show_progress_bar)


def save_embeddings(
    embeddings: np.ndarray,
    corpus: pd.DataFrame,
    id_col: str,
    embeddings_file: str,
    corpus_file: str,
    embeddings_dir=EMBEDDINGS_DIR,
) -> None:
    """Save the embedding matrix (``.npy``) and the aligned ``id + text`` corpus."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / embeddings_file, embeddings)
    corpus[[id_col, "text"]].to_csv(
        embeddings_dir / corpus_file, sep="\t", index=False
    )
