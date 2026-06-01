"""Filesystem paths and project-wide constants.

Paths are resolved from this file's own location so they stay correct no
matter which directory a notebook's kernel happens to start in::

    aux/paths.py  ->  aux/  ->  02-topic_model/  ->  <repo root>
"""
import os
from pathlib import Path

import numpy as np

# <repo root>/02-topic_model/aux/paths.py  →  parents[2] == <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSETS_DIR = PROJECT_ROOT / "assets"
EMBEDDINGS_DIR = ASSETS_DIR / "embeddings"
MODELS_DIR = ASSETS_DIR / "topic_models"
REPORTS_DIR = ASSETS_DIR / "reports"

# Sentence-transformer model used for every corpus.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global random seed for reproducible UMAP / sampling.
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set process-wide seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
