"""Project-wide constants for the topic-model notebooks.

Input and output locations come from the run folder (``assets/<date>/02/``)
returned by ``setup_run.setup()``; see ``run_paths.py`` at the repo root.
"""
import os
from pathlib import Path

import numpy as np

# <repo root>/02-topic_model/aux/paths.py  →  parents[2] == <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sentence-transformer model used for every corpus.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global random seed for reproducible UMAP / sampling.
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set process-wide seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
