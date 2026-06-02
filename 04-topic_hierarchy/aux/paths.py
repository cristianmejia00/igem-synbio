"""Filesystem paths, shared resources, and project-wide constants.

Paths are resolved from this file's own location so they stay correct no matter
which directory a notebook's kernel happens to start in::

    aux/paths.py  ->  aux/  ->  04-topic_hierarchy/  ->  <repo root>
"""
import os
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parents[1]      # 04-topic_hierarchy/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo root>

ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "topic_models"
EMBEDDINGS_DIR = ASSETS_DIR / "embeddings"
REPORTS_DIR = ASSETS_DIR / "reports"

# Shared resources. prompts_hierarchy.yaml stays at the step root; the OpenAI
# key is reused from 03-topic_names (the single place it lives).
PROMPTS_PATH = STEP_DIR / "prompts_hierarchy.yaml"
OPENAI_KEY_PATH = PROJECT_ROOT / "03-topic_names" / "openai.key"

OPENAI_MODEL = "gpt-4.1-nano"
SEED = 42

# High-level k is searched in [HIGH_K_MIN, HIGH_K_MAX]; the mid-level range is
# derived per corpus from the low-topic count (see hierarchy.auto_mid_k_range).
HIGH_K_MIN = 4
HIGH_K_MAX = 11


def set_seed(seed: int = SEED) -> None:
    """Set process-wide seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
