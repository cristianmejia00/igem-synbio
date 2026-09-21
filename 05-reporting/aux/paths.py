"""Palette and constants for the reporting notebooks.

Input and output data locations come from the run folder (``assets/<date>/05/``)
returned by ``setup_run.setup()``; see ``run_paths.py`` at the repo root.
"""
import os
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parents[1]      # 05-reporting/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo root>

SEED = 42

# Persisted joint UMAP coordinates (written by 03-compute_coords.ipynb and reused
# by every other notebook so the projection is computed exactly once).
PAPERS_COORDS_FILE = "joint_umap_papers_xy.tsv"
TEAMS_COORDS_FILE = "joint_umap_teams_xy.tsv"

# Recycled categorical palette for topic colours.
PALETTE = [
    "#f00f15", "#2270e7", "#e5e510", "#ff8103", "#4f3dd1",
    "#26cc3a", "#ec058e", "#9cb8c2", "#fffdd0", "#b40e68",
    "#5afb5a", "#beaed4", "#fdc086", "#99fdff", "#c430ff",
    "#e4dbe0", "#bf5b17", "#666666",
]

OUTLIER_COLOR = "#d3d3d3"


def set_seed(seed: int = SEED) -> None:
    """Set process-wide seeds for reproducible UMAP / sampling."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
