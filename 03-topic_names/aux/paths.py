"""Shared resources and project-wide constants.

Paths are resolved from this file's own location so they stay correct no matter
which directory a notebook's kernel happens to start in::

    aux/paths.py  ->  aux/  ->  03-topic_names/  ->  <repo root>

Input and output data locations come from the run folder (``assets/<date>/03/``)
returned by ``setup_run.setup()``; see ``run_paths.py`` at the repo root.
"""
from pathlib import Path

# <repo root>/03-topic_names/aux/paths.py
STEP_DIR = Path(__file__).resolve().parents[1]      # 03-topic_names/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo root>

# Shared resources kept at the step root. ``openai.key`` deliberately stays here
# (it is also read by 04-topic_hierarchy/name_hierarchy_levels.ipynb).
PROMPTS_PATH = STEP_DIR / "prompts.yaml"
OPENAI_KEY_PATH = STEP_DIR / "openai.key"

# OpenAI model and number of representative documents sent per cluster.
OPENAI_MODEL = "gpt-4.1-nano"
TOP_N_DOCS = 5
