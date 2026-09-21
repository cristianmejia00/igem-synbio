"""Setup for 05-reporting: prepare today's run folder ``assets/<date>/05/``.

Copies the upstream files this stage reads into today's run folder when they are
missing there (from the newest earlier run that has them), then checks that they
are consistent with each other (see ``run_paths.py`` at the repo root).

Called from the first cell of every notebook and script in this folder; it only copies what is
missing, so running it again is harmless. It can also be run on its own::

    python setup_run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for run_paths
from run_paths import prepare  # noqa: E402

STAGE = "05"

# Upstream files read by this stage. ``{c}`` = teams / papers; ``{data}`` = that
# corpus's dataset (00/igem.txt or 01/synbio_openalex.txt).
NEEDS = [
    "{data}",
    "02/{c}_embeddings.npy",
    "02/{c}_doc_topics.txt",
    "03/{c}_topic_names.txt",
    "04/{c}_topic_name_hierarchy.tsv",
]


def setup(corpus=None):
    """Prepare today's run folder for this stage and return its ``Run`` handle.

    ``corpus`` ("teams" / "papers") limits the setup to one corpus; ``None`` = both.
    """
    return prepare(STAGE, NEEDS, corpus)


if __name__ == "__main__":
    setup()
