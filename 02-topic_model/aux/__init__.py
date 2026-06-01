"""Shared helpers for the 02-topic_model pipeline.

The notebooks under ``01-teams/``, ``02-papers/`` and ``03-combined/`` import
from this package.  Splitting the reusable mechanics out of the notebooks keeps
each notebook a thin, dataset-specific orchestration layer and avoids copying
the same function bodies across teams and papers.

Modules
-------
paths           : filesystem paths and project-wide constants (``SEED`` etc.)
embeddings      : corpus text preparation and sentence-transformer encoding
topic_modeling  : build / fit / persist BERTopic models
evaluation      : coherence, diversity, DBCV metrics and the grid search
orphans         : outlier (topic = -1) profiling for the combined analysis
"""
