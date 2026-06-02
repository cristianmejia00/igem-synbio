"""Shared helpers for the 04-topic_hierarchy pipeline.

The notebooks under ``01-teams/`` and ``02-papers/`` import from this package.
Splitting the reusable mechanics out of the notebooks keeps each notebook a
thin, dataset-specific orchestration layer and avoids copying the same function
bodies across teams and papers.

Modules
-------
paths      : filesystem paths, shared resources, and constants
hierarchy  : build the low→mid→high hierarchy from a BERTopic merge tree, and
             write the per-corpus report tables
naming     : LLM (function-calling) naming of the mid / high hierarchy groups
"""
