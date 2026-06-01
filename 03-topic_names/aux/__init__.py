"""Shared helpers for the 03-topic_names pipeline.

The notebooks under ``01-teams/`` and ``02-papers/`` import from this package.
Splitting the reusable mechanics out of the notebooks keeps each notebook a
thin, dataset-specific orchestration layer and avoids copying the same function
bodies across teams and papers.

Modules
-------
paths          : filesystem paths, shared resources, and constants
openai_client  : prompt loading, OpenAI client, and generic chat helpers
tables         : load / save the topic-info, corpus, and topic-name tables
naming         : per-cluster description + name generation (part 1)
global_rename  : function-calling pass for globally unique names (part 2)
"""
