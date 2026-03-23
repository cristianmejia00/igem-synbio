# 03 — Topic Names

## Setup

You need an **OpenAI API key**.  Create a plain-text file called
`openai.key` in this folder containing only the key (no newlines or
quotes).  The file is git-ignored by default.

The prompt templates used by the notebooks are stored in `prompts.yaml`.
You can edit them to adjust style or domain framing without changing any
code.

## Description

After the topic model assigns each document to a cluster, the clusters
still carry auto-generated keyword labels (e.g. "dna_synthetic_gene").
This folder uses a large-language model to produce human-readable,
publication-quality names and descriptions.

### `prompts.yaml`

Contains the theme definition ("Synthetic Biology") and three prompt
templates:

- **cluster_description** — asks the model to read representative
  documents from a cluster and describe what the cluster is about.
- **cluster_description_enhanced** — asks for a polished, single-paragraph
  synthesis of the cluster description.
- **cluster_name** — asks for a short label (a few words) that captures
  the cluster's essence.

### `get_topic_names_part1.ipynb`

1. **Load data** — reads topic summary tables, document-level topic
   assignments, and the text corpora produced by the topic-model step.
2. **Select representative documents** — for each cluster, picks the
   top-ranked documents and concatenates their texts.
3. **Generate descriptions** — sends the representative texts to the LLM
   in two rounds: first a raw description, then an enhanced synthesis.
4. **Generate short names** — asks the LLM to distil each enhanced
   description into a concise label.
5. **Save** — writes a table per corpus (`teams_topic_names.txt`,
   `papers_topic_names.txt`) with columns for the topic identifier,
   short name, enhanced description, and raw description.

### `get_topic_names_part2.ipynb`

A second pass that resolves duplicate or ambiguous names across topics.

1. **Load Part 1 results** — reads the name tables produced above.
2. **Global renaming** — sends *all* topic names and descriptions to the
   LLM in a single call, using OpenAI function calling to enforce a
   structured response.  The model returns one unique, publication-ready
   name per topic.
3. **Validate** — checks that every topic received a name and that no
   two topics share the same name.
4. **Save** — overwrites the Part 1 files, adding a `global_name`
   column with the final names.
