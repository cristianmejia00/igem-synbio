# 03 — Topic Names

After the topic model assigns each document to a cluster, the clusters still
carry auto-generated keyword labels (e.g. "dna_synthetic_gene"). This folder
uses a large-language model to produce human-readable, publication-quality
names and descriptions.

## Setup

You need an **OpenAI API key**. Create a plain-text file called `openai.key` in
this folder containing only the key (no newlines or quotes). The file is
git-ignored by default and is read directly from `03-topic_names/openai.key`
(folder `05-topic_hierarchy` also reads it from here).

The prompt templates used by the notebooks are stored in `prompts.yaml`. You can
edit them to adjust style or domain framing without changing any code.

## Layout

The pipeline is split **per corpus** so the teams and papers naming can be
re-run independently — updating one no longer re-issues (and re-bills) the LLM
calls for the other.

```text
03-topic_names/
├── prompts.yaml         theme definition + prompt templates (shared)
├── openai.key           OpenAI API key (git-ignored, shared)
├── aux/                 shared helper functions imported by every notebook
├── 01-teams/            iGEM teams naming
└── 02-papers/           SynBio OpenAlex papers naming
```

Each of `01-teams/` and `02-papers/` contains the same two notebooks, differing
only in a small CONFIG block at the top (`PREFIX`, `ID_COL`):

| Notebook | Purpose |
|---|---|
| `get_topic_names_part1.ipynb` | Per-cluster naming: send representative documents to GPT for a description, an enhanced synthesis, and a short name → `assets/topic_models/<prefix>_topic_names.txt` |
| `get_topic_names_part2.ipynb` | Global renaming: give the LLM all topics at once (via function calling) so every cluster gets a distinct `global_name` → overwrites the same file |

Run Part 1 before Part 2 for a given corpus.

## Shared resources

### `prompts.yaml`

Contains the theme definition ("Synthetic Biology") and three prompt templates:

- **cluster_description** — read representative documents from a cluster and describe what it is about.
- **cluster_description_enhanced** — polish that into a single cohesive paragraph.
- **cluster_name** — distil the description into a short label.

### `aux/`

Reusable mechanics are factored out of the notebooks so each notebook stays a
thin, dataset-specific orchestration layer. Imported via
`sys.path.insert(0, str(Path.cwd().parent))` then `from aux.<module> import …`.

| Module | Key functions |
|---|---|
| `paths.py` | `PROJECT_ROOT`, `STEP_DIR`, `ASSETS_DIR`, `MODELS_DIR`, `EMBEDDINGS_DIR`, `PROMPTS_PATH`, `OPENAI_KEY_PATH`, `OPENAI_MODEL`, `TOP_N_DOCS` |
| `openai_client.py` | `load_prompts()`, `make_client()`, `ask_gpt()`, `fmt_prompt()` |
| `tables.py` | `load_topic_corpus()`, `load_topic_names()`, `save_topic_names()` |
| `naming.py` | `get_representative_texts()`, `name_topics()` (Part 1) |
| `global_rename.py` | `build_rename_tool()`, `build_user_message()`, `build_system_prompt()`, `rename_topics_global()` (Part 2) |

Paths in `paths.py` are resolved from the module's own location, so the
notebooks work regardless of the kernel's working directory, and `prompts.yaml`
/ `openai.key` are always found at the step root.

## Notes on the pipeline

### Part 1 — per-cluster naming (`get_topic_names_part1.ipynb`)

1. **Load data** — topic info, document-level topic assignments, and the corpus
   text (merged on the corpus ID column).
2. **Select representative documents** — for each non-outlier cluster, take the
   top `TOP_N_DOCS` documents and concatenate their texts (truncated to stay
   within the context limit).
3. **Generate** — three LLM rounds per cluster: a raw description, an enhanced
   synthesis, then a short name.
4. **Save** — `<prefix>_topic_names.txt` with columns `topic`, `name`,
   `description`, `raw_description`.

### Part 2 — global renaming (`get_topic_names_part2.ipynb`)

Per-cluster names are produced in isolation and can collide. This pass sends
*all* topics to the LLM in a single call and uses **OpenAI function calling** to
enforce a structured response — one unique name per topic. It validates that
every topic received a name and that no two names are identical, then overwrites
`<prefix>_topic_names.txt`, adding a `global_name` column.
